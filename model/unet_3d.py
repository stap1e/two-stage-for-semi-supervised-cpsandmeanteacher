# -*- coding: utf-8 -*-
"""
An implementation of the 3D U-Net paper:
     Özgün Çiçek, Ahmed Abdulkadir, Soeren S. Lienkamp, Thomas Brox, Olaf Ronneberger:
     3D U-Net: Learning Dense Volumetric Segmentation from Sparse Annotation. 
     MICCAI (2) 2016: 424-432
Note that there are some modifications from the original paper, such as
the use of batch normalization, dropout, and leaky relu here.
The implementation is borrowed from: https://github.com/ozan-oktay/Attention-Gated-Networks
"""
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init
from utils.networks_other import init_weights
from utils.util import UnetConv3, UnetUp3, UnetUp3_CT


class unet_3D(nn.Module):

    def __init__(self, feature_scale=4, n_classes=21, is_deconv=True, in_channels=3, is_batchnorm=True):
        super(unet_3D, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = UnetConv3(self.in_channels, filters[0], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv2 = UnetConv3(filters[0], filters[1], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = UnetConv3(filters[1], filters[2], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4 = UnetConv3(filters[2], filters[3], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center = UnetConv3(filters[3], filters[4], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))

        # upsampling
        self.up_concat4 = UnetUp3_CT(filters[4], filters[3], is_batchnorm)
        self.up_concat3 = UnetUp3_CT(filters[3], filters[2], is_batchnorm)
        self.up_concat2 = UnetUp3_CT(filters[2], filters[1], is_batchnorm)
        self.up_concat1 = UnetUp3_CT(filters[1], filters[0], is_batchnorm)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0], n_classes, 1)

        self.dropout1 = nn.Dropout(p=0.3)
        self.dropout2 = nn.Dropout(p=0.3)
        
        self.binomial = torch.distributions.binomial.Binomial(probs=0.5)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')

    def forward(self, inputs, comp_drop=False):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)
        
        center = self.center(maxpool4)
        center = self.dropout1(center)
        
        if comp_drop:
            bs = center.shape[0]
            dropout_prob = 0.5
            num_kept = int(bs // 2 * (1 - dropout_prob))
            kept_indexes = torch.randperm(bs // 2)[:num_kept]
            
            dim1 =  center.shape[1]
            dropout_mask1_1 = self.binomial.sample((bs // 2, dim1)).cuda() * 2.0
            dropout_mask2_1 = 2.0 - dropout_mask1_1
            dropout_mask1_1[kept_indexes, :] = 1.0
            dropout_mask2_1[kept_indexes, :] = 1.0
            dropout_mask1 = torch.cat((dropout_mask1_1, dropout_mask2_1))
            center *= dropout_mask1.unsqueeze(2).unsqueeze(3).unsqueeze(4)
            dim2 =  conv4.shape[1]
            dropout_mask1_2 = self.binomial.sample((bs // 2, dim2)).cuda() * 2.0
            dropout_mask2_2 = 2.0 - dropout_mask1_2
            dropout_mask1_2[kept_indexes, :] = 1.0
            dropout_mask2_2[kept_indexes, :] = 1.0
            dropout_mask2 = torch.cat((dropout_mask1_2, dropout_mask2_2))
            conv4 *= dropout_mask2.unsqueeze(2).unsqueeze(3).unsqueeze(4)
            up4 = self.up_concat4(conv4, center)
            
            dim3 =  conv3.shape[1]
            dropout_mask1_3 = self.binomial.sample((bs // 2, dim3)).cuda() * 2.0
            dropout_mask2_3 = 2.0 - dropout_mask1_3
            dropout_mask1_3[kept_indexes, :] = 1.0
            dropout_mask2_3[kept_indexes, :] = 1.0
            dropout_mask3 = torch.cat((dropout_mask1_3, dropout_mask2_3))
            conv3 *= dropout_mask3.unsqueeze(2).unsqueeze(3).unsqueeze(4)
            up3 = self.up_concat3(conv3, up4)
            
            dim4 =  conv2.shape[1]
            dropout_mask1_4 = self.binomial.sample((bs // 2, dim4)).cuda() * 2.0
            dropout_mask2_4 = 2.0 - dropout_mask1_4
            dropout_mask1_4[kept_indexes, :] = 1.0
            dropout_mask2_4[kept_indexes, :] = 1.0
            dropout_mask4 = torch.cat((dropout_mask1_4, dropout_mask2_4))
            conv2 *= dropout_mask4.unsqueeze(2).unsqueeze(3).unsqueeze(4)
            up2 = self.up_concat2(conv2, up3)
            
            dim5 =  conv1.shape[1]
            dropout_mask1_5 = self.binomial.sample((bs // 2, dim5)).cuda() * 2.0
            dropout_mask2_5 = 2.0 - dropout_mask1_5
            dropout_mask1_5[kept_indexes, :] = 1.0
            dropout_mask2_5[kept_indexes, :] = 1.0
            dropout_mask5 = torch.cat((dropout_mask1_5, dropout_mask2_5))
            conv1 *= dropout_mask5.unsqueeze(2).unsqueeze(3).unsqueeze(4)
            up1 = self.up_concat1(conv1, up2)
            up1 = self.dropout2(up1)
            
            out = self.final(up1)
            return out
        
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)
        up1 = self.dropout2(up1)

        final = self.final(up1)

        return final

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p


class Encoder(nn.Module):
    def __init__(self, params):
        super(Encoder, self).__init__()

        self.params = params
        self.is_batchnorm = self.params['is_batchnorm']
        self.in_chns = self.params['in_chns']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        
        self.conv1 = UnetConv3(self.in_chns, self.ft_chns[0], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool1 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv2 = UnetConv3(self.ft_chns[0], self.ft_chns[1], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv3 = UnetConv3(self.ft_chns[1], self.ft_chns[2], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool3 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.conv4 = UnetConv3(self.ft_chns[2], self.ft_chns[3], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        self.maxpool4 = nn.MaxPool3d(kernel_size=(2, 2, 2))

        self.center = UnetConv3(self.ft_chns[3], self.ft_chns[4], self.is_batchnorm, kernel_size=(
            3, 3, 3), padding_size=(1, 1, 1))
        
        self.dropout = nn.Dropout(p=0.3)
        
    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)
        
        center = self.center(maxpool4)
        center = self.dropout(center)
        
        return [conv1, conv2, conv3, conv4, center]
        

# need change
class Decoder(nn.Module):
    def __init__(self, params, in_channels=3, is_batchnorm=True):
        super(Decoder, self).__init__()
        # self.is_batchnorm = is_batchnorm 
        self.params = params
        self.is_batchnorm = self.params['is_batchnorm']
        self.ft_chns = self.params['feature_chns']
        self.n_class = self.params['class_num']
        # upsampling
        self.up_concat4 = UnetUp3_CT(self.ft_chns[4], self.ft_chns[3], is_batchnorm)
        self.up_concat3 = UnetUp3_CT(self.ft_chns[3], self.ft_chns[2], is_batchnorm)
        self.up_concat2 = UnetUp3_CT(self.ft_chns[2], self.ft_chns[1], is_batchnorm)
        self.up_concat1 = UnetUp3_CT(self.ft_chns[1], self.ft_chns[0], is_batchnorm)

        # final conv (without any concat)
        self.final = nn.Conv3d(self.ft_chns[0], self.n_class, 1)
        self.dropout = nn.Dropout(p=0.3)

        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm3d):
                init_weights(m, init_type='kaiming')
                
    def forward(self, features):
        conv1, conv2, conv3, conv4, center = features[:]
        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)
        up1 = self.dropout(up1)

        final = self.final(up1)

        return final

    
class unet_3D_cps(nn.Module):
    def __init__(self, in_chns, class_num):
        super(unet_3D_cps, self).__init__()
        params = {'in_chns': in_chns,
                  'is_batchnorm': True,
                  'feature_scale': 4,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def forward(self, inputs, comp_drop=None):
        features = self.encoder(inputs)
        if comp_drop != None:
            for i in range(0, len(features)):
                features[i] = features[i] * comp_drop[i].unsqueeze(2).unsqueeze(3).unsqueeze(4)
            
            out = self.decoder(features)
            return out
        out = self.decoder(features)
        return out

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p
    

class unet_3D_mt(nn.Module):
    def __init__(self, in_chns, class_num):
        super(unet_3D_mt, self).__init__()
        params = {'in_chns': in_chns,
                  'is_batchnorm': True,
                  'feature_scale': 4,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}
        self.encoder = Encoder(params)
        self.decoder = Decoder(params)
        self.binomial = torch.distributions.binomial.Binomial(probs=0.5)

    def forward(self, inputs, comp_drop=None):
        features = self.encoder(inputs)
        if comp_drop:
            bs = features[0].shape[0]
            dropout_prob = 0.5
            num_kept = int(bs // 2 * (1 - dropout_prob))
            kept_indexes = torch.randperm(bs // 2)[:num_kept]
            for i in range(0, len(features)):   
                dim = features[i].shape[1]
                dropout_mask1 = self.binomial.sample((bs // 2, dim)).cuda() * 2.0
                dropout_mask2 = 2.0 - dropout_mask1
                dropout_mask1[kept_indexes, :] = 1.0
                dropout_mask2[kept_indexes, :] = 1.0
                dropout_mask = torch.cat((dropout_mask1, dropout_mask2))
                features[i] = features[i] * dropout_mask.unsqueeze(2).unsqueeze(3).unsqueeze(4)
            
            out = self.decoder(features)
            return out
        out = self.decoder(features)
        return out

    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)

        return log_p
    
    
def weights_init_normal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)


def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def sparse_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.sparse_(m.weight, sparsity=0.1)
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def init_weights(net, init_type='normal'):
    #print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
    
def main():
    inputs = np.random.randn(1, 3, 256, 256, 256)
    t = torch.Tensor(inputs)
    model1 = unet_3D(feature_scale=4, n_classes=5, is_deconv=True, in_channels=3, is_batchnorm=True)
    outputs = model1(t)
    print(outputs.shape)

if __name__ == '__main__':
    main()