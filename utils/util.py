import os
import logging, math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import torch.distributed as dist
import scipy.ndimage as nd
import matplotlib.pyplot as plt
from skimage import measure

from utils.networks_other import init_weights

def evaluate_3d(loader, model, cfg, ifdist=None, val_mode=None):
    print(f"{val_mode} Validation begin")
    model.eval()
    total_samples = 0
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        with torch.no_grad(): 
            all_mDice_organ = 0
            for img, mask, id in tqdm(loader):
                img, mask = img.cuda(), mask.squeeze(0).cuda()
                if total_samples==0:
                    dice_class_all = torch.zeros((cfg['nclass']-1,), device=img.device)
                total_samples += 1
                dice_class = torch.zeros((cfg['nclass']-1,), device=img.device)      # dismiss background
                total_metric = torch.zeros((cfg['nclass']-1,), device=img.device)    # dismiss background
                b, c, d, h, w = img.shape
                score_map = torch.zeros((cfg['nclass'], ) + torch.Size([d, h, w]), device=img.device)
                count_map = torch.zeros(img.shape[2:], device=img.device)
                patch_d, patch_h, patch_w = cfg['val_patch_size']
                # ========================= new =========================
                num_h = math.ceil(h / patch_h)
                overlap_h = (patch_h * num_h - h) // (num_h - 1) if num_h > 1 else 0
                stride_h = patch_h - overlap_h
                num_w = math.ceil(w / patch_w)
                overlap_w = (patch_w * num_w - w) // (num_w - 1) if num_w > 1 else 0
                stride_w = patch_w - overlap_w
                num_d = math.ceil(d / patch_d)
                overlap_d = (patch_d * num_d - d) // (num_d - 1) if num_d > 1 else 0
                stride_d = patch_d - overlap_d
                d_starts = torch.arange(0, num_d, device=img.device) * stride_d
                d_starts = torch.clamp(d_starts, max=d-patch_d)
                h_starts = torch.arange(0, num_h, device=img.device) * stride_h
                h_starts = torch.clamp(h_starts, max=h-patch_h)
                w_starts = torch.arange(0, num_w, device=img.device) * stride_w
                w_starts = torch.clamp(w_starts, max=w-patch_w)
                d_idx = d_starts[:, None] + torch.arange(patch_d, device=img.device)
                h_idx = h_starts[:, None] + torch.arange(patch_h, device=img.device)   
                w_idx = w_starts[:, None] + torch.arange(patch_w, device=img.device)  

                # Here are a more efficent way if GPU memory is avaible, we can use vector to val.
                # img_patch = img[:, :, d_idx[:, None, None, :, None, None], h_idx[None, :, None, None,  :, None], w_idx[None, None, :,None, None, :]] 
                # pred_patch_logits = model(img_patch)
                # pred_patch_logits = torch.softmax(pred_patch_logits, dim=1)
                # pred_mask = pred_patch_logits.argmax(dim=1)
                for dd in d_idx:
                    for hh in h_idx:
                        for ww in w_idx:
                            d_id = dd.unsqueeze(1).unsqueeze(2).expand(-1, patch_h, patch_w)
                            h_id = hh.unsqueeze(0).unsqueeze(2).expand(patch_d, -1, patch_w)  
                            w_id = ww.unsqueeze(0).unsqueeze(1).expand(patch_d, patch_h, -1)    
                            input = img[:, :, d_id, h_id, w_id]
                            pred_patch_logits = model(input)
                            pred_patch = torch.softmax(pred_patch_logits, dim=1)
                            score_map[:, d_id, h_id, w_id] += pred_patch.squeeze(0)
                            count_map[d_id, h_id, w_id] += 1
                            # if pred_all.max() == 0:
                            #     pred_all = pred_patch
                            # else:
                            #     pred_all = torch.cat((pred_all, pred_patch), dim=0)                    # get patch pred mean result
                score_map /= count_map
                pred_mask = score_map.argmax(dim=0)

                classes = torch.arange(1, cfg['nclass'], device=mask.device)                             # (nclass-1, 1, 1, 1)
                mask_exp = mask.unsqueeze(0) == classes.unsqueeze(1).unsqueeze(2).unsqueeze(3)           # (nclass-1, D, H, W)
                pred_exp = pred_mask.unsqueeze(0) == classes.unsqueeze(1).unsqueeze(2).unsqueeze(3)      # (nclass-1, D, H, W)
                mask_exp, pred_exp = mask_exp.view(cfg['nclass']-1, -1), pred_exp.view(cfg['nclass']-1, -1)
                intersection = (mask_exp * pred_exp).sum(dim=1)
                union = mask_exp.sum(dim=1) + pred_exp.sum(dim=1)
                
                dice_class += (2. * intersection) / (union + 1e-7)
                mDice_organ = dice_class.mean()
                all_mDice_organ += mDice_organ
                dice_class_all += dice_class
            total_samples_tensor = torch.tensor(total_samples).cuda()
    if ifdist:
        dist.all_reduce(all_mDice_organ)
        dist.all_reduce(dice_class_all)
        dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
        all_mDice_organ /= total_samples_tensor
        dice_class_all /= total_samples_tensor
    else:
        all_mDice_organ /= total_samples_tensor
        dice_class_all /= total_samples_tensor

    print(f"{val_mode} Validation end")
    return all_mDice_organ, dice_class_all


class conv2DBatchNorm(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True):
        super(conv2DBatchNorm, self).__init__()

        self.cb_unit = nn.Sequential(nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                               padding=padding, stride=stride, bias=bias),
                                 nn.BatchNorm2d(int(n_filters)),)

    def forward(self, inputs):
        outputs = self.cb_unit(inputs)
        return outputs


class deconv2DBatchNorm(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True):
        super(deconv2DBatchNorm, self).__init__()

        self.dcb_unit = nn.Sequential(nn.ConvTranspose2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                               padding=padding, stride=stride, bias=bias),
                                 nn.BatchNorm2d(int(n_filters)),)

    def forward(self, inputs):
        outputs = self.dcb_unit(inputs)
        return outputs


class conv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size,  stride, padding, bias=True):
        super(conv2DBatchNormRelu, self).__init__()

        self.cbr_unit = nn.Sequential(nn.Conv2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                                padding=padding, stride=stride, bias=bias),
                                 nn.BatchNorm2d(int(n_filters)),
                                 nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs


class deconv2DBatchNormRelu(nn.Module):
    def __init__(self, in_channels, n_filters, k_size, stride, padding, bias=True):
        super(deconv2DBatchNormRelu, self).__init__()

        self.dcbr_unit = nn.Sequential(nn.ConvTranspose2d(int(in_channels), int(n_filters), kernel_size=k_size,
                                                padding=padding, stride=stride, bias=bias),
                                 nn.BatchNorm2d(int(n_filters)),
                                 nn.ReLU(inplace=True),)

    def forward(self, inputs):
        outputs = self.dcbr_unit(inputs)
        return outputs


class unetConv2(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, n=2, ks=3, stride=1, padding=1):
        super(unetConv2, self).__init__()
        self.n = n
        self.ks = ks
        self.stride = stride
        self.padding = padding
        s = stride
        p = padding
        if is_batchnorm:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.BatchNorm2d(out_size),
                                     nn.ReLU(inplace=True),)
                setattr(self, 'conv%d'%i, conv)
                in_size = out_size

        else:
            for i in range(1, n+1):
                conv = nn.Sequential(nn.Conv2d(in_size, out_size, ks, s, p),
                                     nn.ReLU(inplace=True),)
                setattr(self, 'conv%d'%i, conv)
                in_size = out_size

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        x = inputs
        for i in range(1, self.n+1):
            conv = getattr(self, 'conv%d'%i)
            x = conv(x)

        return x


class UnetConv3(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, kernel_size=(3,3,1), padding_size=(1,1,0), init_stride=(1,1,1)):
        super(UnetConv3, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.InstanceNorm3d(out_size),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.InstanceNorm3d(out_size),
                                       nn.ReLU(inplace=True),)
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.ReLU(inplace=True),)

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        return outputs


class FCNConv3(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm, kernel_size=(3,3,1), padding_size=(1,1,0), init_stride=(1,1,1)):
        super(FCNConv3, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.InstanceNorm3d(out_size),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.InstanceNorm3d(out_size),
                                       nn.ReLU(inplace=True),)
            self.conv3 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.InstanceNorm3d(out_size),
                                       nn.ReLU(inplace=True),)
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, init_stride, padding_size),
                                       nn.ReLU(inplace=True),)
            self.conv2 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.ReLU(inplace=True),)
            self.conv3 = nn.Sequential(nn.Conv3d(out_size, out_size, kernel_size, 1, padding_size),
                                       nn.ReLU(inplace=True),)

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        outputs = self.conv3(outputs)
        return outputs


class UnetGatingSignal3(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm):
        super(UnetGatingSignal3, self).__init__()
        self.fmap_size = (4, 4, 4)

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, in_size//2, (1,1,1), (1,1,1), (0,0,0)),
                                       nn.InstanceNorm3d(in_size//2),
                                       nn.ReLU(inplace=True),
                                       nn.AdaptiveAvgPool3d(output_size=self.fmap_size),
                                       )
            self.fc1 = nn.Linear(in_features=(in_size//2) * self.fmap_size[0] * self.fmap_size[1] * self.fmap_size[2],
                                 out_features=out_size, bias=True)
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, in_size//2, (1,1,1), (1,1,1), (0,0,0)),
                                       nn.ReLU(inplace=True),
                                       nn.AdaptiveAvgPool3d(output_size=self.fmap_size),
                                       )
            self.fc1 = nn.Linear(in_features=(in_size//2) * self.fmap_size[0] * self.fmap_size[1] * self.fmap_size[2],
                                 out_features=out_size, bias=True)

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        batch_size = inputs.size(0)
        outputs = self.conv1(inputs)
        outputs = outputs.view(batch_size, -1)
        outputs = self.fc1(outputs)
        return outputs


class UnetGridGatingSignal3(nn.Module):
    def __init__(self, in_size, out_size, kernel_size=(1,1,1), is_batchnorm=True):
        super(UnetGridGatingSignal3, self).__init__()

        if is_batchnorm:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, (1,1,1), (0,0,0)),
                                       nn.InstanceNorm3d(out_size),
                                       nn.ReLU(inplace=True),
                                       )
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size, (1,1,1), (0,0,0)),
                                       nn.ReLU(inplace=True),
                                       )

        # initialise the blocks
        for m in self.children():
            init_weights(m, init_type='kaiming')

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs


class unetUp(nn.Module):
    def __init__(self, in_size, out_size, is_deconv):
        super(unetUp, self).__init__()
        self.conv = unetConv2(in_size, out_size, False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=4, stride=2, padding=1)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('unetConv2') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


class UnetUp3(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, is_batchnorm=True):
        super(UnetUp3, self).__init__()
        if is_deconv:
            self.conv = UnetConv3(in_size, out_size, is_batchnorm)
            self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=(4,4,1), stride=(2,2,1), padding=(1,1,0))
        else:
            self.conv = UnetConv3(in_size+out_size, out_size, is_batchnorm)
            self.up = nn.Upsample(scale_factor=(2, 2, 1), mode='trilinear')

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('UnetConv3') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2, 0]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


class UnetUp3_CT(nn.Module):
    def __init__(self, in_size, out_size, is_batchnorm=True):
        super(UnetUp3_CT, self).__init__()
        self.conv = UnetConv3(in_size + out_size, out_size, is_batchnorm, kernel_size=(3,3,3), padding_size=(1,1,1))
        self.up = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear')

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('UnetConv3') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2, 0]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1))


# Squeeze-and-Excitation Network
class SqEx(nn.Module):

    def __init__(self, n_features, reduction=6):
        super(SqEx, self).__init__()

        if n_features % reduction != 0:
            raise ValueError('n_features must be divisible by reduction (default = 4)')

        self.linear1 = nn.Linear(n_features, n_features // reduction, bias=False)
        self.nonlin1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(n_features // reduction, n_features, bias=False)
        self.nonlin2 = nn.Sigmoid()

    def forward(self, x):

        y = F.avg_pool3d(x, kernel_size=x.size()[2:5])
        y = y.permute(0, 2, 3, 4, 1)
        y = self.nonlin1(self.linear1(y))
        y = self.nonlin2(self.linear2(y))
        y = y.permute(0, 4, 1, 2, 3)
        y = x * y
        return y

class UnetUp3_SqEx(nn.Module):
    def __init__(self, in_size, out_size, is_deconv, is_batchnorm):
        super(UnetUp3_SqEx, self).__init__()
        if is_deconv:
            self.sqex = SqEx(n_features=in_size+out_size)
            self.conv = UnetConv3(in_size, out_size, is_batchnorm)
            self.up = nn.ConvTranspose3d(in_size, out_size, kernel_size=(4,4,1), stride=(2,2,1), padding=(1,1,0))
        else:
            self.sqex = SqEx(n_features=in_size+out_size)
            self.conv = UnetConv3(in_size+out_size, out_size, is_batchnorm)
            self.up = nn.Upsample(scale_factor=(2, 2, 1), mode='trilinear')

        # initialise the blocks
        for m in self.children():
            if m.__class__.__name__.find('UnetConv3') != -1: continue
            init_weights(m, init_type='kaiming')

    def forward(self, inputs1, inputs2):
        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2, 0]
        outputs1 = F.pad(inputs1, padding)
        concat = torch.cat([outputs1, outputs2], 1)
        gated  = self.sqex(concat)
        return self.conv(gated)

class residualBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, n_filters, stride=1, downsample=None):
        super(residualBlock, self).__init__()

        self.convbnrelu1 = conv2DBatchNormRelu(in_channels, n_filters, 3,  stride, 1, bias=False)
        self.convbn2 = conv2DBatchNorm(n_filters, n_filters, 3, 1, 1, bias=False)
        self.downsample = downsample
        self.stride = stride
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.convbnrelu1(x)
        out = self.convbn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


class residualBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, n_filters, stride=1, downsample=None):
        super(residualBottleneck, self).__init__()
        self.convbn1 = nn.Conv2DBatchNorm(in_channels,  n_filters, k_size=1, bias=False)
        self.convbn2 = nn.Conv2DBatchNorm(n_filters,  n_filters, k_size=3, padding=1, stride=stride, bias=False)
        self.convbn3 = nn.Conv2DBatchNorm(n_filters,  n_filters * 4, k_size=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.convbn1(x)
        out = self.convbn2(out)
        out = self.convbn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out




class SeqModelFeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(SeqModelFeatureExtractor, self).__init__()

        self.submodule = submodule
        self.extracted_layers = extracted_layers

    def forward(self, x):
        outputs = []
        for name, module in self.submodule._modules.items():
            x = module(x)
            if name in self.extracted_layers:
                outputs += [x]
        return outputs + [x]


class HookBasedFeatureExtractor(nn.Module):
    def __init__(self, submodule, layername, upscale=False):
        super(HookBasedFeatureExtractor, self).__init__()

        self.submodule = submodule
        self.submodule.eval()
        self.layername = layername
        self.outputs_size = None
        self.outputs = None
        self.inputs = None
        self.inputs_size = None
        self.upscale = upscale

    def get_input_array(self, m, i, o):
        if isinstance(i, tuple):
            self.inputs = [i[index].data.clone() for index in range(len(i))]
            self.inputs_size = [input.size() for input in self.inputs]
        else:
            self.inputs = i.data.clone()
            self.inputs_size = self.input.size()
        print('Input Array Size: ', self.inputs_size)

    def get_output_array(self, m, i, o):
        if isinstance(o, tuple):
            self.outputs = [o[index].data.clone() for index in range(len(o))]
            self.outputs_size = [output.size() for output in self.outputs]
        else:
            self.outputs = o.data.clone()
            self.outputs_size = self.outputs.size()
        print('Output Array Size: ', self.outputs_size)

    def rescale_output_array(self, newsize):
        us = nn.Upsample(size=newsize[2:], mode='bilinear')
        if isinstance(self.outputs, list):
            for index in range(len(self.outputs)): self.outputs[index] = us(self.outputs[index]).data()
        else:
            self.outputs = us(self.outputs).data()

    def forward(self, x):
        target_layer = self.submodule._modules.get(self.layername)

        # Collect the output tensor
        h_inp = target_layer.register_forward_hook(self.get_input_array)
        h_out = target_layer.register_forward_hook(self.get_output_array)
        self.submodule(x)
        h_inp.remove()
        h_out.remove()

        # Rescale the feature-map if it's required
        if self.upscale: self.rescale_output_array(x.size())

        return self.inputs, self.outputs


class UnetDsv3(nn.Module):
    def __init__(self, in_size, out_size, scale_factor):
        super(UnetDsv3, self).__init__()
        self.dsv = nn.Sequential(nn.Conv3d(in_size, out_size, kernel_size=1, stride=1, padding=0),
                                 nn.Upsample(scale_factor=scale_factor, mode='trilinear'), )

    def forward(self, input):
        return self.dsv(input)
    

def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames if filename.endswith(suffix)]

def get_cityscapes_labels():
    return np.array([
        # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32]])

def get_pascal_labels():
    """Load the mapping that associates pascal classes with label colors
    Returns:
        np.ndarray with dimensions (21, 3)
    """
    return np.asarray([[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                       [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                       [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                       [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                       [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                       [0, 64, 128]])


def encode_segmap(mask):
    """Encode segmentation label images as pascal classes
    Args:
        mask (np.ndarray): raw segmentation label image of dimension
          (M, N, 3), in which the Pascal classes are encoded as colours.
    Returns:
        (np.ndarray): class map with dimensions (M,N), where the value at
        a given location is the integer denoting the class index.
    """
    mask = mask.astype(int)
    label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
    for ii, label in enumerate(get_pascal_labels()):
        label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = ii
    label_mask = label_mask.astype(int)
    return label_mask


def decode_seg_map_sequence(label_masks, dataset='pascal'):
    rgb_masks = []
    for label_mask in label_masks:
        rgb_mask = decode_segmap(label_mask, dataset)
        rgb_masks.append(rgb_mask)
    rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
    return rgb_masks

def decode_segmap(label_mask, dataset, plot=False):
    """Decode segmentation class labels into a color image
    Args:
        label_mask (np.ndarray): an (M,N) array of integer values denoting
          the class label at each spatial location.
        plot (bool, optional): whether to show the resulting color image
          in a figure.
    Returns:
        (np.ndarray, optional): the resulting decoded color image.
    """
    if dataset == 'pascal':
        n_classes = 21
        label_colours = get_pascal_labels()
    elif dataset == 'cityscapes':
        n_classes = 19
        label_colours = get_cityscapes_labels()
    else:
        raise NotImplementedError

    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for ll in range(0, n_classes):
        r[label_mask == ll] = label_colours[ll, 0]
        g[label_mask == ll] = label_colours[ll, 1]
        b[label_mask == ll] = label_colours[ll, 2]
    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
    rgb[:, :, 0] = r / 255.0
    rgb[:, :, 1] = g / 255.0
    rgb[:, :, 2] = b / 255.0
    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb

def generate_param_report(logfile, param):
    log_file = open(logfile, 'w')
    # for key, val in param.items():
    #     log_file.write(key + ':' + str(val) + '\n')
    log_file.write(str(param))
    log_file.close()

def cross_entropy2d(logit, target, ignore_index=255, weight=None, size_average=True, batch_average=True):
    n, c, h, w = logit.size()
    # logit = logit.permute(0, 2, 3, 1)
    target = target.squeeze(1)
    if weight is None:
        criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, size_average=False)
    else:
        criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(weight)).float().cuda(), ignore_index=ignore_index, size_average=False)
    loss = criterion(logit, target.long())

    if size_average:
        loss /= (h * w)

    if batch_average:
        loss /= n

    return loss

def lr_poly(base_lr, iter_, max_iter=100, power=0.9):
    return base_lr * ((1 - float(iter_) / max_iter) ** power)


def get_iou(pred, gt, n_classes=21):
    total_iou = 0.0
    for i in range(len(pred)):
        pred_tmp = pred[i]
        gt_tmp = gt[i]

        intersect = [0] * n_classes
        union = [0] * n_classes
        for j in range(n_classes):
            match = (pred_tmp == j) + (gt_tmp == j)

            it = torch.sum(match == 2).item()
            un = torch.sum(match > 0).item()

            intersect[j] += it
            union[j] += un

        iou = []
        for k in range(n_classes):
            if union[k] == 0:
                continue
            iou.append(intersect[k] / union[k])

        img_iou = (sum(iou) / len(iou))
        total_iou += img_iou

    return total_iou

def get_dice(pred, gt):
    total_dice = 0.0
    pred = pred.long()
    gt = gt.long()
    for i in range(len(pred)):
        pred_tmp = pred[i]
        gt_tmp = gt[i]
        dice = 2.0*torch.sum(pred_tmp*gt_tmp).item()/(1.0+torch.sum(pred_tmp**2)+torch.sum(gt_tmp**2)).item()
        print(dice)
        total_dice += dice

    return total_dice

def get_mc_dice(pred, gt, num=2):
    # num is the total number of classes, include the background
    total_dice = np.zeros(num-1)
    pred = pred.long()
    gt = gt.long()
    for i in range(len(pred)):
        for j in range(1, num):
            pred_tmp = (pred[i]==j)
            gt_tmp = (gt[i]==j)
            dice = 2.0*torch.sum(pred_tmp*gt_tmp).item()/(1.0+torch.sum(pred_tmp**2)+torch.sum(gt_tmp**2)).item()
            total_dice[j-1] +=dice
    return total_dice

def post_processing(prediction):
    prediction = nd.binary_fill_holes(prediction)
    label_cc, num_cc = measure.label(prediction,return_num=True)
    total_cc = np.sum(prediction)
    measure.regionprops(label_cc)
    for cc in range(1,num_cc+1):
        single_cc = (label_cc==cc)
        single_vol = np.sum(single_cc)
        if single_vol/total_cc<0.2:
            prediction[single_cc]=0

    return prediction


def count_params(model):
    param_num = sum(p.numel() for p in model.parameters())
    return param_num / 1e6


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.ndim in [1, 2, 3]
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def intersectionAndUnion_3d(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert output.ndim in [1, 2, 3, 4]
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K + 1))
    area_output, _ = np.histogram(output, bins=np.arange(K + 1))
    area_target, _ = np.histogram(target, bins=np.arange(K + 1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


logs = set()

# 需要修改
def focal_loss(inputs, targets, gamma=2.0, alpha=None, reduction='mean'):
    """
    多分类的 Focal Loss 实现。

    Args:
        inputs (torch.Tensor): 模型输出的 logits，形状为 (N, C)，其中 N 是 batch size，C 是类别数。
        targets (torch.Tensor): 真实的类别标签，形状为 (N,)，每个元素是 0 到 C-1 的整数。
        gamma (float): Focusing 参数。
        alpha (torch.Tensor, optional): 每个类别的权重，形状为 (C,)。默认为 None。
        reduction (str): 损失的 reduction 方式，可以是 'none' | 'mean' | 'sum'。默认为 'mean'。

    Returns:
        torch.Tensor: 计算得到的 Focal Loss。
    """
    # 确保输入是概率分布 (通过 softmax)
    probs = F.softmax(inputs, dim=1)

    # 获取每个样本的真实类别对应的概率
    probs_at_target = probs.gather(1, targets.unsqueeze(1)).squeeze(1)

    # 计算 Focal Loss 的调制因子
    focal_weight = (1 - probs_at_target) ** gamma

    # 计算标准的交叉熵损失
    ce_loss = F.cross_entropy(inputs, targets, reduction='none')

    # 将 Focal Weight 应用到交叉熵损失
    focal_loss = focal_weight * ce_loss

    # 应用类别权重 (alpha)
    if alpha is not None:
        # 确保 alpha 的形状正确
        alpha_ = alpha.gather(0, targets)
        focal_loss = alpha_ * focal_loss

    # 根据 reduction 参数返回损失
    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    elif reduction == 'none':
        return focal_loss
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

class GADice(nn.Module):
    def __init__(self, GA=True):
        self.GA = GA
        super(GADice, self).__init__()

    def _one_hot_encoder(self, input_tensor):        
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, cls, score, target, weighted_pixel_map=None):
        target = target.float()
        if weighted_pixel_map is not None:
            target = target * weighted_pixel_map
        smooth = 1e-10

        intersection = 2 * torch.sum(score * target) + smooth
        union = torch.sum(score * score) + torch.sum(target * target) + smooth
        loss = 1 - intersection / union
        
        return loss

    def forward(self, inputs, target, argmax=False, one_hot=True, weight=None, softmax=False, weighted_pixel_map=None):
        self.n_classes = inputs.size()[1]
        if len(inputs.size()) == len(target.size())+1:
            target = target.unsqueeze(1)
        
        if softmax:
            inputs = F.softmax(inputs, dim=1)
        if argmax:
            target = torch.argmax(target, dim=1)
        if one_hot:
            target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        
        assert inputs.size() == target.size(), 'predict & target shape do not match'
        smooth = 1e-10
        loss = 0.0
        for i in range(0, self.n_classes):
            if torch.sum(target[:, i]) > 0:
                dice_loss = self._dice_loss(i, inputs[:, i], target[:, i], weighted_pixel_map)
            else:
                if self.GA:
                    beta = inputs[:, i] / (torch.sum(1 - target[:, i]))
                    dice_loss = torch.sum(beta.detach() * inputs[:, i])                
            loss += dice_loss * weight[i]
        
        return loss / self.n_classes
        
        
        
        
class GACE(torch.nn.CrossEntropyLoss):
    def __init__(self, weight=None, ignore_index=-100, k=10, gama=0.5):
        self.k = k
        self.gama = gama
        super(GACE, self).__init__(weight, False, ignore_index, reduce=False)

    def forward(self, inp, target):
        target = target.long()
        self.n_classes = inp.size()[1]

        i0 = 1
        i1 = 2
        while i1 < len(inp.shape): # this is ugly but torch only allows to transpose two axes at once
            inp = inp.transpose(i0, i1)
            i0 += 1
            i1 += 1

        inp = inp.contiguous()
        inp = inp.view(-1, self.n_classes)
        target = target.view(-1,)
        res = super(GACE, self).forward(inp, target)      
  
  
        n_instance = np.prod(res.shape)
        res, indices = torch.topk(res.view((-1, )), int(n_instance * self.k / 100), sorted=False)      
        target = torch.gather(target, 0, indices)        
        assert res.size() == target.size(), 'predict & target shape do not match'
        
        bg_w = np.power(int(n_instance * self.k / 100), self.gama)
        loss = 0.0
        smooth = 1e-10
        for i in range(0, self.n_classes):        
            target_cls = (target == i).float()  
            w = torch.pow(torch.sum(target_cls) + smooth, 1-self.gama) * bg_w
            loss_cls = torch.sum(res * target_cls) / (w + smooth)
            loss += loss_cls

        return loss
    
    
# alpha值的确认
def get_class_weights(targets, num_classes):
    """计算类别权重，与类别频率成反比."""
    class_counts = torch.zeros(num_classes)
    for t in targets:
        class_counts[t] += 1
    total_samples = len(targets)
    weights = total_samples / (class_counts + 1e-6) # 加一个小的 epsilon 防止除以零
    return weights

# 假设你的训练集目标标签是 'train_targets'，类别总数为 'num_classes'
train_targets = torch.randint(0, 14, (1000,)) # 示例目标标签
num_classes = 14
alpha = get_class_weights(train_targets, num_classes)


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target, index):
        # target = target.float()
        # smooth = 1e-7
        # intersect = torch.sum(score * target)
        # y_sum = torch.sum(target * target)
        # z_sum = torch.sum(score * score)
        # loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        # loss = 1 - loss
        # return loss
        
        # mask1 = target != index
        # mask2 = score != index
        # mask_1 = target == index
        # mask_2 = score == index
        # target[mask1] = 0
        # score[mask2] = 0
        # target[mask_1] = 1
        # score[mask_2] = 1
        # target = target.float()
        # score = score.float()
        # smooth = 1e-7
        # intersect = torch.sum(score * target)
        # y_sum = torch.sum(target)
        # z_sum = torch.sum(score)
        # loss = (2 * intersect) / (z_sum + y_sum)
        # loss = 1 - loss
        # return loss
        mask_positive_target = target == index
        mask_positive_score = score == index

        # 创建新的张量，避免原地修改
        new_target = torch.zeros_like(target, dtype=torch.float)
        new_score = torch.zeros_like(score, dtype=torch.float)

        new_target[mask_positive_target] = 1.0
        new_score[mask_positive_score] = 1.0

        smooth = 1e-7
        intersect = torch.sum(new_score * new_target)
        y_sum = torch.sum(new_target)
        z_sum = torch.sum(new_score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=True):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        # target = self._one_hot_encoder(target)
        predicted_classes = torch.argmax(inputs, dim=1)
        if weight is None:
            weight = [1] * self.n_classes
        assert predicted_classes.size() == target.size(), 'predict & target shape do not match'
        # class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(predicted_classes, target, i)
            # class_wise_dice.append(1.0 - dice.item())
            loss += dice * (1.0 - weight[i]) * self.n_classes
        return loss / self.n_classes
    

def init_log(name, level=logging.INFO, log_file=None, console_output=True):
    if (name, level, log_file, console_output) in logs:
        return logging.getLogger(name)
    # logs.add((name, level))
    logs.add((name, level, log_file, console_output))
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    if not logger.handlers:
        format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
        formatter = logging.Formatter(format_str)
        if console_output:
            ch = logging.StreamHandler()
            ch.setLevel(level)
            if "SLURM_PROCID" in os.environ:
                rank = int(os.environ["SLURM_PROCID"])
                logger.addFilter(lambda record: rank == 0)
            # else:
            #     rank = 0
            ch.setFormatter(formatter)
            logger.addHandler(ch)
        
        if log_file:
            fh = logging.FileHandler(log_file)
            fh.setLevel(level)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
    return logger


def color_map(dataset='pascal'):
    cmap = np.zeros((256, 3), dtype='uint8')

    if dataset == 'pascal' or dataset == 'coco':
        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        for i in range(256):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7-j)
                g = g | (bitget(c, 1) << 7-j)
                b = b | (bitget(c, 2) << 7-j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

    elif dataset == 'cityscapes':
        cmap[0] = np.array([128, 64, 128])
        cmap[1] = np.array([244, 35, 232])
        cmap[2] = np.array([70, 70, 70])
        cmap[3] = np.array([102, 102, 156])
        cmap[4] = np.array([190, 153, 153])
        cmap[5] = np.array([153, 153, 153])
        cmap[6] = np.array([250, 170, 30])
        cmap[7] = np.array([220, 220, 0])
        cmap[8] = np.array([107, 142, 35])
        cmap[9] = np.array([152, 251, 152])
        cmap[10] = np.array([70, 130, 180])
        cmap[11] = np.array([220, 20, 60])
        cmap[12] = np.array([255, 0, 0])
        cmap[13] = np.array([0, 0, 142])
        cmap[14] = np.array([0, 0, 70])
        cmap[15] = np.array([0, 60, 100])
        cmap[16] = np.array([0, 80, 100])
        cmap[17] = np.array([0, 0, 230])
        cmap[18] = np.array([119, 11, 32])

    elif dataset == 'ade20k':
        cmap[0] = np.array([120, 120, 120])
        cmap[1] = np.array([180, 120, 120])
        cmap[2] = np.array([6, 230, 230])
        cmap[3] = np.array([80, 50, 50])
        cmap[4] = np.array([4, 200, 3])
        cmap[5] = np.array([120, 120, 80])
        cmap[6] = np.array([140, 140, 140])
        cmap[7] = np.array([204, 5, 255])
        cmap[8] = np.array([230, 230, 230])
        cmap[9] = np.array([4, 250, 7])
        cmap[10] = np.array([224, 5, 255])
        cmap[11] = np.array([235, 255, 7])
        cmap[12] = np.array([150, 5, 61])
        cmap[13] = np.array([120, 120, 70])
        cmap[14] = np.array([8, 255, 51])
        cmap[15] = np.array([255, 6, 82])
        cmap[16] = np.array([143, 255, 140])
        cmap[17] = np.array([204, 255, 4])
        cmap[18] = np.array([255, 51, 7])
        cmap[19] = np.array([204, 70, 3])
        cmap[20] = np.array([0, 102, 200])
        cmap[21] = np.array([61, 230, 250])
        cmap[22] = np.array([255, 6, 51])
        cmap[23] = np.array([11, 102, 255])
        cmap[24] = np.array([255, 7, 71])
        cmap[25] = np.array([255, 9, 224])
        cmap[26] = np.array([9, 7, 230])
        cmap[27] = np.array([220, 220, 220])
        cmap[28] = np.array([255, 9, 92])
        cmap[29] = np.array([112, 9, 255])
        cmap[30] = np.array([8, 255, 214])
        cmap[31] = np.array([7, 255, 224])
        cmap[32] = np.array([255, 184, 6])
        cmap[33] = np.array([10, 255, 71])
        cmap[34] = np.array([255, 41, 10])
        cmap[35] = np.array([7, 255, 255])
        cmap[36] = np.array([224, 255, 8])
        cmap[37] = np.array([102, 8, 255])
        cmap[38] = np.array([255, 61, 6])
        cmap[39] = np.array([255, 194, 7])
        cmap[40] = np.array([255, 122, 8])
        cmap[41] = np.array([0, 255, 20])
        cmap[42] = np.array([255, 8, 41])
        cmap[43] = np.array([255, 5, 153])
        cmap[44] = np.array([6, 51, 255])
        cmap[45] = np.array([235, 12, 255])
        cmap[46] = np.array([160, 150, 20])
        cmap[47] = np.array([0, 163, 255])
        cmap[48] = np.array([140, 140, 140])
        cmap[49] = np.array([250, 10, 15])
        cmap[50] = np.array([20, 255, 0])
        cmap[51] = np.array([31, 255, 0])
        cmap[52] = np.array([255, 31, 0])
        cmap[53] = np.array([255, 224, 0])
        cmap[54] = np.array([153, 255, 0])
        cmap[55] = np.array([0, 0, 255])
        cmap[56] = np.array([255, 71, 0])
        cmap[57] = np.array([0, 235, 255])
        cmap[58] = np.array([0, 173, 255])
        cmap[59] = np.array([31, 0, 255])
        cmap[60] = np.array([11, 200, 200])
        cmap[61] = np.array([255, 82, 0])
        cmap[62] = np.array([0, 255, 245])
        cmap[63] = np.array([0, 61, 255])
        cmap[64] = np.array([0, 255, 112])
        cmap[65] = np.array([0, 255, 133])
        cmap[66] = np.array([255, 0, 0])
        cmap[67] = np.array([255, 163, 0])
        cmap[68] = np.array([255, 102, 0])
        cmap[69] = np.array([194, 255, 0])
        cmap[70] = np.array([0, 143, 255])
        cmap[71] = np.array([51, 255, 0])
        cmap[72] = np.array([0, 82, 255])
        cmap[73] = np.array([0, 255, 41])
        cmap[74] = np.array([0, 255, 173])
        cmap[75] = np.array([10, 0, 255])
        cmap[76] = np.array([173, 255, 0])
        cmap[77] = np.array([0, 255, 153])
        cmap[78] = np.array([255, 92, 0])
        cmap[79] = np.array([255, 0, 255])
        cmap[80] = np.array([255, 0, 245])
        cmap[81] = np.array([255, 0, 102])
        cmap[82] = np.array([255, 173, 0])
        cmap[83] = np.array([255, 0, 20])
        cmap[84] = np.array([255, 184, 184])
        cmap[85] = np.array([0, 31, 255])
        cmap[86] = np.array([0, 255, 61])
        cmap[87] = np.array([0, 71, 255])
        cmap[88] = np.array([255, 0, 204])
        cmap[89] = np.array([0, 255, 194])
        cmap[90] = np.array([0, 255, 82])
        cmap[91] = np.array([0, 10, 255])
        cmap[92] = np.array([0, 112, 255])
        cmap[93] = np.array([51, 0, 255])
        cmap[94] = np.array([0, 194, 255])
        cmap[95] = np.array([0, 122, 255])
        cmap[96] = np.array([0, 255, 163])
        cmap[97] = np.array([255, 153, 0])
        cmap[98] = np.array([0, 255, 10])
        cmap[99] = np.array([255, 112, 0])
        cmap[100] = np.array([143, 255, 0])
        cmap[101] = np.array([82, 0, 255])
        cmap[102] = np.array([163, 255, 0])
        cmap[103] = np.array([255, 235, 0])
        cmap[104] = np.array([8, 184, 170])
        cmap[105] = np.array([133, 0, 255])
        cmap[106] = np.array([0, 255, 92])
        cmap[107] = np.array([184, 0, 255])
        cmap[108] = np.array([255, 0, 31])
        cmap[109] = np.array([0, 184, 255])
        cmap[110] = np.array([0, 214, 255])
        cmap[111] = np.array([255, 0, 112])
        cmap[112] = np.array([92, 255, 0])
        cmap[113] = np.array([0, 224, 255])
        cmap[114] = np.array([112, 224, 255])
        cmap[115] = np.array([70, 184, 160])
        cmap[116] = np.array([163, 0, 255])
        cmap[117] = np.array([153, 0, 255])
        cmap[118] = np.array([71, 255, 0])
        cmap[119] = np.array([255, 0, 163])
        cmap[120] = np.array([255, 204, 0])
        cmap[121] = np.array([255, 0, 143])
        cmap[122] = np.array([0, 255, 235])
        cmap[123] = np.array([133, 255, 0])
        cmap[124] = np.array([255, 0, 235])
        cmap[125] = np.array([245, 0, 255])
        cmap[126] = np.array([255, 0, 122])
        cmap[127] = np.array([255, 245, 0])
        cmap[128] = np.array([10, 190, 212])
        cmap[129] = np.array([214, 255, 0])
        cmap[130] = np.array([0, 204, 255])
        cmap[131] = np.array([20, 0, 255])
        cmap[132] = np.array([255, 255, 0])
        cmap[133] = np.array([0, 153, 255])
        cmap[134] = np.array([0, 41, 255])
        cmap[135] = np.array([0, 255, 204])
        cmap[136] = np.array([41, 0, 255])
        cmap[137] = np.array([41, 255, 0])
        cmap[138] = np.array([173, 0, 255])
        cmap[139] = np.array([0, 245, 255])
        cmap[140] = np.array([71, 0, 255])
        cmap[141] = np.array([122, 0, 255])
        cmap[142] = np.array([0, 255, 184])
        cmap[143] = np.array([0, 92, 255])
        cmap[144] = np.array([184, 255, 0])
        cmap[145] = np.array([0, 133, 255])
        cmap[146] = np.array([255, 214, 0])
        cmap[147] = np.array([25, 194, 194])
        cmap[148] = np.array([102, 255, 0])
        cmap[149] = np.array([92, 0, 255])

    return cmap
