""" based on guidednet train.py """
import argparse
import logging
import os
import random
import sys
import time, yaml, h5py
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms
from ssl_repo.brats2019 import (BraTS2019, RandomCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler)
from ssl_repo import losses, ramps
from utils.classes import CLASSES
from datasets.dataset import FlareDataset
from utils.util import evaluate_3d
from model.unet_3d import unet_3D_cps

os.environ["CUDA_VISIBLE_DEVICES"] = '7'
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='./configs/Your.yaml')
parser.add_argument('--base_dir', type=str, default='./27_FLARE2022')
parser.add_argument('--num', default=None, type=int)
parser.add_argument('--root_path', type=str, default='./27_FLARE2022', help='Name of Experiment')  
parser.add_argument('--exp', type=str, default='Your experiments description', help='experiment_name')
parser.add_argument('--model', type=str, default='unet_3D', help='model_name')
parser.add_argument('--dataset', type=str, default='flare22_ssl', help='dataset_name and without background')
parser.add_argument('--num_classes', type=int, default=14, help='number of class for FLARE22')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=4, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=2, help='labeled_batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--patch_size', type=list,  default=[64, 128, 128], help='patch size of network input')
parser.add_argument('--val_patch_size', type=list, default=[64, 160, 160],help='patch size of network input')
parser.add_argument('--val_xy', type=int, default=80,help='patch size of val network input')
parser.add_argument('--val_z', type=int, default=32,help='patch size of val network input')
parser.add_argument('--labeled_num', type=int, default=42, help='labeled data')
parser.add_argument('--data_num', type=int, default=420, help='all data')
parser.add_argument('--test_num', type=int, default=60, help='all data')
parser.add_argument('--base_lr', type=float,  default=0.1, help='segmentation network learning rate')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--conf_thresh', type=float, default=0.98, help='ema_decay')
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency_type', type=str, default="mse", help='consistency_type')
parser.add_argument('--cps_rampup', action='store_true', default=True) # <--
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
args = parser.parse_args()


def EMA(cur_weight, past_weight, momentum=0.9):
    new_weight = momentum * past_weight + (1 - momentum) * cur_weight
    return new_weight

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

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


def train(args, snapshot_path,):
    base_lr = args.base_lr
    train_data_path = args.root_path
    batch_size = args.batch_size
    max_iterations = args.max_iterations
    best_performance1_test = 0.7
    best_performance2_test = 0.7
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    net1 = unet_3D_cps(1, args.num_classes).cuda()
    net2 = unet_3D_cps(1, args.num_classes).cuda()
    model1 = kaiming_normal_init_weight(net1)
    model2 = xavier_normal_init_weight(net2)
    model1.train()
    model2.train()  
    db_train = BraTS2019(base_dir=train_data_path,
                         split='train',
                         num=None,
                         transform=transforms.Compose([
                             RandomRotFlip(),
                             RandomCrop(args.patch_size),
                             ToTensor(),
                         ]))
    
    valset = FlareDataset('val', args, [64, 128, 128])
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False) 

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    labeled_idxs = list(range(0, args.labeled_num))
    unlabeled_idxs = list(range(args.labeled_num, args.data_num))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=4, pin_memory=True) # , worker_init_fn=worker_init_fn
    
    optimizer1 = optim.AdamW( 
        params=model1.parameters(),
        lr=base_lr, betas=(0.9, 0.999), weight_decay=0.01)
    optimizer2 = optim.AdamW( 
        params=model2.parameters(),
        lr=base_lr, betas=(0.9, 0.999), weight_decay=0.01)
    iter_num = 0

    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(args.num_classes)

    writer = SummaryWriter(snapshot_path + '/tensorboardlog')
    logging.info("{} iterations per epoch".format(len(trainloader)))

    max_epoch = max_iterations // len(trainloader) + 1
    epoch = -1
    best_epoch1, best_epoch2 = 0, 0
    best_performance1, best_performance2 = 0.7, 0.7

    start_time = time.time()
    if os.path.exists(os.path.join(snapshot_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(snapshot_path, 'latest.pth'), weights_only=False)
        model1.load_state_dict(checkpoint['model1'])
        model2.load_state_dict(checkpoint['model2'])
        optimizer1.load_state_dict(checkpoint['optimizer1'])
        optimizer2.load_state_dict(checkpoint['optimizer2'])
        epoch = checkpoint['epoch']
        best_performance1= checkpoint['previous_best1']
        best_performance2 = checkpoint['previous_best2']
        best_epoch1 = checkpoint['best_epoch1']
        best_epoch2 = checkpoint['best_epoch2']
        iter_num = checkpoint['iter_num']
        logging.info('***** Load from checkpoint at epoch: {}, iter_num: {} *****'.format(epoch, iter_num))
        
    for epoch in range(epoch + 1, max_epoch):
        for i_batch, sampled_batch in enumerate(trainloader):
            if iter_num % (len(trainloader) * 30) == 0:
                logging.info(f"Epoch: {epoch}/{max_epoch}, exp: {args.exp.split('_')[-1]}")

            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            
            # complmentray dropout for cps
            features1 = model1.encoder(volume_batch)
            bs = features1[-1].shape[0]
            dropout_prob = 0.5
            num_kept = int(bs * (1 - dropout_prob))
            binomial = torch.distributions.binomial.Binomial(probs=0.5)
            kept_indexes = torch.randperm(bs)[:num_kept]
            dropoutmask1, dropoutmask2 = [], []
            for j in range(0, len(features1)):
                d =  features1[j].shape[1]
                dropout_mask1 = binomial.sample((bs, d)).cuda() * 2.0
                dropout_mask2 = 2.0 - dropout_mask1
                dropout_mask1[kept_indexes, :] = 1.0
                dropout_mask2[kept_indexes, :] = 1.0
                dropoutmask1.append(dropout_mask1)
                dropoutmask2.append(dropout_mask2)
            outputs1 = model1(volume_batch, comp_drop=dropoutmask1)
            # outputs1 = model1(volume_batch)
            outputs_soft1 = torch.softmax(outputs1, dim=1)

            # outputs2 = model2(volume_batch)
            outputs2 = model2(volume_batch, comp_drop=dropoutmask2)
            outputs_soft2 = torch.softmax(outputs2, dim=1)
                
            loss1 = 0.5 * (ce_loss(outputs1[:args.labeled_bs],
                                   label_batch[:][:args.labeled_bs].long()) + dice_loss(
                outputs_soft1[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))
            loss2 = 0.5 * (ce_loss(outputs2[:args.labeled_bs],
                                   label_batch[:][:args.labeled_bs].long()) + dice_loss(
                outputs_soft2[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1)))
            consistency_weight = get_current_consistency_weight(iter_num // 150)

            max_A = torch.argmax(outputs1.detach(), dim=1, keepdim=False).long()
            max_B = torch.argmax(outputs2.detach(), dim=1, keepdim=False).long()
            unsup_loss_func_A = CrossEntropyLoss()
            unsup_loss_func_B = CrossEntropyLoss()
            loss_cps = unsup_loss_func_B(outputs1, max_B) + unsup_loss_func_A(outputs2, max_A)

            model1_loss = loss1
            model2_loss = loss2
            loss = model1_loss + model2_loss + consistency_weight*loss_cps
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer1.step()
            optimizer2.step()
            iter_num = iter_num + 1
            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group1 in optimizer1.param_groups:
                param_group1['lr'] = lr_
            for param_group2 in optimizer2.param_groups:
                param_group2['lr'] = lr_

            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar(
                'ratio/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('loss/model1_loss',
                              model1_loss, iter_num)
            writer.add_scalar('loss/model2_loss',
                              model2_loss, iter_num)
            writer.add_scalar('loss/loss1',
                              loss1, iter_num)
            writer.add_scalar('loss/loss2',
                              loss2, iter_num)
            writer.add_scalar('loss/loss_cps',
                              loss_cps, iter_num)
            if iter_num % (len(trainloader) * 3) == 0:
                logging.info(f'iteration: {iter_num}/{args.max_iterations} model1 loss: {round(loss1.item(), 3)}, model2 loss: {round(loss2.item(), 3)},'
                             f'consistency_weight: {round(consistency_weight, 5)}, lr: {round(lr_, 7)}')
            if iter_num >= max_iterations * 0.85 and iter_num % 200 == 0:
                model1.eval()
                ############## model1_test ###############
                mDICE1, dice_class1 = evaluate_3d(valloader, model1, cfg, val_mode='model1')                     
                if mDICE1 > best_performance1_test:
                    best_performance1_test = mDICE1.item()
                    save_mode_path = os.path.join(snapshot_path,'model1_iter{}_test_dice_{}.pth'.format(
                                                      iter_num, round(best_performance1_test, 4)))
                    torch.save(model1.state_dict(), save_mode_path)
                writer.add_scalar('test_model_1/model1_dice_score', mDICE1, iter_num)
                for (cls_idx, dice) in enumerate(dice_class1):
                    logging.info(f'*****model1_dice_{CLASSES[args.dataset][cls_idx]}{cls_idx}: {dice}')
                    writer.add_scalar(f'test_model_1/model1_dice{CLASSES[args.dataset][cls_idx]}', dice, iter_num)
                logging.info('iteration %d/%d : pre best model1 mdice: %.4f model1_test_mdice : %.4f ' % (
                        iter_num, max_iterations, best_performance1_test, mDICE1))
                model1.train()

                model2.eval()
                ############## model2_test #################
                mDICE2, dice_class2 = evaluate_3d(valloader, model2, cfg, val_mode='model2')
                if mDICE2 > best_performance2_test:
                    best_performance2_test = mDICE2.item()
                    save_mode_path = os.path.join(snapshot_path,'model2_iter{}_test_dice_{}.pth'.format(
                                                      iter_num, round(best_performance2_test, 4)))
                    torch.save(model2.state_dict(), save_mode_path)
                writer.add_scalar('test_model_2/model2_dice_score', mDICE2, iter_num)
                for (cls_idx, dice) in enumerate(dice_class2):
                    logging.info(f'*****model2_dice_{CLASSES[args.dataset][cls_idx]}{cls_idx}: {dice}')
                    writer.add_scalar(f'test_model_2/model2_dice{CLASSES[args.dataset][cls_idx]}', dice, iter_num)
                logging.info('iteration %d/%d : pre best model2 mdice: %.4f model2_test_dice_score : %.4f' % (
                        iter_num,  max_iterations, best_performance2_test, mDICE2))
                model2.train()
            if iter_num >= max_iterations:
                break
        checkpoint = {
            'model1': model1.state_dict(),
            'model2': model2.state_dict(),
            'optimizer1': optimizer1.state_dict(),
            'optimizer2': optimizer2.state_dict(),
            'epoch': epoch,
            'previous_best1': best_performance1,
            'previous_best2': best_performance2,
            'best_epoch1': best_epoch1,
            'best_epoch2': best_epoch2,
            'iter_num': iter_num
        }
        torch.save(checkpoint, os.path.join(snapshot_path, 'latest.pth'))
        if iter_num >= max_iterations:
            break
    writer.close()
    end_time = time.time()
    total_time = end_time - start_time  
    print(f"Model TIMES: {round(total_time, 8)}")

if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = f"./{args.exp}_labeled{args.labeled_num}_{args.dataset}_{args.model}_{args.base_lr}"
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    train(args, snapshot_path)
