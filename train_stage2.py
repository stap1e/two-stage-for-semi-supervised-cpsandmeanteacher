"""
 * URL: https://github.com/LiheYoung/UniMatch-V2
 * 
 * Copyright (c) LiheYoung
"""
import argparse
from copy import deepcopy
import logging
import os
import pprint
import numpy as np
import yaml, time
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler

from datasets.dataset import FlareDataset
from model.unet_3d import unet_3D_mt, kaiming_normal_init_weight
from utils.classes import CLASSES
from utils.util import count_params, init_log, AverageMeter, evaluate_3d
from utils.dist_helper import setup_distributed

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = '5, 4'

parser = argparse.ArgumentParser(description='stage2 for FLARE2022')
parser.add_argument('--config', type=str, default='./configs/FLARE22_stage2.yaml')
parser.add_argument('--base_dir', type=str, default='./27_FLARE2022')
parser.add_argument('--dataset', type=str, default='flare22_ssl',help='dataset name')
parser.add_argument('--num_classes', type=int, default=14, help='number of class')
parser.add_argument('--save_path', type=str, default='./log')
parser.add_argument('--log_file', type=str, default='')
parser.add_argument('--exp', type=str, default='Your experiments description', help='expriment description')
parser.add_argument('--checkpoint_path', type=str, default='./checkpoints')
parser.add_argument('--teacher_ckpt', type=str, default='./mt.pth')
parser.add_argument('--val_patch_size', type=list, default=[64, 160, 160],help='patch size of network input')
parser.add_argument('--val_xy', type=int, default=80,help='patch size of val network input')
parser.add_argument('--val_z', type=int, default=32,help='patch size of val network input')
parser.add_argument('--local_rank', '--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
parser.add_argument('--num', default=None, type=int, help='if is amos: label num(18 or 90), if is flare: unlabeled num(None or 42)')
parser.add_argument('--test_num', type=int, default=60, help='all test data num')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--consistency', type=float, default=1, help='consistency')

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))
    
    
def main():
    args = parser.parse_args()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    
    def get_current_consistency_weight(epoch):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)
    
    os.makedirs(args.save_path, exist_ok=True)
    save_path = os.path.join(args.save_path, 'Ep{}_{}_thresh{}_{}'.format(cfg['epochs'], cfg['dataset'], cfg['conf_thresh'], args.exp))
    cp_path = os.path.join(args.checkpoint_path, 'Ep{}_{}_thresh{}_{}'.format(cfg['epochs'], cfg['dataset'], cfg['conf_thresh'], args.exp))
    os.makedirs(cp_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    logger = init_log('global', logging.INFO, os.path.join(save_path, args.exp))
    logger.propagate = 0
    
    rank, world_size = setup_distributed(port=args.port)
    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        writer = SummaryWriter(save_path)

    cudnn.enabled = True
    cudnn.benchmark = True
    start_time = time.time()

    model = unet_3D_mt(in_chns=1, class_num=cfg['nclass']).cuda()
    model = kaiming_normal_init_weight(model)
    
    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local_rank], broadcast_buffers=False, output_device=local_rank)

    model_ema = deepcopy(model)
    state_dict = torch.load(args.teacher_ckpt, weights_only=False)
    model_ema.load_state_dict(state_dict['model1'])
    model_ema.eval()
    for param in model_ema.parameters():
        param.requires_grad = False
        
    optimizer = AdamW( 
        params=model.parameters(),
        lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=0.01)
    
    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda(local_rank)
    
    num_gpus = torch.cuda.device_count()
    if rank == 0:
        logger.info('use {} gpus!'.format(num_gpus))
        logger.info('Total params: {:.3f}M'.format(count_params(model)))
    #===================================== Dataset ==================================#
    trainset_u = FlareDataset('train_u', args, cfg['crop_size'])
    trainset_l = FlareDataset('train_l', args, cfg['crop_size'], nsample=len(trainset_u.name_list))
    valset = FlareDataset('val', args, cfg['crop_size'])
    
    trainsampler_l = DistributedSampler(trainset_l)
    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'], pin_memory=True, num_workers=6, drop_last=True, sampler=trainsampler_l)

    trainsampler_u = DistributedSampler(trainset_u)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'], pin_memory=True, num_workers=6, drop_last=True, sampler=trainsampler_u)
    valsampler = DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False, sampler=valsampler)  # val batch_size must be 1
    
    
    total_iters = len(trainloader_u) * cfg['epochs']
    if rank == 0:
        print('Total iters: %d' % total_iters)
    pre_best_dice1, pre_best_dice2 = 0.75, 0.75
    best_epoch_1, best_epoch_2 = 0, 0
    epoch = -1
    iter_num = 0
    if os.path.exists(os.path.join(cp_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(cp_path, 'latest.pth'), weights_only=False)
        model.load_state_dict(checkpoint['model'])
        model_ema.load_state_dict(checkpoint['model_ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        pre_best_dice1 = checkpoint['previous_best_1']
        pre_best_dice2 = checkpoint['previous_best_2']
        best_epoch_1 = checkpoint['best_epoch']
        best_epoch_2 = checkpoint['best_epoch_ema']
        iter_num = checkpoint['iter_num']
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
            
    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {}/{}, Previous best mdice model: {:.4f} @epoch: {}, '
                    'ema: {:.4f} @epoch: {}'.format(epoch, cfg['epochs'], pre_best_dice1, best_epoch_1, pre_best_dice2, best_epoch_2))
        
        total_loss  = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_s1 = AverageMeter()
        total_loss_s2 = AverageMeter()
        total_mask_ratio = AverageMeter()

        if hasattr(trainloader_l.sampler, "set_epoch"):
            trainloader_l.sampler.set_epoch(epoch)
        if hasattr(trainloader_u.sampler, "set_epoch"):
            trainloader_u.sampler.set_epoch(epoch)

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u)
        is_best = False
        model.train()
        for i, ((img_x, mask_x),
                (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2)) in enumerate(loader):
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w, img_u_s1, img_u_s2 = img_u_w.cuda(), img_u_s1.cuda(), img_u_s2.cuda()
            ignore_mask, cutmix_box1, cutmix_box2 = ignore_mask.cuda(), cutmix_box1.cuda(), cutmix_box2.cuda()
            with torch.no_grad():           
                pred_u_w = model_ema(img_u_w).detach()
                conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
                mask_u_w = pred_u_w.argmax(dim=1)

            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = img_u_s1.flip(0)[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = img_u_s2.flip(0)[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]
            
            pred_x = model(img_x)
            pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2)), comp_drop=True).chunk(2)
            
            mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
            mask_u_w_cutmixed2, conf_u_w_cutmixed2, ignore_mask_cutmixed2 = mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()

            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w.flip(0)[cutmix_box1 == 1]
            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w.flip(0)[cutmix_box1 == 1]
            ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask.flip(0)[cutmix_box1 == 1]
            
            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w.flip(0)[cutmix_box2 == 1]
            conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w.flip(0)[cutmix_box2 == 1]
            ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask.flip(0)[cutmix_box2 == 1]
            loss_x = criterion_l(pred_x, mask_x)

            loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
            loss_u_s1 = loss_u_s1 * ((conf_u_w_cutmixed1 >= cfg['conf_thresh']) & (ignore_mask_cutmixed1 != 255))
            loss_u_s1 = loss_u_s1.sum() / (ignore_mask_cutmixed1 != 255).sum().item()
            
            loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)
            loss_u_s2 = loss_u_s2 * ((conf_u_w_cutmixed2 >= cfg['conf_thresh']) & (ignore_mask_cutmixed2 != 255))
            loss_u_s2 = loss_u_s2.sum() / (ignore_mask_cutmixed2 != 255).sum().item()
            
            consistency_weight = get_current_consistency_weight(iter_num // 200)
            loss_u_s = (loss_u_s1 + loss_u_s2) / 2.0
             
            loss = (loss_x + loss_u_s) / 2.0
            
            iter_num = iter_num + 1
            optimizer.zero_grad()
            torch.autograd.set_detect_anomaly(True)
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update(loss_u_s.item())
            total_loss_s1.update(loss_u_s1.item())
            total_loss_s2.update(loss_u_s2.item())
            mask_ratio = ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255)).sum().item() / (ignore_mask != 255).sum()
            total_mask_ratio.update(mask_ratio.item())

            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            ema_ratio = min(1 - 1 / (iters + 1), 0.996)
            
            for param, param_ema in zip(model.parameters(), model_ema.parameters()):
                param_ema.copy_(param_ema * ema_ratio + param.detach() * (1 - ema_ratio))
            for buffer, buffer_ema in zip(model.buffers(), model_ema.buffers()):
                buffer_ema.copy_(buffer_ema * ema_ratio + buffer.detach() * (1 - ema_ratio))
                
            if rank == 0:
                writer.add_scalar(
                'consistency_weight/consistency_weight', consistency_weight, iter_num)
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/lr', lr, iters)
                writer.add_scalar('train/loss_x', loss_x.item(), iters)
                writer.add_scalar('train/loss_s', loss_u_s.item(), iters)
                writer.add_scalar('train/loss_s1', loss_u_s1.item(), iters)
                writer.add_scalar('train/loss_s2', loss_u_s2.item(), iters)
                writer.add_scalar('train/mask_ratio', mask_ratio, iters)

                if (i % (len(trainloader_u) // 3) == 0):
                    logger.info('Iters: {}/{}, LR: {:.7f}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, consistency_weight: {:.5f}, Mask ratio: {:.3f}'.format(
                        iter_num, total_iters, lr, total_loss.avg, total_loss_x.avg, total_loss_s.avg, consistency_weight, total_mask_ratio.avg))
        if iter_num >= (0.75 * total_iters) and epoch % 2 == 0:
            model.eval()
            mDICE1, dice_class1 = evaluate_3d(valloader, model    , cfg, val_mode='model')
            model.train()
            mDICE2, dice_class2 = evaluate_3d(valloader, model_ema, cfg, val_mode='ema model')
            if rank == 0:
                for (cls_idx, dice) in enumerate(dice_class1):
                    if cls_idx == 0:
                        continue
                    logger.info('*** Evaluation: Class [{:} {:}] Dice model: {:.3f}, '
                                'ema: {:.3f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], dice, dice_class2[cls_idx]))
                    writer.add_scalar('eval/%smodel_DICE' % (CLASSES[cfg['dataset']][cls_idx]), dice, epoch)
                    writer.add_scalar('eval/%smodel_ema_DICE' % (CLASSES[cfg['dataset']][cls_idx]), dice_class2[cls_idx], epoch)
                logger.info('*** Evaluation  {}:  MeanDice model1: {:.3f}, model2: {:.3f}'.format(args.exp.split('_')[-1], mDICE1, mDICE2))
                writer.add_scalar('eval/mDice', mDICE1, epoch)
                writer.add_scalar('eval/mDICE_ema', mDICE2, epoch)
                
            is_best = (mDICE1.item() >= pre_best_dice1) or (mDICE2.item() >= pre_best_dice2)
            
            pre_best_dice1 = max(mDICE1.item(), pre_best_dice1)
            pre_best_dice2 = max(mDICE2.item(), pre_best_dice2)
            if mDICE1.item() == pre_best_dice1:
                best_epoch_1 = epoch
            if mDICE2.item() == pre_best_dice2:
                best_epoch_2 = epoch
            
        checkpoint = {
            'model': model.state_dict(),
            'model_ema': model_ema.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'previous_best_1': pre_best_dice1,
            'previous_best_2': pre_best_dice2,
            'best_epoch': best_epoch_1,
            'best_epoch_ema': best_epoch_2,
            'iter_num': iter_num
        }
        torch.save(checkpoint, os.path.join(cp_path, 'latest.pth'))
        if is_best:
            torch.save(checkpoint, os.path.join(cp_path, 'ep{}_bs{}mdice{:.4f}_ema{:.4f}stage2.pth'.format(epoch, cfg['batch_size'], mDICE1, mDICE2)))
        if epoch >= (cfg['epochs'] - 1):
            end_time = time.time()
            logger.info('Training time: {:.2f}s'.format((end_time - start_time)))
            

if __name__ == '__main__':
    main()

