import argparse
import logging
import os
import pprint, time
import numpy as np
import yaml
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from datasets.dataset import FlareDataset
from model.unet_3d import unet_3D_cps, kaiming_normal_init_weight, xavier_normal_init_weight
from utils.classes import CLASSES
from utils.util import count_params, init_log, AverageMeter, evaluate_3d, DiceLoss

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = '5'

parser = argparse.ArgumentParser(description='prcv25 stage1 for Flare22')
parser.add_argument('--config', type=str, default='./configs/Your.yaml')
parser.add_argument('--base_dir', type=str, default='./27_FLARE2022')
parser.add_argument('--save_path', type=str, default='./log')
parser.add_argument('--checkpoint_path', type=str, default='./checkpoints')
parser.add_argument('--exp', type=str, default='Your experiments description',help='expriment description')
parser.add_argument('--local_rank', '--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
parser.add_argument('--num', default=None, type=int)
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
    start_time = time.time()
    args = parser.parse_args()
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    
    def get_current_consistency_weight(epoch):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return args.consistency * sigmoid_rampup(epoch, args.consistency_rampup)
    
    os.makedirs(args.save_path, exist_ok=True)
    save_path = os.path.join(args.save_path, 'Ep{}BS{}_{}_{}_{}'.format(cfg['epochs'], cfg['batch_size'], cfg['dataset'], cfg['model'], args.exp))
    cp_path = os.path.join(args.checkpoint_path, 'Ep{}BS{}_{}_{}_{}'.format(cfg['epochs'], cfg['batch_size'], cfg['dataset'], cfg['model'], args.exp))
    os.makedirs(cp_path, exist_ok=True)
    os.makedirs(save_path, exist_ok=True)

    logger = init_log('global', logging.INFO, os.path.join(save_path, args.exp))
    logger.propagate = 0
    all_args = {**cfg, **vars(args)}
    logger.info('{}\n'.format(pprint.pformat(all_args)))
    writer = SummaryWriter(save_path)

    model1 = unet_3D_cps(1, 14).cuda()
    model1 = kaiming_normal_init_weight(model1)
    model2 = unet_3D_cps(1, 14).cuda()
    model2 = xavier_normal_init_weight(model2)
    
    optimizer1 = AdamW( 
        params=model1.parameters(),
        lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=0.01
    )
    optimizer2 = AdamW( 
        params=model2.parameters(),
        lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=0.01
    )
    
    # dice_loss = DiceLoss(cfg['nclass']).cuda(local_rank)
    dice_loss = DiceLoss(cfg['nclass']).cuda()
    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda()
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    criterion_u = nn.CrossEntropyLoss(reduction='none').cuda()

    logger.info('Total params: {:.3f}M'.format(count_params(model1) + count_params(model2)))
    #===================================== Dataset ==================================#
    trainset_u = FlareDataset('train_u', args, cfg['crop_size'])
    trainset_l = FlareDataset('train_l', args, cfg['crop_size'], nsample=len(trainset_u.name_list))
    valset = FlareDataset('val', args, cfg['crop_size'])
    
    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'], pin_memory=True, num_workers=6, drop_last=True)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'], pin_memory=True, num_workers=6, drop_last=True)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False)
    
    
    total_iters = len(trainloader_u) * cfg['epochs']
    print('Total iters: %d' % total_iters)
    pre_best_dice1, pre_best_dice2 = 0.7, 0.7
    best_epoch_1, best_epoch_2 = 0, 0
    epoch = -1
    iter_num = 0
    if os.path.exists(os.path.join(cp_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(cp_path, 'latest.pth'), weights_only=False)
        model1.load_state_dict(checkpoint['model1'])
        model2.load_state_dict(checkpoint['model2'])
        optimizer1.load_state_dict(checkpoint['optimizer1'])
        optimizer2.load_state_dict(checkpoint['optimizer2'])
        epoch = checkpoint['epoch']
        pre_best_dice1 = checkpoint['previous_best_1']
        pre_best_dice2 = checkpoint['previous_best_2']
        best_epoch_1 = checkpoint['best_epoch_1']
        best_epoch_2 = checkpoint['best_epoch_2']
        iter_num = checkpoint['iter_num']
        logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
            
    for epoch in range(epoch + 1, cfg['epochs']):
        logger.info('===========> Epoch: {}/{}, Previous best mdice model1: {:.4f} @epoch: {}, '
                'model2: {:.4f} @epoch: {}'.format(epoch, cfg['epochs'], pre_best_dice1, best_epoch_1, pre_best_dice2, best_epoch_2))
        
        total_loss  = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_x1 = AverageMeter()
        total_loss_x2 = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_s1 = AverageMeter()
        total_loss_s2 = AverageMeter()
        total_mask_ratio1 = AverageMeter()
        total_mask_ratio2 = AverageMeter()


        loader = zip(trainloader_l, trainloader_u)
        
        model1.train()
        model2.train()
        is_best = False
        for i, ((img_x, mask_x),
                (img_u_w, img_u_s1, img_u_s2, ignore_mask, cutmix_box1, cutmix_box2)) in enumerate(loader):
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w, img_u_s1, img_u_s2 = img_u_w.cuda(), img_u_s1.cuda(), img_u_s2.cuda()
            ignore_mask, cutmix_box1, cutmix_box2 = ignore_mask.cuda(), cutmix_box1.cuda(), cutmix_box2.cuda()
            with torch.no_grad():           
                pred_u_w1 = model2(img_u_w).detach()
                conf_u_w1 = pred_u_w1.softmax(dim=1).max(dim=1)[0]
                mask_u_w1 = pred_u_w1.argmax(dim=1)
                pred_u_w2 = model2(img_u_w).detach()
                conf_u_w2 = pred_u_w2.softmax(dim=1).max(dim=1)[0]
                mask_u_w2 = pred_u_w2.argmax(dim=1)
            
            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = img_u_s1.flip(0)[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = img_u_s2.flip(0)[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]
            
            pred_x1 = model1(img_x)
            pred_x2 = model2(img_x)
            
            features1 = model1.encoder(img_u_s1)
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
            pred_u_s1 = model1(img_u_s1, comp_drop=dropoutmask1) 
            pred_u_s2 = model2(img_u_s1, comp_drop=dropoutmask2) 
            # pred_u_s1, pred_u_s2 = model1(torch.cat((img_u_s1, img_u_s2)), comp_drop=True).chunk(2)  # cps 形成互补dropout
            
            mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = mask_u_w1.clone(), conf_u_w1.clone(), ignore_mask.clone()
            mask_u_w_cutmixed2, conf_u_w_cutmixed2, ignore_mask_cutmixed2 = mask_u_w2.clone(), conf_u_w2.clone(), ignore_mask.clone()

            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w1.flip(0)[cutmix_box1 == 1]
            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w1.flip(0)[cutmix_box1 == 1]
            ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask.flip(0)[cutmix_box1 == 1]
            
            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w1.flip(0)[cutmix_box2 == 1]
            conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w2.flip(0)[cutmix_box2 == 1]
            ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask.flip(0)[cutmix_box2 == 1]
            
            dice_loss_x1 = dice_loss(pred_x1, mask_x)
            dice_loss_x2 = dice_loss(pred_x2, mask_x)
            loss_x_1 = (criterion_l(pred_x1, mask_x) + dice_loss_x1) / 2.0  
            loss_x_2 = (criterion_l(pred_x2, mask_x) + dice_loss_x2) / 2.0
            loss_x = (loss_x_1 + loss_x_2) / 2.0  # cps

            loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed2)
            loss_u_s1 = loss_u_s1 * ((conf_u_w_cutmixed1 >= cfg['conf_thresh']) & (ignore_mask_cutmixed1 != 255))
            loss_u_s1 = loss_u_s1.sum() / (ignore_mask_cutmixed1 != 255).sum().item()
            
            loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed1)
            loss_u_s2 = loss_u_s2 * ((conf_u_w_cutmixed2 >= cfg['conf_thresh']) & (ignore_mask_cutmixed2 != 255))
            loss_u_s2 = loss_u_s2.sum() / (ignore_mask_cutmixed2 != 255).sum().item()
            
            consistency_weight = get_current_consistency_weight(iter_num // 200)
            loss_u_s = (loss_u_s1 + loss_u_s2) / 2.0
            
            loss = ((2.0 - consistency_weight) * loss_x + consistency_weight * loss_u_s) / 2.0        # 考虑添加focal_loss
            
            iter_num = iter_num + 1
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer1.step()
            optimizer2.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_x1.update(loss_x_1.item())
            total_loss_x2.update(loss_x_2.item())
            total_loss_s.update(loss_u_s.item())
            total_loss_s1.update(loss_u_s1.item())
            total_loss_s2.update(loss_u_s2.item())
            mask_ratio1 = ((conf_u_w1 >= cfg['conf_thresh']) & (ignore_mask != 255)).sum().item() / (ignore_mask != 255).sum()
            mask_ratio2 = ((conf_u_w2 >= cfg['conf_thresh']) & (ignore_mask != 255)).sum().item() / (ignore_mask != 255).sum()
            total_mask_ratio1.update(mask_ratio1.item())
            total_mask_ratio2.update(mask_ratio2.item())

            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer1.param_groups[0]["lr"] = lr
            optimizer2.param_groups[0]["lr"] = lr
           
            writer.add_scalar(
            'consistency_weight/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('train/loss_all', loss.item(), iters)
            writer.add_scalar('train/loss_x', loss_x.item(), iters)
            writer.add_scalar('train/loss_x1', loss_x_1.item(), iters)
            writer.add_scalar('train/loss_x2', loss_x_2.item(), iters)
            writer.add_scalar('train/dice_loss_x1', dice_loss_x1.item(), iters)
            writer.add_scalar('train/dice_loss_x2', dice_loss_x2.item(), iters)
            writer.add_scalar('train/loss_s', loss_u_s.item(), iters)
            writer.add_scalar('train/loss_s1', loss_u_s1.item(), iters)
            writer.add_scalar('train/loss_s2', loss_u_s2.item(), iters)
            writer.add_scalar('train/mask_ratio1', mask_ratio1, iters)
            writer.add_scalar('train/mask_ratio2', mask_ratio2, iters)

            if (i % (len(trainloader_u) // 3) == 0):
                logger.info('Iters: {}/{}, LR: {:.7f}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, consistency_weight: {:.5f}, Mask ratio1: {:.3f}, Mask ratio2: '
                            '{:.3f}'.format(iter_num, total_iters, lr, total_loss.avg, total_loss_x.avg, 
                                            total_loss_s.avg, consistency_weight, total_mask_ratio1.avg, total_mask_ratio2.avg))
        
        if iter_num >= (0.75 * total_iters) and epoch % 2 == 0:
            model1.eval()
            mDICE1, dice_class1 = evaluate_3d(valloader, model1, cfg, val_mode='model1')
            model1.train() 
            model2.eval()                       
            mDICE2, dice_class2 = evaluate_3d(valloader, model2, cfg, val_mode='model2')
            model2.train()
            # if rank == 0:
            for (cls_idx, dice) in enumerate(dice_class1):
                if cls_idx == 0:
                        continue
                logger.info('*** Evaluation: Class [{:} {:}] Dice model: {:.3f}, '
                            'ema: {:.3f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], dice, dice_class2[cls_idx]))
                writer.add_scalar('eval/%smodel_DICE' % (CLASSES[cfg['dataset']][cls_idx]), dice, epoch)
                writer.add_scalar('eval/%smodel_ema_DICE' % (CLASSES[cfg['dataset']][cls_idx]), dice_class2[cls_idx], epoch)
            logger.info('*** Evaluation  {}:  MeanDice model: {:.3f}, model ema: {:.3f}'.format(args.exp.split('_')[-1], mDICE1, mDICE2))
            writer.add_scalar('eval/mDice1', mDICE1.item(), epoch)
            writer.add_scalar('eval/mDICE2', mDICE2.item(), epoch) 

            is_best = (mDICE1 >= pre_best_dice1 and mDICE2 >= pre_best_dice2)
        
            pre_best_dice1 = max(mDICE1, pre_best_dice1)
            pre_best_dice2 = max(mDICE2, pre_best_dice2)
            if mDICE1 == pre_best_dice1:
                best_epoch_1 = epoch
            if mDICE2 == pre_best_dice2:
                best_epoch_2 = epoch
        
        checkpoint = {
            'model1': model1.state_dict(),
            'model2': model2.state_dict(),
            'optimizer1': optimizer1.state_dict(),
            'optimizer2': optimizer2.state_dict(),
            'epoch': epoch,
            'previous_best_1': pre_best_dice1,
            'previous_best_2': pre_best_dice2,
            'best_epoch_1': best_epoch_1,
            'best_epoch_2': best_epoch_2,
            'iter_num': iter_num
        }
        torch.save(checkpoint, os.path.join(cp_path, 'latest.pth'))
        if is_best:
            torch.save(checkpoint, os.path.join(cp_path, 'ep{}_bs{}_mdice1{}_mdice2{}_stage1.pth'.format(epoch, cfg['batch_size'], mDICE1, mDICE2)))
        if epoch >= (cfg['epochs'] - 1):
            end_time = time.time()
            logger.info('Training time: {:.2f}s'.format((end_time - start_time)))

if __name__ == '__main__':
    main()

