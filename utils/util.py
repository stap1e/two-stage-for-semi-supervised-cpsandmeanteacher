import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import torch.distributed as dist
import scipy.ndimage as nd
from skimage import measure
import SimpleITK as sitk
from medpy import metric

from utils.classes import CLASSES
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

def dice_coefficient(prediction, target, class_num = 14):
    dice_coefficient = []

    for i in range(class_num - 1):
        dice_cls = metric.binary.dc(prediction == (i + 1), target == (i + 1))
        dice_coefficient.append(dice_cls)

    return dice_coefficient


def jaccard_coefficient(prediction, target, slice,class_num = 14 ):

    jaccard_coefficient = []
    a = np.unique(target)
    for i in range(class_num - 1):
        try:
            dice_cls = metric.binary.jc(prediction == (i + 1), target == (i + 1))
            jaccard_coefficient.append(dice_cls)
        except ZeroDivisionError:
            pass

    return sum(jaccard_coefficient)/len(jaccard_coefficient)


def calculate_metrics(pred_folder, label_folder, args):
    pred_files = [os.path.join(pred_folder, file) for file in os.listdir(pred_folder)]
    label_files = [os.path.join(label_folder, file) for file in os.listdir(label_folder)]

    accuracy_lists = []
    jaccard_scores = []
    num_samples = args.test_num 
    class_accuracy = [0.0] * (args.num_classes - 1)
    dice_lists = {str(i): [] for i in range(args.num_classes - 1)} # 不考虑背景类别

    for pred_path, label_path in zip(pred_files, label_files):
        pred = sitk.GetArrayFromImage(sitk.ReadImage(pred_path))
        label = sitk.GetArrayFromImage(sitk.ReadImage(label_path))
        dice_scores = dice_coefficient(pred, label, args.num_classes)
        accuracy_lists.append(dice_scores)
        # for class_id, dice_score in enumerate(dice_scores, start=1):
        for class_id, dice_score in enumerate(dice_scores):
            # if class_id == 1:
            #     print(f">>>>1=>>>>")
            # if class_id == 4:
            #     print(f">>>>4====>")
            # if class_id == 14:
            #     print(f">>>>14====>---")
            # if class_id == 15:
            #     print(f">>>>15====>---!!!!!!!!")
            print(f"Class{CLASSES[args.dataset][class_id+1]}_{class_id} Dice: {dice_score:.4f}")
            dice_lists[str(class_id)].append(round(dice_score, 4))
        print(f"{pred_path}_____meandice_____:   {np.array(dice_scores).mean():.4f}\n==============")


        jaccard_scores.append(jaccard_coefficient(pred, label, slice))
    # for i in range(args.num_classes-1):
    #     print(f"Class{CLASSES[args.dataset][i+1]} mDice: {sum(dice_lists[str(i)]) / len(dice_lists[str(i)]):.4f}")
    
    for i in range(4):
        accuracy_lists[-(i+1)][8] = 0
        
    for accuracy_list in accuracy_lists:
        for class_id in range(args.num_classes-1):
            class_accuracy[class_id] += accuracy_list[class_id]

    # 计算每个类别的平均精度
    average_accuracy = [acc / num_samples for acc in class_accuracy]    

    average_accuracy[8] = average_accuracy[8] * args.test_num / 10    # test中有四个案例没有8这个器官，对于flare22数据集而言

    avg_jaccard = np.mean(jaccard_scores)


    # 打印每个类别的平均精度
    for class_id, avg_acc in enumerate(average_accuracy):
        print(f"Class {CLASSES[args.dataset][class_id+1]} Average Accuracy: {avg_acc*100:.2f}")

    print(f'******************{args.dataset} Average Accuracy: {np.mean(average_accuracy)}')
    print(f'******************{args.dataset} Average jaccard: {avg_jaccard}')
    return average_accuracy, avg_jaccard



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
