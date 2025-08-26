import os
import argparse
import math, time
import numpy as np
import torch
import torch.nn.functional as F
import SimpleITK as sitk
import logging
from tqdm import tqdm   

from utils.classes import CLASSES
from utils.util import calculate_metrics
from model.unet_3d import unet_3D_mt


parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=str, default='0', help='GPU to use')
parser.add_argument('--ratio', type=int, default='10', help='laebeled data to use')
parser.add_argument('--num_classes', type=int, default=14, help='laebeled data to use')
parser.add_argument('--test_num', type=int, default=14, help='test num to use: FLARE22:14, AMOS22:60')
parser.add_argument('--dataset', type=str, default='flare22', help='GPU to use')
FLAGS = parser.parse_args()
# os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

def getFiles(targetdir):
    ls = []
    for fname in os.listdir(targetdir):
        path = os.path.join(targetdir, fname)
        if os.path.isdir(path):
            continue
        ls.append(fname)
    return ls

def extract_categories(label_image):
    unique_classes = np.unique(label_image)
    return unique_classes.tolist()

def test_all_case(net, imdir, maskdir, output2, num_classes, patch_size=(112, 112, 80), stride_xy=18, stride_z=4, save_result=True,
                  test_save_path=None, preproc_fn=None, pbar=None):
    for _, fname in enumerate(sorted(getFiles(imdir))):
        # load files
        print(f"Processing {fname.replace('_0000.nii.gz', '')}")
        sitk_im = sitk.ReadImage(os.path.join(imdir, fname))  # img
        im_x_y = sitk.GetArrayFromImage(sitk_im)  

        if preproc_fn is not None:
            image = preproc_fn(image)
        prediction, _ = test_single_case(net, im_x_y, stride_xy, stride_z, patch_size, num_classes=num_classes) 
        
        prediction = prediction.cpu().numpy().astype(np.uint8)

        categories = extract_categories(prediction)
        print(categories)

        saveprediction = sitk.GetImageFromArray(prediction)
        saveprediction.SetSpacing(sitk_im.GetSpacing())
        saveprediction.SetOrigin(sitk_im.GetOrigin())
        saveprediction.SetDirection(sitk_im.GetDirection())


        sitk.WriteImage(saveprediction, output2 + fname.split('_')[0] + "_"
                        + fname.split('_')[1] + ".nii.gz")
        if pbar:
            pbar.update(1)


def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1, pbar=None):
    w, h, d = image.shape
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    image = torch.from_numpy(image).to(device)
    net.to(device)
    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2] - d
        add_pad = True
    else:
        d_pad = 0
    w_pad = max(0, patch_size[0] - w)
    h_pad = max(0, patch_size[1] - h)
    d_pad = max(0, patch_size[2] - d)

    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
    padding_tuple = (dl_pad, dr_pad, hl_pad, hr_pad, wl_pad, wr_pad)
    if any(padding_tuple):
        image = F.pad(image, padding_tuple, mode='constant', value=0)

    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_z) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_xy) + 1
    print("{}, {}, {}".format(sx, sy, sz))
    score_map = torch.zeros((num_classes,) + image.shape, dtype=torch.float32, device=device)
    cnt = torch.zeros(image.shape, dtype=torch.float32, device=device)
    net.eval()
    with torch.no_grad():
        for x in range(0, sx):
            xs = min(stride_z * x, ww - patch_size[0])
            for y in range(0, sy):
                ys = min(stride_xy * y, hh - patch_size[1])
                for z in range(0, sz):
                    zs = min(stride_xy * z, dd - patch_size[2])
                    test_patch = image[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]]
                    test_patch = test_patch.unsqueeze(0).unsqueeze(0).float()
                    y1 = net(test_patch)
                    y = F.softmax(y1, dim=1)

                    y = y.squeeze(0) 
                    score_map[:, xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] += y
                    cnt[xs:xs + patch_size[0], ys:ys + patch_size[1], zs:zs + patch_size[2]] += 1

    score_map /= (cnt.unsqueeze(0) + 1e-7)
    label_map = torch.argmax(score_map, dim=0)
    if add_pad:
        label_map = label_map[wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
        score_map = score_map[:, wl_pad:wl_pad + w, hl_pad:hl_pad + h, dl_pad:dl_pad + d]
    return label_map, score_map

def test_calculate_metric():
    start_time = time.time()
    imdir = "./FLARE22_val"            # Flare test data
    cp_path = "./Your.pth"                            

    path1 = './predict/'               # prediction
    os.makedirs(path1, exist_ok=True)
    path2 = "./FLARE22_val_label"      # test label the validation label for flare22
    logging.basicConfig(filename= str(FLAGS.ratio) + '_flare.log',
                        level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')
    num_classes = 14                   # datasets all classes number
    net = unet_3D_mt(in_chns=1, class_num=num_classes).cuda()
    checkpoint = torch.load(cp_path, map_location='cpu', weights_only=False)
    new_state_dict = {}
    for k, v in checkpoint['model_ema'].items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v  
        else:
            new_state_dict[k] = v
    net.load_state_dict(new_state_dict, strict=False)
    logging.info("### init weight from {}".format(cp_path))
    net.eval()
    pbar = tqdm(total=FLAGS.test_num, desc="Validation", unit="file")

    test_all_case(net, imdir, path2, path1, num_classes=num_classes,
                    patch_size=(64, 160, 160), stride_xy=80, stride_z=32, save_result=True, pbar=pbar)
    
    average_accuracy, avg_jaccard = calculate_metrics(path1, path2, FLAGS)    

    for i, (cls_dice) in enumerate(average_accuracy):
        logging.info(f'{i}class {CLASSES[FLAGS.dataset][i+1]} dice:{cls_dice}')
    logging.info('dataset_{} avg_jaccard:{}'.format(FLAGS.dataset, avg_jaccard)) 
    logging.info('avg_dice:{}'.format(np.mean(average_accuracy)))
    print(f"checkpoints from: {cp_path}")
    end_time = time.time() - start_time
    print(f"Mean time for every item: {(end_time/FLAGS.test_num):.2f} seconds")
    logging.info(f"Mean time for every item: {(end_time/FLAGS.test_num):.2f} seconds")

if __name__ == '__main__':
    metric = test_calculate_metric()
    print(metric)