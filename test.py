import os
import time

import numpy as np
import rawpy
import glob

import torch
import torch.nn as nn
from model import UNet, RawHDR, RawHDR_woTR
from process import process
from skimage.metrics import structural_similarity as ssim
import scipy.io as sio
import torch.optim as optim
from pytorch_msssim import ms_ssim
import cv2

from PIL import Image

from model import UNet

name = 'canon_logl2_rawhdr'
m_path = './saved_model/{}/'.format(name)
m_name = 'checkpoint_canon_e2500.pth'

result_dir = './test_results/{}/'.format(name)
fig_root= './fig_results/'
test_dir = '/data/HDR/MAT_test/'

test_fns = sorted(glob.glob(test_dir + '7U*.mat'))
test_ids = []   # [3449, ...]
for i in range(len(test_fns)):
    _, test_fn = os.path.split(test_fns[i])
    test_ids.append(int(test_fn[4:8]))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

model = RawHDR(1024, RB_gudie=True, G_guidance=True, RGB=False, softmask=True, softblending=False)
# model = RawHDR_woTR(2048)
model.load_state_dict(torch.load(m_path + m_name)['net'])
print('model loaded')
model = model.to(device)

def mu_tonemap(hdr_image, mu=5000):
    """ This function computes the mu-law tonemapped image of a given input linear HDR image.
    Args:
        hdr_image (np.ndarray): Linear HDR image with values in the range of [0-1]
        mu (float): Parameter controlling the compression performed during tone mapping.
    Returns:
        np.ndarray (): Returns the mu-law tonemapped image.
    """
    return np.log(1 + mu * hdr_image) / np.log(1 + mu)

def norm_mu_tonemap(hdr_image, norm_value, mu=5000):
    """ This function normalizes the input HDR linear image by the specified norm_value and then computes
    the mu-law tonemapped image.
    Args:
        hdr_image (np.ndarray): Linear HDR image with values in the range of [0-1]
        norm_value (float): Value for the normalization (i.e. hdr_image/norm_value)
        mu (float): Parameter controlling the compression performed during tone mapping.
    Returns:
        np.ndarray (): Returns the mu-law tonemapped image.
    """
    return mu_tonemap(hdr_image/norm_value, mu)

def psnr(im0, im1):
    """ This function computes the Peak Signal to Noise Ratio (PSNR) between two images whose ranges are [0-1].
        the mu-law tonemapped image.
        Args:
            im0 (np.ndarray): Image 0, should be of same shape and type as im1
            im1 (np.ndarray: Image 1,  should be of same shape and type as im0
        Returns:
            np.ndarray (): Returns the mean PSNR value for the complete image.
        """
    return -10*np.log10(np.mean(np.power(im0-im1, 2)))

def normalized_psnr(im0, im1, norm):
    """ This function computes the Peak Signal to Noise Ratio (PSNR) between two images that are normalized by the
    specified norm value.
        the mu-law tonemapped image.
        Args:
            im0 (np.ndarray): Image 0, should be of same shape and type as im1
            im1 (np.ndarray: Image 1,  should be of same shape and type as im0
            norm (float) : Normalization value for both images.
        Returns:
            np.ndarray (): Returns the mean PSNR value for the complete image.
        """
    return psnr(im0/norm, im1/norm)

def get_metric(img1, img2):
    # calculate PSNR, PSNR-mu, SSIM, HDR for C*H*W numpy images
    img1 = np.clip(img1, 0, 8)
    img2 = np.clip(img2, 0, 8)
    PSNR = psnr(img1, img2)
    mu_img1 = norm_mu_tonemap(img1, 8, 5000)
    mu_img2 = norm_mu_tonemap(img2, 8, 5000)
    PSNR_mu = psnr(mu_img1, mu_img2)
    SSIM = ssim(mu_img1, mu_img2, data_range=1, multichannel=True)
    MS_SSIM = ms_ssim(torch.from_numpy(mu_img1[None].transpose(0, 3, 1, 2)), torch.from_numpy(mu_img2[None].transpose(0, 3, 1, 2)), data_range=1).numpy()
    return PSNR, PSNR_mu, SSIM, float(MS_SSIM)

if not os.path.isdir(result_dir):
    os.makedirs(result_dir)

results = []
for i, test_id in enumerate(test_ids):
    print(f'{i} / {len(test_ids)}')
    model.eval()
    with torch.no_grad():
        #test the first image in each sequence
        in_path = os.path.join(test_dir, '7U6A%04d.mat'%test_id)
        data = sio.loadmat(in_path)
        input_full = np.expand_dims(data['input'], axis=0) # 1*C*H*W
        gt_full = np.expand_dims(data['gt'], axis=0) # 1*C*H*W
        wb = data['wb']
        pattern = data['pattern']
        cam2rgb = data['cam2rgb']

        ps = 2048
        H, W = input_full.shape[2:]
        yy = (H - ps) // 2
        xx = (W - ps) // 2
        input_patch = input_full[:, :, yy:yy+ps, xx:xx+ps]
        gt_patch = gt_full[:, :, yy:yy+ps, xx:xx+ps]

        in_img = torch.from_numpy(input_patch).to(device)
        gt_img = torch.from_numpy(gt_patch).to(device)
        
        max_channel = torch.max(in_img, dim=1)[0]
        min_channel = torch.min(in_img, dim=1)[0]
        mask_over_h = torch.zeros_like(max_channel)
        mask_over_h[max_channel>0.6] = 1
        mask_under_h = torch.zeros_like(min_channel)
        mask_under_h[min_channel<0.001] = 1

        out_img, mask_over, mask_under = model(in_img)

        wb = torch.from_numpy(wb).float()
        cam2rgb = torch.from_numpy(cam2rgb).float()

        mu = torch.from_numpy(np.array([5000.])).cuda()
        out_mu = torch.log(1 + mu*out_img) / torch.log(1 + mu)
        gt_mu = torch.log(1 + mu*gt_img) / torch.log(1 + mu)
        in_mu = torch.log(1 + mu*in_img) / torch.log(1 + mu)
        
        out_hdr = process(out_img.cpu().detach(), wbs=wb[None], cam2rgbs=cam2rgb[None], gamma=2.2, use_demosaic=False, use_tonemapping=True, data_range=8.0)[0].numpy().transpose((1,2,0))
        gt_hdr = process(gt_img.cpu().detach(), wbs=wb[None], cam2rgbs=cam2rgb[None], gamma=2.2, use_demosaic=False, use_tonemapping=True, data_range=8.0)[0].numpy().transpose((1,2,0))
        in_hdr = process(in_img.cpu().detach()*3, wbs=wb[None], cam2rgbs=cam2rgb[None], gamma=1.5, use_demosaic=False, use_tonemapping=False, data_range=1.0)[0].numpy().transpose((1,2,0))


        # # For comparison with rgb
        # out_img = process(out_img/8, wbs=wb.cuda()[None], cam2rgbs=cam2rgb.cuda()[None], gamma=2.2, use_demosaic=False)
        # gt_img = process(gt_img/8, wbs=wb.cuda()[None], cam2rgbs=cam2rgb.cuda()[None], gamma=2.2, use_demosaic=False)


        gt_np = gt_img.cpu().detach().numpy()[0].transpose((1,2,0))
        out_np = out_img.cpu().detach().numpy()[0].transpose((1,2,0))
        mask_over = mask_over.cpu().detach().numpy()[0]
        mask_under = mask_under.cpu().detach().numpy()[0]
        mask_over_h = mask_over_h.cpu().detach().numpy()[0]
        mask_under_h = mask_under_h.cpu().detach().numpy()[0]
        metrics = get_metric(gt_np, out_np)
        print(metrics)
        results.append(np.array(metrics))
        # metrics = psnr(mu_tonemap(gt_np, 5000),mu_tonemap(out_np, 5000))
        # print(metrics)
        out_hdr = (out_hdr*255).astype(np.uint8)
        gt_hdr = (gt_hdr*255).astype(np.uint8)
        in_hdr = (in_hdr*255).astype(np.uint8)

        fig_dir = os.path.join(fig_root, f'{test_id:04}')
        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)
        Image.fromarray(out_hdr).save(os.path.join(fig_dir, name+'.png'))
        Image.fromarray(gt_hdr).save(os.path.join(fig_dir, 'gt.png'))
        Image.fromarray(in_hdr).save(os.path.join(fig_dir, 'input.png'))
        temp = np.concatenate((in_hdr, out_hdr, gt_hdr),axis=1)
        Image.fromarray(temp).save(result_dir + f'{test_id:04}_hdr_test.jpg')

result = np.array(results)
np.save(os.path.join(result_dir, 'result.npy'), result)
avg_result = np.mean(result, 0)
# print(avg_result)
result_str = '{:<20}  PSNR: {:<10f}  PSNR-mu: {:<10f}  SSIM: {:<10f}  MS-SSIM: {:<10f}'.format(name, avg_result[0], avg_result[1], avg_result[2], avg_result[3])
print(result_str)
f = open('RESULT.txt', 'a')
f.write(result_str)
f.close()

