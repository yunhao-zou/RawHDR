import os
import time

import numpy as np
from py import process
import rawpy
import glob

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import lpips

from PIL import Image
import scipy.io as sio
from model import RawHDR
from torch.nn.modules.loss import _Loss
from process import process
import re



def findLastCheckpoint(save_dir):
    file_list = glob.glob(os.path.join(save_dir, '*.pth'))
    if file_list:
        epochs_exist = []
        for file_ in file_list:
            result = re.findall(".*checkpoint_canon_e(.*).pth", file_)
            epochs_exist.append(int(result[0]))
        initial_epoch = max(epochs_exist)
    else:
        initial_epoch = 0
    return initial_epoch

def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode='fan_in')
        nn.init.constant_(m.bias, 0.0)

train_dir = '/data1/HDR/MAT_train/'
test_dir = '/data1/HDR/MAT_test/'
result_dir = './train_results/canon_logl2/'
model_dir = './saved_model/canon_logl2/'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

#get train and test IDs
train_fns = glob.glob(train_dir + '7U*.mat')
train_ids = []  # [3449, ...]
for i in range(len(train_fns)):
    _, train_fn = os.path.split(train_fns[i])
    train_ids.append(int(train_fn[4:8]))

test_fns = glob.glob(test_dir + '7U*.mat')
test_ids = []   # [3449, ...]
for i in range(len(test_fns)):
    _, test_fn = os.path.split(test_fns[i])
    test_ids.append(int(test_fn[4:8]))



ps = 256 #patch size for training
save_freq = 100


def reduce_mean(out_im, gt_im):
    return torch.abs(out_im - gt_im).mean()

# loss function
class l1_loss():
    def __init__(self, weight=1.0):
        self.loss = F.l1_loss
        self.weight = weight

    def __call__(self, pred, target):
        return self.weight * self.loss(pred, target)

class log_l1_loss():
    def __init__(self, weight=1.0):
        self.loss = F.l1_loss
        self.weight = weight

    def __call__(self, pred, target):
        return self.weight * self.loss(torch.log(torch.clamp(pred, min=0)+1/5000), torch.log(target+1/5000))

class log_l2_loss():
    def __init__(self, weight=1.0):
        self.loss = F.mse_loss
        self.weight = weight

    def __call__(self, pred, target):
        return self.weight * self.loss(torch.log(torch.clamp(pred, min=0)+1/5000), torch.log(target+1/5000))

class lpips_loss():
    def __init__(self, net='alex', weight=1.0):
        self.loss_fn = lpips.LPIPS(net=net)
        self.weight = weight

    def __call__(self, pred, target):
        # LPIPS expects inputs in range [-1, 1]
        # Convert from [0, 1] to [-1, 1] 
        pred_norm = pred * 2.0 - 1.0
        target_norm = target * 2.0 - 1.0
        return self.weight * self.loss_fn(pred_norm, target_norm).mean()


#Raw data takes long time to load. Keep them in memory after loaded.
training_data = {}

loss_logl2 = log_l2_loss(weight=1)
loss_l1 = l1_loss(weight=1)
loss_mask = l1_loss(weight=0.5)
loss_lpips = lpips_loss(weight=0.1)  # Using smaller weight for LPIPS as it typically has larger magnitudes

g_loss = np.zeros((5000,1))

# model = UNet()
# model = RawHDR_woTR(256)
model = RawHDR(256, RB_gudie=True, G_guidance=True, RGB=False, softmask=True, softblending=False)
# model._initialize_weights()
model.apply(weights_init)
lastepoch = findLastCheckpoint(model_dir)

if lastepoch > 0:
    print('resuming by loading epoch %03d' % lastepoch)
    checkpoint = torch.load(os.path.join(model_dir, 'checkpoint_canon_e%04d.pth' % lastepoch))
    model.load_state_dict(checkpoint['net'])

learning_rate = 1e-4
model = model.to(device)
loss_lpips.loss_fn = loss_lpips.loss_fn.to(device)

opt = optim.Adam(model.parameters(), lr = learning_rate)
for epoch in range(lastepoch, 2001):
    if os.path.isdir("result/%04d"%epoch):
        continue    
    cnt=0
    if epoch > 800:
        for g in opt.param_groups:
            g['lr'] = 1e-5
  
    losses_over = []
    losses_under = []
    losses_log = []
    losses_l1 = []
    losses_lpips = []
    for ind in np.random.permutation(len(train_ids)):
        # get the path from image id
        train_id = train_ids[ind]
        # in_files = glob.glob(train_dir + '7U6A%04d.mat'%train_id)
        # in_path = in_files[np.random.randint(0,len(in_files))]
        in_path = os.path.join(train_dir, '7U6A%04d.mat'%train_id)
        # _, in_fn = os.path.split(in_path)

        # gt_files = glob.glob(gt_dir + '%05d_00*.ARW'%train_id)
        # gt_path = gt_files[0]
        # _, gt_fn = os.path.split(gt_path)
        # in_exposure =  float(in_fn[9:-5])
        # gt_exposure =  float(gt_fn[9:-5])
        # ratio = min(gt_exposure/in_exposure,300)
          
        st=time.time()
        cnt+=1

        if train_id not in training_data:
            # data = sio.loadmat(in_path)
            # input_images[ind] = np.expand_dims(data['input'], axis=0) # 1*C*H*W
            # gt_images[ind] = np.expand_dims(data['gt'], axis=0)  # C*H*W

            training_data[train_id] = {}
            in_path = os.path.join(train_dir, '7U6A%04d.mat'%train_id)
            data = sio.loadmat(in_path)
            training_data[train_id]['input'] = np.expand_dims(data['input'], axis=0) # 1*C*H*W
            training_data[train_id]['gt'] = np.expand_dims(data['gt'], axis=0) # 1*C*H*W
            training_data[train_id]['wb'] = data['wb']
            training_data[train_id]['pattern'] = data['pattern']
            training_data[train_id]['cam2rgb'] = data['cam2rgb']

         
        #crop
        H = training_data[train_id]['input'].shape[2]
        W = training_data[train_id]['input'].shape[3]

        yy = np.random.randint(0, H-ps)
        xx = np.random.randint(0, W-ps)
        input_patch = training_data[train_id]['input'][:, :, yy:yy+ps, xx:xx+ps]
        gt_patch = training_data[train_id]['gt'][:, :, yy:yy+ps, xx:xx+ps]
       

        if np.random.randint(2,size=1)[0] == 1:  # random flip 
            input_patch = np.flip(input_patch, axis=2)
            gt_patch = np.flip(gt_patch, axis=2)
        if np.random.randint(2,size=1)[0] == 1: 
            input_patch = np.flip(input_patch, axis=3)
            gt_patch = np.flip(gt_patch, axis=3)
        if np.random.randint(2,size=1)[0] == 1:  # random transpose 
            input_patch = np.transpose(input_patch, (0, 1, 3, 2))
            gt_patch = np.transpose(gt_patch, (0, 1, 3, 2))
        
        input_patch = np.ascontiguousarray(input_patch)
        gt_patch = np.ascontiguousarray(gt_patch)
        # input_patch = np.minimum(input_patch,1.0)
        # gt_patch = np.maximum(gt_patch, 0.0)
        
        in_img = torch.from_numpy(input_patch).to(device)
        gt_img = torch.from_numpy(gt_patch).to(device)

        max_channel = torch.max(in_img, dim=1)[0]
        min_channel = torch.min(in_img, dim=1)[0]
        mask_over_h = torch.zeros_like(max_channel)
        mask_over_h[max_channel>0.6] = 1
        mask_under_h = torch.zeros_like(min_channel)
        mask_under_h[min_channel<0.001] = 1
        # print(torch.min(in_img), torch.max(in_img))
        # print(torch.min(gt_img), torch.max(gt_img))
        opt.zero_grad()
        out_img, mask_over, mask_under = model(in_img)

        log_loss = loss_logl2(out_img, gt_img)
        l1_loss = loss_l1(out_img, gt_img)
        loss_over = loss_mask(mask_over, mask_over_h)
        loss_under = loss_mask(mask_under, mask_under_h)
        perceptual_loss = loss_lpips(out_img, gt_img)
        loss = log_loss + l1_loss + loss_over + loss_under + perceptual_loss  # 1 * C * H * W
        loss.backward()

        opt.step()
        g_loss[ind]=loss.data.cpu()
        losses_log.append(log_loss.data.cpu().numpy())
        losses_l1.append(l1_loss.data.cpu().numpy())
        losses_over.append(loss_over.data.cpu().numpy())
        losses_under.append(loss_under.data.cpu().numpy())
        losses_lpips.append(perceptual_loss.data.cpu().numpy())

        mean_loss = np.mean(g_loss[np.where(g_loss)])
        print(f"Epoch: {epoch} \t Count: {cnt} \t TotalLoss={mean_loss:.4} LogLoss={np.mean(losses_log):.4} L1Loss={np.mean(losses_l1):.4} OverLoss={np.mean(losses_over):.4} UnderLoss={np.mean(losses_under):.4} LPIPS={np.mean(losses_lpips):.4} \t Time={time.time()-st:.3}")
        

        if epoch%save_freq==0:
            epoch_result_dir = result_dir + f'{epoch:04}/'

            if not os.path.isdir(epoch_result_dir):
                os.makedirs(epoch_result_dir)

            # output = out_img.permute(0, 2, 3, 1).cpu().data#.numpy()
            # output = np.minimum(np.maximum(output,0),1)

            wb = torch.from_numpy(training_data[train_id]['wb']).float()
            cam2rgb = torch.from_numpy(training_data[train_id]['cam2rgb']).float()
            out_hdr = process(out_img.cpu().detach()*10, wbs=wb[None], cam2rgbs=cam2rgb[None], gamma=2.2, use_demosaic=False)[0].numpy().transpose((1,2,0))
            gt_hdr = process(gt_img.cpu().detach()*10, wbs=wb[None], cam2rgbs=cam2rgb[None], gamma=2.2, use_demosaic=False)[0].numpy().transpose((1,2,0))
            in_hdr = process(in_img.cpu().detach()*10, wbs=wb[None], cam2rgbs=cam2rgb[None], gamma=2.2, use_demosaic=False)[0].numpy().transpose((1,2,0))

            out_hdr = (out_hdr*255).astype(np.uint8)
            gt_hdr = (gt_hdr*255).astype(np.uint8)
            in_hdr = (in_hdr*255).astype(np.uint8)

            temp = np.concatenate((in_hdr, out_hdr, gt_hdr),axis=1)
            Image.fromarray(temp).save(epoch_result_dir + f'{train_id:04}_00_train.jpg')

    if epoch%save_freq==0:
        state = {
        	'net': model.state_dict(),
        	'optimizer': opt.state_dict(),
        	'epoch': epoch+1,
        }
        torch.save(state, model_dir+'checkpoint_canon_e%04d.pth'%epoch)

