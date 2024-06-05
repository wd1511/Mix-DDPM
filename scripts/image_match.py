import cv2
import numpy as np
from skimage.exposure import match_histograms

import argparse
import os

import numpy as np
import torch

import os
import time
from improved_diffusion.score.both import get_inception_and_fid_score
import math
from skimage.exposure import match_histograms

data_dir1='/data/diffusion_logs/s_l_4000_t_1/test/80'
data_dir2='/data/diffusion_logs/m_l_4000_t_1/test/80'
fid_cache='./stats/cifar10.train.npz'

image_size = 32

imagelist1 = os.listdir(data_dir1)
images1 = []
print('start')
print('Image1')
for filename in imagelist1:
    print(os.path.join(data_dir1, filename))
    x = torch.load(os.path.join(data_dir1, filename))
    images1.append(x)
images1_sum = torch.cat(images1, dim=0).permute(1,2,3,0).reshape(3,image_size,-1).numpy()
images1 = torch.cat(images1, dim=0).numpy()

imagelist2 = os.listdir(data_dir2)
images2 = []
print('Image2')
for filename in imagelist2:
    print(os.path.join(data_dir2, filename))
    x = torch.load(os.path.join(data_dir2, filename))
    images2.append(x)
images2_sum = torch.cat(images2, dim=0).permute(1,2,3,0).reshape(3,image_size,-1).numpy()
images2 = torch.cat(images2, dim=0).numpy()

(IS1, IS_std1), FID1 = get_inception_and_fid_score(images1, fid_cache, 50000, use_torch=False, verbose=True)
(IS2, IS_std2), FID2 = get_inception_and_fid_score(images2, fid_cache, 50000, use_torch=False, verbose=True)

print('match')
matched = []
for i in range(3):
    print(i)
    x = images1_sum[i,:,:]
    y = images2_sum[i,:,:]
    z = match_histograms(y, x)
    z = torch.from_numpy(z).view(1, image_size, -1)
    matched.append(z)
matched = torch.cat(matched, dim=0).reshape(3,image_size,image_size,-1).permute(3,0,1,2).numpy()

(IS3, IS_std3), FID3 = get_inception_and_fid_score(matched, fid_cache, 50000, use_torch=False, verbose=True)

print('Image_s:')
print('IS:'+'{:.2f}'.format(IS1), 'IS_std:'+'{:.2f}'.format(IS_std1), 'FID:'+'{:.2f}'.format(FID1))
print('Image_m:')
print('IS:'+'{:.2f}'.format(IS2), 'IS_std:'+'{:.2f}'.format(IS_std2), 'FID:'+'{:.2f}'.format(FID2))
print('Matched:')
print('IS:'+'{:.2f}'.format(IS3), 'IS_std:'+'{:.2f}'.format(IS_std3), 'FID:'+'{:.2f}'.format(FID3))