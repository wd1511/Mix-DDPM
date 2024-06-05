import argparse
import os

import numpy as np
import torch

import os
import time
from improved_diffusion.score.both import get_inception_and_fid_score
import math

data_dir='/data/diffusion_logs/m_l_1000_f_1/test/240'
fid_cache='./stats/cifar10.train.npz'

imagelist = os.listdir(data_dir)
images = []
print('start')
for filename in imagelist:
    print(os.path.join(data_dir, filename))
    x = torch.load(os.path.join(data_dir, filename))
    images.append(x)
images = torch.cat(images, dim=0).numpy()
(IS, IS_std), FID = get_inception_and_fid_score(images, fid_cache, 50000, use_torch=False, verbose=True)
print('IS:'+'{:.2f}'.format(IS), 'IS_std:'+'{:.2f}'.format(IS_std), 'FID:'+'{:.2f}'.format(FID))
