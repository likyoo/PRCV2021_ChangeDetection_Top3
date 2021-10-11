import json
import os
import os.path as osp
import random
import sys
import time

import cv2
import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, StepLR
from torch.utils.data import DataLoader, Dataset

import change_detection_pytorch as cdp
from change_detection_pytorch.datasets import LEVIR_CD_Dataset, SVCD_Dataset
from change_detection_pytorch.datasets.PRCV_CD import PRCV_CD_Dataset
from change_detection_pytorch.utils.lr_scheduler import GradualWarmupScheduler


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

seed_torch(seed=1024)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(DEVICE)

# test_dataset = PRCV_CD_Dataset('/cache/train_val/val_set',
#                                  sub_dir_1='image1',
#                                  sub_dir_2='image2',
#                                  img_suffix='.png',
#                                  ann_dir='/cache/train_val/val_set/label',
#                                  size=512,
#                                  debug=False,
#                                  test_mode=True)

test_dataset = PRCV_CD_Dataset('/cache/test_set/test_set',
                                 sub_dir_1='image1',
                                 sub_dir_2='image2',
                                 img_suffix='.png',
                                 ann_dir=None,
                                 size=512,
                                 debug=False,
                                 test_mode=True)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)



model_path_1 = './final_models/final_unet.pth'
save_dir = './res'

model = torch.load(model_path_1)

start = time.time()

with torch.no_grad():

    model.eval()

    for (x1, x2, filename) in test_loader:

        x1, x2 = x1.float(), x2.float()
        x1, x2 = x1.to(DEVICE), x2.to(DEVICE)

        y_pred = model.forward(x1, x2)
        if not isinstance(y_pred, torch.Tensor):
            y_pred = y_pred[-1]

        y_pred = torch.argmax(y_pred, dim=1).squeeze().cpu().numpy().round()
        y_pred = y_pred * 255
        filename = filename[0].split('.')[0] + '.png'

        cv2.imwrite(osp.join(save_dir, filename), y_pred)


end = time.time()
print('time: ', end - start)

