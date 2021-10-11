import os
import cv2
import numpy as np
import tqdm

class Metric_clu:
    __name__ = "miou_oa"

    def __init__(self, threshold=0.5):
        super().__init__()
        self.eps = 1e-8
        self.threshold = threshold*255.0
        self.tp = 0.
        self.tn = 0.
        self.fn = 0.
        self.fp = 0.
        self.mIou = 0
        self.oa = 0
        self.res = None

    def reset(self):
        self.tp = 0.
        self.tn = 0.
        self.fn = 0.
        self.fp = 0.

    def __call__(self, prediction, target):
        prediction = (prediction > self.threshold)
        target = (target > self.threshold)
        self.tp += np.sum(prediction * target)
        self.tn += np.sum((prediction == 0) * (target == 0))
        self.fn += np.sum((prediction == 0) * (target == 1))
        self.fp += np.sum((prediction == 1) * (target == 0))

        self.iou_0 = self.tp / (self.tp + self.fp + self.fn + self.eps)
        self.iou_1 = self.tn / (self.tn + self.fp + self.fn + self.eps)
        self.mIou = 0.5 * self.iou_0 + 0.5 * self.iou_1
        self.oa = (self.tp + self.tn)/(self.tp + self.tn + self.fp + self.fn)
        self.res = {'miou = ': self.mIou, 'oa = ': self.oa, 'iou_0 = ': self.iou_0, 'iou_1 = ': self.iou_1}
        return self.res


model = 'val'
pred_path = 'C:/Users/likyoo/Desktop/res1'
threshold = 0.5


pre_path = pred_path
gtpath = 'F:/PRCV_CD/train_val/val_set'

if model != 'val':
    gtpath = '/config_data/dataset/PRCV2021/val_set'
    pre_path = os.path.join(pred_path, 'testres')

gt_path = os.path.join(gtpath, 'label')
imlist = os.listdir(pre_path)
change_metric = Metric_clu(threshold)

for imname in tqdm.tqdm(imlist):
    prec = cv2.imread(os.path.join(pre_path, imname), -1)
    gtc = cv2.imread(os.path.join(gt_path, imname), -1)
    change_metric(prec, gtc)

print(change_metric.res)
