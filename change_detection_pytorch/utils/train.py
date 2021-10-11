import os.path as osp
import sys

import torch
from tqdm import tqdm as tqdm
import numpy as np
import torchvision.transforms.functional as F

from .meter import AverageValueMeter


class Epoch:

    def __init__(self, model, loss, metrics, stage_name, device='cpu', verbose=True, scale=512):
        self.model = model
        self.loss = loss
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device
        self.scale = scale
        self.multi_scale = False
        self.scaler = torch.cuda.amp.GradScaler()

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.loss.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s

    def batch_update(self, x1, x2, y, epoch, len_data):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def check_tensor(self, data, is_label):
        if not is_label:
            return data if data.ndim <= 4 else data.squeeze()
        return data if data.ndim <= 3 else data.squeeze()

    def infer_vis(self, dataloader, save=True, evaluate=False, slide=False, image_size=1024,
                  window_size=256, save_dir='./res', suffix='.tif'):
        """
        Infer and save results.
        Note: Currently only batch_size=1 is supported.
        Weakly robust.
        'image_size' and 'window_size' work when slide is True.
        """
        import cv2
        import numpy as np

        self.model.eval()
        logs = {}
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for (x1, x2, y, filename) in iterator:

                assert y is not None or not evaluate, "When the label is None, the evaluation mode cannot be turned on."

                x1, x2, y = self.check_tensor(x1, False), self.check_tensor(x2, False), \
                            self.check_tensor(y, True)
                x1, x2, y = x1.float(), x2.float(), y.long()
                x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)
                y_pred = self.model.forward(x1, x2)

                if not isinstance(y_pred, torch.Tensor):
                    y_pred = y_pred[-1]

                if evaluate:
                    # update metrics logs
                    for metric_fn in self.metrics:
                        metric_value = metric_fn(y_pred, y).detach().cpu().numpy()
                        metrics_meters[metric_fn.__name__].add(metric_value)
                    metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                    logs.update(metrics_logs)

                    if self.verbose:
                        s = self._format_logs(logs)
                        iterator.set_postfix_str(s)

                if save:

                    y_pred = torch.argmax(y_pred, dim=1).squeeze().cpu().numpy().round()
                    y_pred = y_pred * 255

                    filename = filename[0].split('.')[0] + suffix

                    if slide:
                        inf_seg_maps = []
                        window_num = image_size // window_size
                        window_idx = [i for i in range(0, window_num ** 2 + 1, window_num)]
                        for row_idx in range(len(window_idx) - 1):
                            inf_seg_maps.append(np.concatenate([y_pred[i] for i in range(window_idx[row_idx],
                                                                                         window_idx[row_idx + 1])], axis=1))
                        inf_seg_maps = np.concatenate([row for row in inf_seg_maps], axis=0)
                        cv2.imwrite(osp.join(save_dir, filename), inf_seg_maps)
                    else:
                        # To be verified
                        cv2.imwrite(osp.join(save_dir, filename), y_pred)

            print(logs)

    def random_scale(self, x1, x2, y, img_scale, ratio_range, div=32):
        def random_sample_ratio(img_scale, ratio_range):
            """Randomly sample an img_scale when ``ratio_range`` is specified.
            A ratio will be randomly sampled from the range specified by
            ``ratio_range``. Then it would be multiplied with ``img_scale`` to
            generate sampled scale.
            Args:
                img_scale (tuple): Images scale base to multiply with ratio.
                ratio_range (tuple[float]): The minimum and maximum ratio to scale
                    the ``img_scale``.
            Returns:
                (tuple, None): Returns a tuple ``(scale, None)``, where
                    ``scale`` is sampled ratio multiplied with ``img_scale`` and
                    None is just a placeholder to be consistent with
                    :func:`random_select`.
            """

            assert isinstance(img_scale, tuple) and len(img_scale) == 2
            min_ratio, max_ratio = ratio_range
            assert min_ratio <= max_ratio
            ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
            scale = [int(img_scale[0] * ratio), int(img_scale[1] * ratio)]
            return scale

        scale = random_sample_ratio(img_scale, ratio_range)
        scale = [(s // div) * div for s in scale]
        x1 = F.resize(x1, scale)
        x2 = F.resize(x2, scale)
        y = F.resize(y, scale)

        return x1, x2, y


    def run(self, dataloader):

        # 混合精度加速
        batch_i = 0
        len_data = len(dataloader)

        # 多尺度
        if isinstance(self.scale, (list, tuple)):
            self.multi_scale = True

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in self.metrics}

        with tqdm(dataloader, desc=self.stage_name, file=sys.stdout, disable=not (self.verbose)) as iterator:
            for (x1, x2, y, filename) in iterator:

                x1, x2, y = self.check_tensor(x1, False), self.check_tensor(x2, False), \
                            self.check_tensor(y, True)

                if self.multi_scale == True:
                    x1, x2, y = self.random_scale(x1, x2, y, x1.shape[2:], self.scale)

                x1, x2, y = x1.float(), x2.float(), y.long()
                x1, x2, y = x1.to(self.device), x2.to(self.device), y.to(self.device)

                loss, y_pred = self.batch_update(x1, x2, y, batch_i, len_data)
                batch_i += 1

                # update loss logs
                loss_value = loss.detach().cpu().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__name__: loss_meter.mean}
                logs.update(loss_logs)

                if not isinstance(y_pred, torch.Tensor):
                    y_pred = y_pred[-1]
                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).detach().cpu().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs


class TrainEpoch(Epoch):

    def __init__(self, model, loss, metrics, optimizer, device='cpu', verbose=True, extra_pre=False, accumulation=False, scale=512):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='train',
            device=device,
            verbose=verbose,
            scale=scale,
        )
        self.optimizer = optimizer
        self.extra_pre = extra_pre
        self.accumulation = accumulation

    def on_epoch_start(self):
        self.model.train()

    def batch_update(self, x1, x2, y, batch_idx, len_data):
        prediction = self.model.forward(x1, x2)

        if self.accumulation:

            accumulation_steps = 2
            loss = self.loss(prediction, y)

            # 剃度累加
            loss = loss / accumulation_steps
            self.scaler.scale(loss).backward()  # 将loss缩放

            # 梯度累加
            index_add = batch_idx + 1
            if index_add % accumulation_steps == 0 or index_add == len_data:
                # self.optimizer.step()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            return loss, prediction
        # elif isinstance(prediction, torch.Tensor):  # not self.extra_pre:
        else:
            self.optimizer.zero_grad()
            loss = self.loss(prediction, y)
            loss.backward()
            self.optimizer.step()
            return loss, prediction
        # else:  # for OCR head
        #     self.optimizer.zero_grad()
        #     loss = self.loss(prediction[0], y) * 0.4 + self.loss(prediction[1], y)
        #     loss.backward()
        #     self.optimizer.step()
        #     return loss, prediction[-1]



class ValidEpoch(Epoch):

    def __init__(self, model, loss, metrics, device='cpu', verbose=True, extra_pre=False, TTA=False):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name='valid',
            device=device,
            verbose=verbose,
        )
        self.TTA = TTA
        self.extra_pre = extra_pre

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x1, x2, y, epoch, len_data):
        with torch.no_grad():
            prediction = self.model.forward(x1, x2, TTA=self.TTA)
            loss = self.loss(prediction, y)
            return loss, prediction
            # if isinstance(prediction, torch.Tensor):  # not self.extra_pre:
            #     loss = self.loss(prediction, y)
            #     return loss, prediction
            # else:  # for OCR head
            #     loss = self.loss(prediction[0], y) * 0.4 + self.loss(prediction[1], y)
            #     return loss, prediction[-1]
