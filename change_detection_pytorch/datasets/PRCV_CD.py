import os.path as osp

import albumentations as A
from albumentations.pytorch import ToTensorV2
from .custom import CustomDataset
from .transforms.albu import ExchangeTime, Mosaic


class PRCV_CD_Dataset(CustomDataset):
    """PRCV-CD dataset"""

    def __init__(self, img_dir, sub_dir_1='image1', sub_dir_2='image2', ann_dir=None, img_suffix='.png',
                 seg_map_suffix='.png',
                 transform=None, split=None, data_root=None, test_mode=False, size=512, debug=False):
        super().__init__(img_dir, sub_dir_1, sub_dir_2, ann_dir, img_suffix, seg_map_suffix, transform, split,
                         data_root, test_mode, size, debug)

    def get_default_transform(self):
        """Set the default transformation."""

        transform = A.Compose([
            A.RandomResizedCrop(self.size, self.size, p=0.8),

            A.OneOf([
                A.GaussNoise(p=1),
                A.Blur(p=1),
                A.IAASharpen(p=1),
                A.ISONoise(p=1),
                A.MotionBlur(p=1),
                A.IAAEmboss(),
                A.CLAHE(),
                # A.RandomBrightnessContrast(p=1),  # here
                A.NoOp(p=1),
            ], p=1),

            A.OneOf([
                A.RandomGridShuffle(grid=(2, 2), p=1),
                # Mosaic(256, self.img_infos, self.__len__(), p=1),
                A.NoOp(p=1),
            ], p=0.3),

            A.Normalize(),
            ToTensorV2()
        ], additional_targets={'image_2': 'image'})

        return transform

    def get_test_transform(self):
        """Set the test transformation."""

        test_transform = A.Compose([
            A.Normalize(),
            ToTensorV2(),
        ], additional_targets={'image_2': 'image'})
        return test_transform

    def __getitem__(self, idx):
        """Get training/test data after pipeline.
        Args:
            idx (int): Index of data.
        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                False).
        """

        if not self.ann_dir:
            img1, img2, filename = self.prepare_img(idx)
            transformed_data = self.transform(image=img1, image_2=img2)
            img1, img2 = transformed_data['image'], transformed_data['image_2']
            return img1, img2, filename

        else:
            img1, img2, ann, filename = self.prepare_img_ann(idx)
            transformed_data = self.transform(image=img1, image_2=img2, mask=ann)
            img1, img2, ann = transformed_data['image'], transformed_data['image_2'], transformed_data['mask']

        return img1, img2, ann, filename
