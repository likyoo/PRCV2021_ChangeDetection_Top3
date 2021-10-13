# PRCV2021_ChangeDetection_Top3
PRCV2021变化检测竞赛第三名解决方案



本仓库基于[change_detection.pytorch](https://github.com/likyoo/change_detection.pytorch)，部分代码还未经整理，可读性可能较差。后续会将比较有效的模型和策略添加到**change_detection.pytorch**中，欢迎star！:)



#### 单模方案（test mIOU 0.8516）：

siamese [regnet-y](https://arxiv.org/pdf/2003.13678.pdf) + ufpn (unet & deep_supervision) + [eca](https://arxiv.org/abs/1910.03151) + [conditional classifier](https://arxiv.org/pdf/2109.10322.pdf) 

具体细节见`tran.py`和`datasets/PRCV_CD.py`



#### 集成方案（test mIOU 0.8573）：

ufpn + linknet + deeplabV3+ + unet++



技术报告将在稍后公布