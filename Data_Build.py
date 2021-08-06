#!/user/bin/env python
# coding=utf-8
"""
@project : FastAIStudy
@author  : shanyi
#@file   : Data_Build.py
#@ide    : PyCharm
#@time   : 2021-08-06 11:46:00
# 使用FastAI提供的函数接口，使构建数据包的流程更加符合逻辑且更灵活
"""
from fastai.vision import *

path_data = untar_data(URLs.MNIST_SAMPLE)
# 设置变换列表
# 是否进行水平翻转、是否进行垂直翻转、
transforms_one = get_transforms(flip_vert=True, max_lighting=0.1, max_zoom=1.05, max_warp=0.)

data_f = (ImageList.from_folder(path_data)  # 从目录获取数据
          .split_by_folder()  # 按照文件夹名称进行划分
          .label_from_folder()  # 由文件路径的最后一层的名称指定标签
          # .transform(transforms_one, size=32)  # 指定数据变换
          .databunch(bs=128)  # 构建数据包
          .normalize(imagenet_stats))  # 数据归一化(统计分布方式)
# data_f.show_batch(rows=3, figsize=(4, 4))
image_test = open_image("images/sss.png", False, 'RGB', Image, None)
# 改变图像明暗，通过对图像的logit pixel进行加减常量实现
image_test = brightness(image_test, 0.1)
# 调整对比度
image_test = contrast(image_test, 0.5)
# 图像裁剪
crop(), crop_pad()
# 镜像翻转、水平翻转、旋转
dihedral(), flip_lr(), rotate()
# 邻域像素替换、制作孔洞
jitter(), cutout()
# 透视变换
perspective_warp(), symmetric_warp()
# 图像缩放
image_test.resize((3, 28, 28)), zoom()
# 扭曲、拉伸、倾斜
skew(), squish(), tilt()
image_test.show(figsize=(3, 3))
plt.show()
