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

path = untar_data(URLs.MNIST_SAMPLE)
data_f = (ImageList.from_folder(path)  # 从目录获取数据
          .split_by_folder()  # 按照文件夹名称进行划分
          .label_from_folder()  # 由文件路径的最后一层的名称指定标签
          .transform(size=28)  # 指定数据变换
          .databunch()  # 构建数据包
          .normalize(imagenet_stats))  # 数据归一化(统计分布方式)
data_f.show_batch(rows=3, figsize=(4, 4))
plt.show()
