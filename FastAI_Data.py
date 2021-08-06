#!/user/bin/env python
# coding=utf-8
"""
@project : FastAIStudy
@author  : shanyi
#@file   : FastAI_Data.py
#@ide    : PyCharm
#@time   : 2021-08-06 09:32:56
"""

# 导入FastAI库机器视觉包
from fastai.vision import *

# 构造Image对象
# 函数参数分别为文件路径、是否归一化、转换方式、返回类型、打开文件后的回调
image_test = open_image("images/sss.png", False, 'RGB', Image, None)
# data以tensor形式储存图像像素数据、shape储存C*H*W、size为H*W
image_test.data, image_test.shape, image_test.size
# show()参数分别用于指定图对象、图大小、标题、是否隐藏坐标系、color map、是否额外显示
image_test.show(None, (3, 3), "Show_Test", True, None, None)
plt.show()
# apply_tfms用于图像变换
# 参数分别为变换列表、是否设置随机化参数、变换所需额外参数、输出尺寸、最终所要的尺寸、最终所得图像的尺寸是mult的倍数、填充方法
tfms = []
image_test.apply_tfms(tfms, True, None, None, None, None, 'reflection')
