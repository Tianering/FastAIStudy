#!/user/bin/env python
# coding=utf-8
"""
@project : FastAIStudy
@author  : shanyi
#@file   : Date_Fun.py
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
# image_test.show(None, (3, 3), "Show_Test", True, None, None)
# 调用plt.show显示图像窗口（未找到原因）
# plt.show()
# apply_tfms用于图像变换
# 参数分别为变换列表、是否设置随机化参数、变换所需额外参数、输出尺寸、最终所要的尺寸、最终所得图像的尺寸是mult的倍数、填充方法
tfms = []
image_test.apply_tfms(tfms, True, None, None, None, None, 'reflection')

# vision.ImageDataBunch
# 包含训练集、验证集、测试集的数据迭代器

# 下载数据文件到~/.fastai/data目录
path = untar_data(URLs.MNIST_SAMPLE)
# ImageDataBunch.from_folder
# 数据目录,训练集的文件夹名称(默认train)、验证集的文件夹名称（默认valid）、比例参数、随机种子、选取类
# 可以设置size 用于限制数据大小
data_folder = ImageDataBunch.from_folder(path, 'train', 'valid', None, None, None, size=(28, 28))
# ImageDataBunch.from_df
# 数据目录,存储图像文件及其对应标签、相对于path的子路径、比例参数、随机种子、数据文件和标签的列、文件ID是否需要添加后缀
df = pd.read_csv(path / 'labels.csv', header='infer')
data_df = ImageDataBunch.from_df(path, df, None, None, 0.2, None, 0, 1, '')
# ImageDataBunch.from_csv
# 基於from_df實現
# csv文件的名称为labels.csv,可省略csv参数
data_csv = ImageDataBunch.from_csv(path)


# from_name_func 使用文件名提取数据标签
# 从文件名判断类别的函数
def get_labels(file_path):
    return '3' if '/3/' in str(file_path) else '7'


# 参数分别为文件路径、文件列表、提取标签的函数
fnames = [path / file for file in df["name"]]
data_name = ImageDataBunch.from_name_func(path, fnames, label_func=get_labels)
# from_name_re 使用正则表达式提取数据标签
pat = r"/(\d)/\d+\.png$"
data_namere = ImageDataBunch.from_name_re(path, fnames, pat=pat)
# from_list 使用列表提取数据标签
labels_list = list(map(get_labels, fnames))
data_list = ImageDataBunch.from_lists(path, fnames, labels_list)

data_list.show_batch(rows=3, figsize=(4, 4))
plt.show()
