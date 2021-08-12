#!/user/bin/env python
# coding=utf-8
"""
@project : FastAIStudy
@author  : shanyi
#@file   : Learner_build.py
#@ide    : PyCharm
#@time   : 2021-08-06 17:32:37
# 构建学习器
"""
from fastai.vision import *

path = untar_data(URLs.MNIST_SAMPLE)
data_folder = ImageDataBunch.from_folder(path, size=(28, 28))
# 基于预训练的网络，自动构建分类器
# learn = cnn_learner(data_folder, models.resnet34)
# learn.fit_one_cycle(10)
# print(learn.model)

# 使用自定义的网路结构
model = nn.Sequential(
    nn.Conv2d(3, 6, 5),
    nn.MaxPool2d(2, 2),

    nn.Conv2d(6, 16, 5),
    nn.MaxPool2d(2, 2),

    Flatten(),
    nn.Linear(16 * 4 * 4, 120),
    nn.ReLU(),

    nn.Linear(120, 2)  # 使用的是mnist_sample数据集，只有两类
)

learn = Learner(data_folder, model, metrics=[accuracy])
clbk = [  # callbacks.EarlyStoppingCallback(learn), # 如果给定的指标/验证损失没有改善，则停止训练
    # callbacks.LRFinder(learn),
    callbacks.MixedPrecision(learn),  # 使用半精度浮点数，使GPU处理更加快速
    callbacks.SaveModelCallback(learn)  # 保存验证损失的最佳模型
]
learn.callbacks = clbk
learn.lr_find(start_lr=1e-07, end_lr=10)
learn.fit(3, 1e-2)
learn.unfreeze()
learn.fit_one_cycle(3)
# learn.fit_one_cycle(10, 1e-2)
# 搜索学习速率
# 对网络训练若干个batch，每次迭代时按等比序列更新lr，记录网络输出的损失率
# learn.lr_find()
# learn.recorder.plot_lr()
print(learn.model)

# 对单一数据进行推理
print(learn.predict(learn.data.train_ds[0][0]))
# 对某一数据集的推理
print(learn.pred_batch("train"))
# 对某一数据集计算metrics
print(learn.validate(learn.data.valid_dl))
# 随机抽取图像进行结果的可视化
# learn.show_results()
# 创建结果解析器
interp = learn.interpret()
# 按照损失值排序，返回排序后的损失值及索引
print(interp.top_losses())
# 将损失值最大的k张图像可视化
# interp.plot_top_losses(5)
# 计算混淆矩阵
print(interp.confusion_matrix())
# 绘制混淆矩阵
interp.plot_confusion_matrix()
# 找出最容易混淆的类
print(interp.most_confused())
plt.show()
# 模型的保存和加载
learn.export('/home/shanyi/WorkSpace/FastAIStudy/models/mnist.ckpt')
saved_learn = load_learner('/home/shanyi/WorkSpace/FastAIStudy/models/', file='mnist.ckpt')
