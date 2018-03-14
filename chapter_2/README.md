### 2. CIFAR-10与ImageNet图像识别

**2.1.2 下载CIFAR-10 数据**

```
python cifar10_download.py
```

**2.1.3 TensorFlow 的数据读取机制**

实验脚本：
```
python test.py
```

**2.1.4 实验：将CIFAR-10 数据集保存为图片形式**

```
python cifar10_extract.py
```

**2.2.3 训练模型**

```
python cifar10_train.py --train_dir cifar10_train/ --data_dir cifar10_data/
```

**2.2.4 在TensorFlow 中查看训练进度**
```
tensorboard --logdir cifar10_train/
```

**2.2.5 测试模型效果**
```
python cifar10_eval.py --data_dir cifar10_data/ --eval_dir cifar10_eval/ --checkpoint_dir cifar10_train/
```

使用TensorBoard查看性能验证情况：
```
tensorboard --logdir cifar10_eval/ --port 6007
```


#### 拓展阅读

- 关于CIFAR-10 数据集， 读者可以访问它的官方网站https://www.cs.toronto.edu/~kriz/cifar.html 了解更多细节。此外， 网站 http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#43494641522d3130 中收集了在CIFAR-10 数据集上表 现最好的若干模型，包括这些模型对应的论文。
- ImageNet 数据集上的表现较好的几个著名的模型是深度学习的基石， 值得仔细研读。建议先阅读下面几篇论文：ImageNet Classification with Deep Convolutional Neural Networks（AlexNet 的提出）、Very Deep Convolutional Networks for Large-Scale Image Recognition （VGGNet）、Going Deeper with Convolutions（GoogLeNet）、Deep Residual Learning for Image Recognition（ResNet）
- 在第2.1.3 节中，简要介绍了TensorFlow的一种数据读入机制。事实上，目前在TensorFlow 中读入数据大致有三种方法：（1）用占位符（即placeholder）读入，这种方法比较简单；（2）用队列的形式建立文件到Tensor的映射；（3）用Dataset API 读入数据，Dataset API 是TensorFlow 1.3 版本新引入的一种读取数据的机制，可以参考这 篇中文教程：https://zhuanlan.zhihu.com/p/30751039。
