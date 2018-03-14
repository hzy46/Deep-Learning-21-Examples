### 8. GAN与DCGAN入门

本节的程序来自于项目 https://github.com/carpedm20/DCGAN-tensorflow 。

**8.3.1 生成MNIST图像**

下载MNIST数据集：
```
python download.py mnist
```

训练：
```
python main.py --dataset mnist --input_height=28 --output_height=28 --train
```

生成图像保存在samples文件夹中。

**8.3.2 使用自己的数据集训练**

在数据目录chapter_8_data/中已经准备好了一个动漫人物头像数据集faces.zip。在源代码的data目录中再新建一个anime目录（如果没有data 目录可以自行新建），并将faces.zip 中所有的图像文件解压到anime 目录中。

训练命令：
```
python main.py --input_height 96 --input_width 96 \
  --output_height 48 --output_width 48 \
  --dataset anime --crop -–train \
  --epoch 300 --input_fname_pattern "*.jpg"
```

生成图像保存在samples文件夹中。


#### 拓展阅读

- 本章只讲了GAN 结构和训练方法，在提出GAN 的原始论文 Generative Adversarial Networks 中，还有关于GAN 收敛性的理论证明以及更多实验细节，读者可以阅读来深入理解GAN 的思想。

- 有关DCGAN的更多细节， 可以阅读其论文Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks。

- 除了本章所讲的GAN 和DCGAN 外，还有研究者对原始GAN 的损 失函数做了改进，改进后的模型可以在某些数据集上获得更稳定的 生成效果，相关的论文有：Wasserstein GAN、Least Squares Generative Adversarial Networks。

- 相比一般的神经网络，训练GAN 往往会更加困难。Github 用户 Soumith Chintala 收集了一份训练GAN 的技巧清单：https://github.com/soumith/ganhacks ，在实践中很有帮助。
