### 11. CycleGAN与非配对图像转换

本节的程序来源于项目https://github.com/vanhuyz/CycleGAN-TensorFlow ，并做了细微修改。

**11.2.1 下载数据集并训练**

下载一个事先准备好的数据集：
```
bash download_dataset.sh apple2orange
```

将图片转换成tfrecords格式：

```
python build_data.py \
  --X_input_dir data/apple2orange/trainA \
  --Y_input_dir data/apple2orange/trainB \
  --X_output_file data/tfrecords/apple.tfrecords \
  --Y_output_file data/tfrecords/orange.tfrecords
```

训练模型：
```
python train.py \
  --X data/tfrecords/apple.tfrecords \
  --Y data/tfrecords/orange.tfrecords \
  --image_size 256
```

打开TensorBoard(需要将--logdir checkpoints/20170715-1622 中的目录替换为自己机器中的对应目录)：
```
tensorboard --logdir checkpoints/20170715-1622
```

导出模型(同样要注意将20170715-1622 替换为自己机器中的对应目录)：
```
python export_graph.py \
  --checkpoint_dir checkpoints/20170715-1622 \
  --XtoY_model apple2orange.pb \
  --YtoX_model orange2apple.pb \
  --image_size 256
```

使用测试集中的图片进行测试：
```
python inference.py \
  --model pretrained/apple2orange.pb \
  --input data/apple2orange/testA/n07740461_1661.jpg \
  --output data/apple2orange/output_sample.jpg \
  --image_size 256
```

转换生成的图片保存在data/apple2orange/output_sample. jpg。

**11.2.2 使用自己的数据进行训练**

在chapter_11_data/中，事先提供了一个数据集man2woman.zip。，解压后共包含两个文件夹：a_resized 和b_resized。将它们放到目录~/datasets/man2woman/下。使用下面的命令将数据集转换为tfrecords：
```
python build_data.py \
  --X_input_dir ~/datasets/man2woman/a_resized/ \
  --Y_input_dir ~/datasets/man2woman/b_resized/ \
  --X_output_file ~/datasets/man2woman/man.tfrecords \
  --Y_output_file ~/datasets/man2woman/woman.tfrecords
```

训练：
```
python train.py \
  --X ~/datasets/man2woman/man.tfrecords \
  --Y ~/datasets/man2woman/woman.tfrecords \
  --image_size 256
```

导出模型和测试图片的指令可参考11.2.1。

#### 拓展阅读

- 本章主要讲了模型CycleGAN ， 读者可以参考论文Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks 了解更多细节

- CycleGAN 不需要成对数据就可以训练，具有较强的通用性，由此产生了大量有创意的应用，例如男女互换（即本章所介绍的）、猫狗互换、利用手绘地图还原古代城市等。可以参考https://zhuanlan.zhihu.com/p/28342644 以及https://junyanz.github.io/CycleGAN/ 了解这些有趣的实验

- CycleGAN 可以将将某一类图片转换成另外一类图片。如果想要把一张图片转换为另外K类图片，就需要训练K个CycleGAN，这是比较麻烦的。对此，一种名为StarGAN 的方法改进了CycleGAN， 可以只用一个模型完成K类图片的转换，有兴趣的读者可以参阅其论文StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation。

- 如果读者还想学习更多和GAN 相关的模型， 可以参考 https://github.com/hindupuravinash/the-gan-zoo 。这里列举了迄今几乎所有的名字中带有“GAN”的模型和相应的论文。
