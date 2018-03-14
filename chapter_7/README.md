### 7. 图像风格迁移

**7.2.1 使用预训练模型**

在chapter_7_data/ 中提供了7 个预训练模型： wave.ckpt-done 、cubist.ckpt-done、denoised_starry.ckpt-done、mosaic.ckpt-done、scream.ckpt-done、feathers.ckpt-done。

以wave.ckpt-done的为例，在chapter_7/中新建一个models 文件
夹， 然后把wave.ckpt-done复制到这个文件夹下，运行命令：
```
python eval.py --model_file models/wave.ckpt-done --image_file img/test.jpg
```

成功风格化的图像会被写到generated/res.jpg。

**7.2.2 训练自己的模型**

准备工作：

- 在地址http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz 下载VGG16模型，将下载到的压缩包解压后会得到一个vgg16.ckpt 文件。在chapter_7/中新建一个文件夹pretrained，并将vgg16.ckpt 复制到pretrained 文件夹中。最后的文件路径是pretrained/vgg16.ckpt。这个vgg16.ckpt文件在chapter_7_data/中也有提供。

- 在地址http://msvocds.blob.core.windows.net/coco2014/train2014.zip 下载COCO数据集。将该数据集解压后会得到一个train2014 文件夹，其中应该含有大量jpg 格式的图片。在chapter_7中建立到这个文件夹的符号链接：
```
ln –s <到train2014 文件夹的路径> train2014
```

训练wave模型：
```
python train.py -c conf/wave.yml
```

打开TensorBoard：
```
tensorboard --logdir models/wave/
```

训练中保存的模型在文件夹models/wave/中。

#### 拓展阅读

- 关于第7.1.1 节中介绍的原始的图像风格迁移算法，可以参考论文A Neural Algorithm of Artistic Style 进一步了解其细节。关于第7.1.2 节 中介绍的快速风格迁移， 可以参考论文Perceptual Losses for Real-Time Style Transfer and Super-Resolution。

- 在训练模型的过程中，用Instance Normalization 代替了常用的Batch Normalization，这可以提高模型生成的图片质量。关于Instance Normalization 的细节，可以参考论文Instance Normalization: The Missing Ingredient for Fast Stylization。

- 尽管快速迁移可以在GPU 下实时生成风格化图片，但是它还有一个 很大的局限性，即需要事先为每一种风格训练单独的模型。论文 Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization 中提出了一种“Arbitrary Style Transfer”算法，可以 为任意风格实时生成风格化图片，读者可以参考该论文了解其实现 细节。
