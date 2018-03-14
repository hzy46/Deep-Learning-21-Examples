### 17. 看图说话：将图像转换为文字

**17.2.2 环境准备**

机器中没有Bazel的需要安装Bazel，这里以Ubuntu系统为例，其他系统可以参考其官方网站https://docs.bazel.build/versions/master/install.html 进行安装。

在Ubuntu 系统上安装Bazel，首先要添加Bazel 对应的源：

```
echo "deb [arch=amd64] http://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
curl https://bazel.build/bazel-release.pub.gpg | sudo apt-key add -
```

apt-get安装：
```
sudo apt-get update && sudo apt-get install bazel
```

此外还需要安装nltk：
```
pip install nltk
```

**17.2.3 编译和数据准备**

编译源码：
```
bazel build //im2txt:download_and_preprocess_mscoco
bazel build -c opt //im2txt/...
bazel build -c opt //im2txt:run_inference
```

下载训练数据(请保证网络畅通，并确保至少有150GB 的硬盘空间可
以使用)：
```
bazel-bin/im2txt/download_and_preprocess_mscoco "data/mscoco"
```

下载http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz ，解压后得到inception_v3.ckpt。在data目录下新建一个pretrained目录，并将inception_v3.ckpt复制进去。

最后，在data 目录下新建model 文件夹。并在该目录下新建train 和eval 两个文件夹，这两个文件夹分别用来保存训练时的模型、日志和验证时的日志。最终，文件夹结构应该是：
```
im2txt/
  data/
    mscoco/
    pretrained/
      inception_v3.ckpt
    model/
      train/
      eval/
```

**17.2.4 训练和验证**

训练：
```
bazel-bin/im2txt/train \
  --input_file_pattern="data/mscoco/train-?????-of-00256" \
  --inception_checkpoint_file="data/pretrained/inception_v3.ckpt" \
  --train_dir="data/model/train" \
  --train_inception=false \
  --number_of_steps=1000000
```

打开TensorBoard：
```
tensorboard –logdir data/model/train
```

验证困惑度指标：
```
bazel-bin/im2txt/evaluate \
  --input_file_pattern="data/mscoco/val-?????-of-00004" \
  --checkpoint_dir="data/model/train" \
  --eval_dir="data/model/eval"
```

打开TensorBoard 观察验证数据集上困惑度的变化：
```
tensorboard --logdir data/model/eval
```


**17.2.5 测试单张图片**

```
bazel-bin/im2txt/run_inference \
  --checkpoint_path=data/model/train \
  --vocab_file=data/mscoco/word_counts.txt \
  --input_files=data/test.jpg
```

#### 拓展阅读

- Image Caption 是一项仍在不断发展的新技术，除了本章提到的论文 Show and Tell: A Neural Image Caption Generator、Neural machine translation by jointly learning to align and translate、What Value Do Explicit High Level Concepts Have in Vision to Language Problems? 外，还可阅读Mind’s Eye: A Recurrent Visual Representation for Image Caption Generation、From Captions to Visual Concepts and Back 等论 文，了解其更多发展细节。
