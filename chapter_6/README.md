### 6. 人脸检测和人脸识别

本节的程序来自于项目https://github.com/davidsandberg/facenet 。

**6.4.1 项目环境设置**

参考6.4.1小节。

**6.4.2 LFW 人脸数据库**

在地址http://vis-www.cs.umass.edu/lfw/lfw.tgz 下载lfw数据集，并解压到~/datasets/中：
```
cd ~/datasets
mkdir -p lfw/raw
tar xvf ~/Downloads/lfw.tgz -C ./lfw/raw --strip-components=1
```

**6.4.3 LFW 数据库上的人脸检测和对齐**

对LFW进行人脸检测和对齐：

```
python src/align/align_dataset_mtcnn.py \
  ~/datasets/lfw/raw \
  ~/datasets/lfw/lfw_mtcnnpy_160 \
  --image_size 160 --margin 32 \
  --random_order
```

在输出目录~/datasets/lfw/lfw_mtcnnpy_160中可以找到检测、对齐后裁剪好的人脸。

**6.4.4 使用已有模型验证LFW 数据库准确率**

在百度网盘的chapter_6_data/目录或者地址https://drive.google.com/file/d/0B5MzpY9kBtDVZ2RpVDYwWmxoSUk 下载解压得到4个模型文件夹，将它们拷贝到~/models/facenet/20170512-110547/中。

之后运行代码：
```
python src/validate_on_lfw.py \
  ~/datasets/lfw/lfw_mtcnnpy_160 \
  ~/models/facenet/20170512-110547/
```

即可验证该模型在已经裁剪好的lfw数据集上的准确率。

**6.4.5 在自己的数据上使用已有模型**

计算人脸两两之间的距离：
```
python src/compare.py \
  ~/models/facenet/20170512-110547/ \
  ./test_imgs/1.jpg ./test_imgs/2.jpg ./test_imgs/3.jpg
```

**6.4.6 重新训练新模型**

以CASIA-WebFace数据集为例，读者需自行申请该数据集，申请地址为http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html 。获得CASIA-WebFace 数据集后，将它解压到~/datasets/casia/raw 目录中。此时文件夹~/datasets/casia/raw/中的数据结构应该类似于：
```
0000045
  001.jpg
  002.jpg
  003.jpg
  ……
0000099
  001.jpg
  002.jpg
  003.jpg
  ……
……
```

先用MTCNN进行检测和对齐：
```
python src/align/align_dataset_mtcnn.py \
  ~/datasets/casia/raw/ \
  ~/datasets/casia/casia_maxpy_mtcnnpy_182 \
  --image_size 182 --margin 44
```

再进行训练：
```
python src/train_softmax.py \
  --logs_base_dir ~/logs/facenet/ \
  --models_base_dir ~/models/facenet/ \
  --data_dir ~/datasets/casia/casia_maxpy_mtcnnpy_182 \
  --image_size 160 \
  --model_def models.inception_resnet_v1 \
  --lfw_dir ~/datasets/lfw/lfw_mtcnnpy_160 \
  --optimizer RMSPROP \
  --learning_rate -1 \
  --max_nrof_epochs 80 \
  --keep_probability 0.8 \
  --random_crop --random_flip \
  --learning_rate_schedule_file
  data/learning_rate_schedule_classifier_casia.txt \
  --weight_decay 5e-5 \
  --center_loss_factor 1e-2 \
  --center_loss_alfa 0.9
```

打开TensorBoard的命令(<开始训练时间>需要进行替换)：
```
tensorboard --logdir ~/logs/facenet/<开始训练时间>/
```

#### 拓展阅读

- MTCNN是常用的人脸检测和人脸对齐模型，读者可以参考论文Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks 了解其细节。

- 训练人脸识别模型通常需要包含大量人脸图片的训练数据集，常用 的人脸数据集有CAISA-WebFace（http://www.cbsr.ia.ac.cn/english/CASIA-WebFace-Database.html ）、VGG-Face（http://www.robots.ox.ac.uk/~vgg/data/vgg_face/ ）、MS-Celeb-1M（https://www.microsoft.com/en-us/research/project/ms-celeb-1m-challenge-recognizing-one-millioncelebrities-real-world/ ）、MegaFace（ http://megaface.cs.washington.edu/ ）。更多数据集可以参考网站：http://www.face-rec.org/databases

- 关于Triplet Loss 的详细介绍，可以参考论文FaceNet: A Unified Embedding for Face Recognition and Clustering，关于Center Loss 的 详细介绍，可以参考论文A Discriminative Feature Learning Approach for Deep Face Recognition。
