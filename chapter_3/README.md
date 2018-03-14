### 3. 打造自己的图像识别模型

#### 运行方法

**3.2 数据准备**

首先需要将数据转换成tfrecord的形式。在data_prepare文件夹下，运行：
```
python data_convert.py -t pic/ \
  --train-shards 2 \
  --validation-shards 2 \
  --num-threads 2 \
  --dataset-name satellite
```
这样在pic文件夹下就会生成4个tfrecord文件和1个label.txt文件。

**3.3.2 定义新的datasets 文件**

参考3.3.2小节对Slim源码做修改。

**3.3.3 准备训练文件夹**

在slim文件夹下新建一个satellite目录。在这个目录下做下面几件事情：
- 新建一个data 目录，并将第3.2中准备好的5个转换好格式的训练数据复制进去。
- 新建一个空的train_dir目录，用来保存训练过程中的日志和模型。
- 新建一个pretrained目录，在slim的GitHub页面找到Inception V3 模型的下载地址http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz 下载并解压后，会得到一个inception_v3.ckpt 文件，将该文件复制到pretrained 目录下（这个文件在chapter_3_data/文件中也提供了）

**3.3.4 开始训练**

（在slim文件夹下运行）训练Logits层：
```
python train_image_classifier.py \
  --train_dir=satellite/train_dir \
  --dataset_name=satellite \
  --dataset_split_name=train \
  --dataset_dir=satellite/data \
  --model_name=inception_v3 \
  --checkpoint_path=satellite/pretrained/inception_v3.ckpt \
  --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
  --trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
  --max_number_of_steps=100000 \
  --batch_size=32 \
  --learning_rate=0.001 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=300 \
  --save_summaries_secs=2 \
  --log_every_n_steps=10 \
  --optimizer=rmsprop \
  --weight_decay=0.00004
```

训练所有层：
```
python train_image_classifier.py \
  --train_dir=satellite/train_dir \
  --dataset_name=satellite \
  --dataset_split_name=train \
  --dataset_dir=satellite/data \
  --model_name=inception_v3 \
  --checkpoint_path=satellite/pretrained/inception_v3.ckpt \
  --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
  --max_number_of_steps=100000 \
  --batch_size=32 \
  --learning_rate=0.001 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=300 \
  --save_summaries_secs=10 \
  --log_every_n_steps=1 \
  --optimizer=rmsprop \
  --weight_decay=0.00004
```

**3.3.6 验证模型准确率**

在slim文件夹下运行：
```
python eval_image_classifier.py \
  --checkpoint_path=satellite/train_dir \
  --eval_dir=satellite/eval_dir \
  --dataset_name=satellite \
  --dataset_split_name=validation \
  --dataset_dir=satellite/data \
  --model_name=inception_v3
```

**3.3.7 TensorBoard 可视化与超参数选择**

打开TensorBoard：
```
tensorboard --logdir satellite/train_dir
```

**3.3.8 导出模型并对单张图片进行识别**

在slim文件夹下运行：
```
python export_inference_graph.py \
  --alsologtostderr \
  --model_name=inception_v3 \
  --output_file=satellite/inception_v3_inf_graph.pb \
  --dataset_name satellite
```

在chapter_3文件夹下运行（需将5271改成train_dir中保存的实际的模型训练步数）：
```
python freeze_graph.py \
  --input_graph slim/satellite/inception_v3_inf_graph.pb \
  --input_checkpoint slim/satellite/train_dir/model.ckpt-5271 \
  --input_binary true \
  --output_node_names InceptionV3/Predictions/Reshape_1 \
  --output_graph slim/satellite/frozen_graph.pb
```

运行导出模型分类单张图片：
```
python classify_image_inception_v3.py \
  --model_path slim/satellite/frozen_graph.pb \
  --label_path data_prepare/pic/label.txt \
  --image_file test_image.jpg
```


#### 拓展阅读

- TensorFlow Slim 是TensorFlow 中用于定义、训练和验证复杂网络的 高层API。官方已经使用TF-Slim 定义了一些常用的图像识别模型， 如AlexNet、VGGNet、Inception模型、ResNet等。本章介绍的Inception V3 模型也是其中之一， 详细文档请参考： https://github.com/tensorflow/models/tree/master/research/slim。
- 在第3.2节中，将图片数据转换成了TFRecord文件。TFRecord 是 TensorFlow 提供的用于高速读取数据的文件格式。读者可以参考博文（ http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/ ）详细了解如何将数据转换为TFRecord 文件，以及 如何从TFRecord 文件中读取数据。
- Inception V3 是Inception 模型（即GoogLeNet）的改进版，可以参考论文Rethinking the Inception Architecture for Computer Vision 了解 其结构细节。
