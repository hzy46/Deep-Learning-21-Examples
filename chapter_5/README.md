### 5. 深度学习中的目标检测

**5.2.1 安装TensorFlow Object Detection API**

参考5.2.1小节完成相应操作。

**5.2.3 训练新的模型**

先在地址http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar 下载VOC2012数据集并解压。

在项目的object_detection文件夹中新建voc目录，并将解压后的数据集拷贝进来，最终形成的目录为：

```
research/
  object_detection/
    voc/
      VOCdevkit/
        VOC2012/
          JPEGImages/
            2007_000027.jpg
            2007_000032.jpg
            2007_000033.jpg
            2007_000039.jpg
            2007_000042.jpg
            ………………
          Annotations/
            2007_000027.xml
            2007_000032.xml
            2007_000033.xml
            2007_000039.xml
            2007_000042.xml
            ………………
          ………………
```

在object_detection目录中执行如下命令将数据集转换为tfrecord：

```
python create_pascal_tf_record.py --data_dir voc/VOCdevkit/ --year=VOC2012 --set=train --output_path=voc/pascal_train.record
python create_pascal_tf_record.py --data_dir voc/VOCdevkit/ --year=VOC2012 --set=val --output_path=voc/pascal_val.record
```

此外，将pascal_label_map.pbtxt 数据复制到voc 文件夹下：
```
cp data/pascal_label_map.pbtxt voc/
```

下载模型文件http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_11_06_2017.tar.gz 并解压，解压后得到frozen_inference_graph.pb 、graph.pbtxt 、model.ckpt.data-00000-of-00001 、model.ckpt.index、model.ckpt.meta 5 个文件。在voc文件夹中新建一个
pretrained 文件夹，并将这5个文件复制进去。

复制一份config文件：
```
cp samples/configs/faster_rcnn_inception_resnet_v2_atrous_pets.config \
  voc/voc.config
```

并在voc/voc.config中修改7处需要重新配置的地方（详见书本）。

训练模型的命令：
```
python train.py --train_dir voc/train_dir/ --pipeline_config_path voc/voc.config
```

使用TensorBoard：
```
tensorboard --logdir voc/train_dir/
```

**5.2.4 导出模型并预测单张图片**

运行(需要根据voc/train_dir/里实际保存的checkpoint，将1582改为合适的数值)：
```
python export_inference_graph.py \
  --input_type image_tensor \
  --pipeline_config_path voc/voc.config \
  --trained_checkpoint_prefix voc/train_dir/model.ckpt-1582
  --output_directory voc/export/
```

导出的模型是voc/export/frozen_inference_graph.pb 文件。

#### 拓展阅读

- 本章提到的R-CNN、SPPNet、Fast R-CNN、Faster R-CNN 都是基于 区域的深度目标检测方法。可以按顺序阅读以下论文了解更多细节： Rich Feature Hierarchies for Accurate Object Detection and Semantic Segmentation (R-CNN) 、Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition（SPPNet）、Fast R-CNN （Fast R-CNN）、Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks（Faster R-CNN）。

- 限于篇幅，除了本章提到的这些方法外，还有一些有较高参考价值 的深度学习目标检测方法，这里同样推荐一下相关的论文：R-FCN: Object Detection via Region-based Fully Convolutional Networks （R-FCN）、You Only Look Once: Unified, Real-Time Object Detection （YOLO）、SSD: Single Shot MultiBox Detector（SSD）、YOLO9000: Better, Faster, Stronger（YOLO v2 和YOLO9000）等。
