### 4. Deep Dream

本节的代码参考了TensorFlow 源码中的示例程序[https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/deepdream](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/tutorials/deepdream)，并做了适当修改。

**4.2.1 导入Inception 模型**

在chapter_4_data/中或者网址https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip 下载解压得到模型文件tensorflow_inception_graph.pb，将该文件拷贝到当前文件夹中（即chapter_4/中）。

使用下面的命令加载模型并打印一些基础信息：
```
python load_inception.py
```

**4.2.2 生成原始的Deep Dream 图像**

```
python gen_naive.py
```

**4.2.3 生成更大尺寸的Deep Dream 图像**
```
python gen_multiscale.py
```

**4.2.4 生成更高质量的Deep Dream 图像**
```
python gen_lapnorm.py
```

**4.2.5 最终的Deep Dream 模型**
```
python gen_deepdream.py
```
