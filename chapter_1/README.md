### 1. MNIST机器学习入门

**1.1.1 简介**

下载MNIST数据集，并打印一些基本信息：
```
python download.py
```

**1.1.2 实验：将MNIST数据集保存为图片**

```
python save_pic.py
```

**1.1.3 图像标签的独热表示**

打印MNIST数据集中图片的标签：
```
python label.py
```

**1.2.1 Softmax 回归**

```
python softmax_regression.py
```

**1.2.2 两层卷积网络分类**
```
python convolutional.py
```

#### 可能出现的错误

下载数据集时可能出现网络问题，可以用下面两种方法中的一种解决：1. 使用合适的代理 2.在MNIST的官方网站上下载文件train-images-idx3-ubyte.gz、train-labels-idx1-ubyte.gz、t10k-images-idx3-ubyte.gz、t10k-labels-idx1-ubyte.gz，并将它们存储在MNIST_data/文件夹中。


#### 拓展阅读

- 本章介绍的MNIST 数据集经常被用来检验机器学习模型的性能，在它的官网（地址：http://yann.lecun.com/exdb/mnist/ ）中，可以找到多达68 种模型在该数据集上的准确率数据，包括相应的论文出处。这些模型包括线性分类器、K 近邻方法、普通的神经网络、卷积神经网络等。
- 本章的两个MNIST 程序实际上来自于TensorFlow 官方的两个新手教程，地址为https://www.tensorflow.org/get_started/mnist/beginners 和 https://www.tensorflow.org/get_started/mnist/pros 。读者可以将本书的内容和官方的教程对照起来进行阅读。这两个新手教程的中文版地址为http://www.tensorfly.cn/tfdoc/tutorials/mnist_beginners.html 和http://www.tensorfly.cn/tfdoc/tutorials/mnist_pros.html。
- 本章简要介绍了TensorFlow 的tf.Tensor 类。tf.Tensor 类是TensorFlow的核心类，常用的占位符（tf.placeholder）、变量（tf.Variable）都可以看作特殊的Tensor。读者可以参阅https://www.tensorflow.org/programmers_guide/tensors 来更深入地学习它的原理。
- 常用tf.Variable 类来存储模型的参数， 读者可以参阅[https://www.tensorflow.org/programmers_guide/variables](https://www.tensorflow.org/programmers_guide/variables) 详细了解它的运行机制， 文档的中文版地址为http://www.tensorfly.cn/tfdoc/how_tos/ variables.html。
- 只有通过会话（Session）才能计算出tf.Tensor 的值。强烈建议读者 在学习完tf.Tensor 和tf.Variable 后，阅读https://www.tensorflow.org/programmers_guide/graphs 中的内容，该文档描述了TensorFlow 中 计算图和会话的基本运行原理，对理解TensorFlow 的底层原理有很 大帮助。
