### 12. RNN基本结构与Char RNN文本生成

**12.5.4 训练模型与生成文字**

训练生成英文的模型：
```
python train.py \
  --input_file data/shakespeare.txt \
  --name shakespeare \
  --num_steps 50 \
  --num_seqs 32 \
  --learning_rate 0.01 \
  --max_steps 20000
```

测试模型：
```
python sample.py \
  --converter_path model/shakespeare/converter.pkl \
  --checkpoint_path model/shakespeare/ \
  --max_length 1000
```

训练写诗模型：
```
python train.py \
  --use_embedding \
  --input_file data/poetry.txt \
  --name poetry \
  --learning_rate 0.005 \
  --num_steps 26 \
  --num_seqs 32 \
  --max_steps 10000
```


测试模型：
```
python sample.py \
  --use_embedding \
  --converter_path model/poetry/converter.pkl \
  --checkpoint_path model/poetry/ \
  --max_length 300
```

训练生成C代码的模型：
```
python train.py \
  --input_file data/linux.txt \
  --num_steps 100 \
  --name linux \
  --learning_rate 0.01 \
  --num_seqs 32 \
  --max_steps 20000
```

测试模型：
```
python sample.py \
  --converter_path model/linux/converter.pkl \
  --checkpoint_path model/linux \
  --max_length 1000
```

#### 拓展阅读

- 如果读者想要深入了解RNN 的结构及其训练方法，建议阅读书籍 Deep Learning（Ian Goodfellow、Yoshua Bengio 和 Aaron Courville 所著）的第10章“Sequence Modeling: Recurrent and Recursive Nets”。 此外，http://karpathy.github.io/2015/05/21/rnn-effectiveness/ 中详细地介绍了RNN 以及Char RNN 的原理，也是很好的阅读材料。

- 如果读者想要深入了解LSTM 的结构， 推荐阅读 http://colah.github.io/posts/2015-08-Understanding-LSTMs/ 。有网友对这篇博文做了翻译，地址为：http://blog.csdn.net/jerr__y/article/ details/58598296。

- 关于TensorFlow 中的RNN 实现，有兴趣的读者可以阅读TensorFlow 源码进行详细了解，地址为：https://github.com/tensorflow/tensorflow/ blob/master/ tensorflow/python/ops/rnn_cell_impl.py 。该源码文件中有BasicRNNCell、BasicLSTMCell、RNNCell、LSTMCell 的实现。
