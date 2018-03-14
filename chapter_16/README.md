### 16. 神经网络机器翻译技术

**16.3.1 示例：将越南语翻译为英语**

下载平行语料库：
```
nmt/scripts/download_iwslt15.sh /tmp/nmt_data
```

训练：
```
mkdir /tmp/nmt_model
python -m nmt.nmt \
  --src=vi --tgt=en \
  --vocab_prefix=/tmp/nmt_data/vocab \
  --train_prefix=/tmp/nmt_data/train \
  --dev_prefix=/tmp/nmt_data/tst2012 \
  --test_prefix=/tmp/nmt_data/tst2013 \
  --out_dir=/tmp/nmt_model \
  --num_train_steps=12000 \
  --steps_per_stats=100 \
  --num_layers=2 \
  --num_units=128 \
  --dropout=0.2 \
  --metrics=bleu
```

测试时，创建一个/tmp/my_infer_file.vi 文件， 并将/tmp/nmt_data/tst2013.vi 中的越南语句子复制一些到/tmp/my_infer_file.vi 里，接着使用下面的命令生成其英语翻译：
```
python -m nmt.nmt \
--out_dir=/tmp/nmt_model \
--inference_input_file=/tmp/my_infer_file.vi \
--inference_output_file=/tmp/nmt_model/output_infer
```

翻译之后的结果在/tmp/nmt_model/output_infer。

训练一个带有注意力机制的模型：
```
mkdir /tmp/nmt_attention_model
python -m nmt.nmt \
  --attention=scaled_luong \
  --src=vi --tgt=en \
  --vocab_prefix=/tmp/nmt_data/vocab \
  --train_prefix=/tmp/nmt_data/train \
  --dev_prefix=/tmp/nmt_data/tst2012 \
  --test_prefix=/tmp/nmt_data/tst2013 \
  --out_dir=/tmp/nmt_attention_model \
  --num_train_steps=12000 \
  --steps_per_stats=100 \
  --num_layers=2 \
  --num_units=128 \
  --dropout=0.2 \
  --metrics=bleu
```

测试模型：
```
python -m nmt.nmt \
--out_dir=/tmp/nmt_attention_model \
--inference_input_file=/tmp/my_infer_file.vi \
--inference_output_file=/tmp/nmt_attention_model/output_infer
```

生成的翻译会被保存在/tmp/nmt_attention_model/output_infer 文件中。


**16.3.2 构建中英翻译引擎**

在chapter_16_data 中提供了一份整理好的中英平行语料数据，共分为train.en、train.zh、dev.en、dev.zh、test.en、test.zh。将它们复制到/tmp/nmt_zh/中。

训练模型：
```
mkdir -p /tmp/nmt_model_zh
python -m nmt.nmt \
  --src=en --tgt=zh \
  --attention=scaled_luong \
  --vocab_prefix=/tmp/nmt_zh/vocab \
  --train_prefix=/tmp/nmt_zh/train \
  --dev_prefix=/tmp/nmt_zh/dev \
  --test_prefix=/tmp/nmt_zh/test \
  --out_dir=/tmp/nmt_model_zh \
  --step_per_stats 100 \
  --num_train_steps 200000 \
  --num_layers 3 \
  --num_units 256 \
  --dropout 0.2 \
  --metrics bleu
```

在/tmp/my_infer_file.en中保存一些需要翻译的英文句子（格式为：每一行一个英文句子，句子中每个英文单词，包括标点符号之间都要有
空格分隔。可以从test.en中复制）。使用下面的命令进行测试：

```
python -m nmt.nmt \
  --out_dir=/tmp/nmt_model_zh \
  --inference_input_file=/tmp/my_infer_file.en \
  --inference_output_file=/tmp/output_infer
```

翻译后的结果被保存在/tmp/output_infer文件中。

#### 拓展阅读

- 关于用Encoder-Decoder 结构做机器翻译任务的更多细节，可以参考 原始论文Learning Phrase Representations using RNN Encoder– Decoder for Statistical Machine Translation。

- 关于注意力机制的更多细节，可以参考原始论文Neural Machine Translation by Jointly Learning to Align and Translate。此外还有改进 版的注意力机制：Effective Approaches to Attention-based Neural Machine Translation。
