# Char-RNN-TensorFlow

Multi-language Char RNN in TensorFlow. You can use this code to generate English text, Chinese poetries and lyrics, Japanese text and text in other language.

一个基于最新版本TensorFlow的Char RNN实现。可以实现生成英文、写诗、歌词、小说、生成代码、生成日文等功能。


## Requirements
- Python 2.7.X
- TensorFlow >= 1.2

## Generate English Text

To train:

```
python train.py \
  --input_file data/shakespeare.txt  \
  --name shakespeare \
  --num_steps 50 \
  --num_seqs 32 \
  --learning_rate 0.01 \
  --max_steps 20000
```

To sample:

```
python sample.py \
  --converter_path model/shakespeare/converter.pkl \
  --checkpoint_path model/shakespeare/ \
  --max_length 1000
```

Result:

```
BROTON:
When thou art at to she we stood those to that hath
think they treaching heart to my horse, and as some trousting.

LAUNCE:
The formity so mistalied on his, thou hast she was
to her hears, what we shall be that say a soun man
Would the lord and all a fouls and too, the say,
That we destent and here with my peace.

PALINA:
Why, are the must thou art breath or thy saming,
I have sate it him with too to have me of
I the camples.

```

## Generate Chinese Poetries

To train:

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

To sample:

```
python sample.py \
  --use_embedding \
  --converter_path model/poetry/converter.pkl \
  --checkpoint_path model/poetry/ \
  --max_length 300
```

Result:
```
何人无不见，此地自何如。
一夜山边去，江山一夜归。
山风春草色，秋水夜声深。
何事同相见，应知旧子人。
何当不相见，何处见江边。
一叶生云里，春风出竹堂。
何时有相访，不得在君心。
```

## Generate Chinese Novels

To train (The file "novel.txt" is not included in this repo. You should find one and make sure it is utf-8 encoded!):
```
python train.py \
  --use_embedding True \
  --input_file data/novel.txt \
  --num_steps 80 \
  --name novel \
  --learning_rate 0.005 \
  --num_seqs 32 \
  --num_layers 3 \
  --embedding_size 256 \
  --lstm_size 256 \
  --max_steps 1000000
```

To sample:
```
python sample.py \
  --converter_path model/dpcq/converter.pkl \
  --checkpoint_path  model/novel \
  --use_embedding \
  --max_length 2000 \
  --num_layers 3 \
  --lstm_size 256 \
  --embedding_size 256
```

Result:
```
闻言，萧炎一怔，旋即目光转向一旁的那名灰袍青年，然后目光在那位老者身上扫过，那里，一个巨大的石台上，有着一个巨大的巨坑，一些黑色光柱，正在从中，一道巨大的黑色巨蟒，一股极度恐怖的气息，从天空上暴射而出 ，然后在其中一些一道道目光中，闪电般的出现在了那些人影，在那种灵魂之中，却是有着许些强者的感觉，在他们面前，那一道道身影，却是如同一道黑影一般，在那一道道目光中，在这片天地间，在那巨大的空间中，弥漫而开……

“这是一位斗尊阶别，不过不管你，也不可能会出手，那些家伙，可以为了这里，这里也是能够有着一些异常，而且他，也是不能将其他人给你的灵魂，所以，这些事，我也是不可能将这一个人的强者给吞天蟒，这般一次，我们的实力，便是能够将之击杀……”

“这里的人，也是能够与魂殿强者抗衡。”

萧炎眼眸中也是掠过一抹惊骇，旋即一笑，旋即一声冷喝，身后那些魂殿殿主便是对于萧炎，一道冷喝的身体，在天空之上暴射而出，一股恐怖的劲气，便是从天空倾洒而下。

“嗤！”
```

## Generate Chinese Lyrics

To train:

```
python train.py  \
  --input_file data/jay.txt \
  --num_steps 20 \
  --batch_size 32 \
  --name jay \
  --max_steps 5000 \
  --learning_rate 0.01 \
  --num_layers 3 \
  --use_embedding
```

To sample:

```
python sample.py --converter_path model/jay/converter.pkl \
  --checkpoint_path  model/jay  \
  --max_length 500  \
  --use_embedding \
  --num_layers 3 \
  --start_string 我知道
```

Result:
```
我知道
我的世界 一种解
我一直实现 语不是我
有什么(客) 我只是一口
我想想我不来 你的微笑
我说 你我你的你
只能有我 一个梦的
我说的我的
我不能再想
我的爱的手 一点有美
我们 你的我 你不会再会爱不到
```

## Generate Linux Code

To train:

```
python train.py  \
  --input_file data/linux.txt \
  --num_steps 100 \
  --name linux \
  --learning_rate 0.01 \
  --num_seqs 32 \
  --max_steps 20000
```

To sample:

```
python sample.py \
  --converter_path model/linux/converter.pkl \
  --checkpoint_path  model/linux \
  --max_length 1000 
```

Result:

```
static int test_trace_task(struct rq *rq)
{
        read_user_cur_task(state);
        return trace_seq;
}

static int page_cpus(struct flags *str)
{
        int rc;
        struct rq *do_init;
};

/*
 * Core_trace_periods the time in is is that supsed,
 */
#endif

/*
 * Intendifint to state anded.
 */
int print_init(struct priority *rt)
{       /* Comment sighind if see task so and the sections */
        console(string, &can);
}
```

## Generate Japanese Text

To train:
```
python train.py  \
  --input_file data/jpn.txt \
  --num_steps 20 \
  --batch_size 32 \
  --name jpn \
  --max_steps 10000 \
  --learning_rate 0.01 \
  --use_embedding
```

To sample:
```
python sample.py \
  --converter_path model/jpn/converter.pkl \
  --checkpoint_path model/jpn \
  --max_length 1000 \
  --use_embedding
```

Result:
```
「ああ、それだ、」とお夏は、と夏のその、
「そうだっていると、お夏は、このお夏が、その時、
（あ、」
　と声にはお夏が、これは、この膝の方を引寄って、お夏に、
「まあ。」と、その時のお庇《おも》ながら、
```

## Acknowledgement

Some codes are borrowed from [NELSONZHAO/zhihu/anna_lstm](https://github.com/NELSONZHAO/zhihu/tree/master/anna_lstm)

