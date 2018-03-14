### 9. pix2pix模型与自动上色技术

本节的程序来源于项目 https://github.com/affinelayer/pix2pix-tensorflow 。

**9.3.1 执行已有的数据集**

下载Facades数据集：
```
python tools/download-dataset.py facades
```

训练：
```
python pix2pix.py \
  --mode train \
  --output_dir facades_train \
  --max_epochs 200 \
  --input_dir facades/train \
  --which_direction BtoA
```

测试：
```
python pix2pix.py \
  --mode test \
  --output_dir facades_test \
  --input_dir facades/val \
  --checkpoint facades_train
```

结果在facades_test文件夹中。


**9.4.1 为食物图片上色**


在chapter_9_data/中提供的food_resized.zip 文件解压到目录~/datasets/colorlization/下，最终形成的文件
夹结构应该是：

```
~/datasets
  colorlization/
    food_resized/
      train/
      val/
```

训练命令：
```
python pix2pix.py \
--mode train \
--output_dir colorlization_food \
--max_epochs 70 \
--input_dir ~/datasets/colorlization/food_resized/train \
--lab_colorization
```

测试命令：
```
python pix2pix.py \
  --mode test \
  --output_dir colorlization_food_test \
  --input_dir ~/datasets/colorlization/food_resized/val \
  --checkpoint colorlization_food
```

结果在colorlization_food_test文件夹中。

**9.4.2 为动漫图片上色**

将chapter_9_data/中提供的动漫图像数据集anime_reized.zip 解压到~/datasets/colorlization/目录下，形成的文件夹结构为：

```
~/datasets
  colorlization/
    anime_resized/
      train/
      val/
```

训练命令：
```
python pix2pix.py \
  --mode train \
  --output_dir colorlization_anime \
  --max_epochs 5 \
  --input_dir ~/datasets/colorlization/anime_resized/train \
  --lab_colorization
```

测试命令：
```
python pix2pix.py \
  --mode test \
  --output_dir colorlization_anime_test \
  --input_dir ~/datasets/colorlization/anime_resized/val \
  --checkpoint colorlization_anime
```

结果在colorlization_anime_test文件夹中。


#### 拓展阅读

- 本章主要讲了cGAN 和pix2pix 两个模型。读者可以参考它们的原始 论文Conditional Generative Adversarial Nets 和Image-to-Image Translation with Conditional Adversarial Networks 学习更多细节。

- 针对pix2pix 模型，这里有一个在线演示Demo，已经预训练好了多 种模型， 可以在浏览器中直接体验pix2pix 模型的效果： https://affinelayer.com/pixsrv/ 。
