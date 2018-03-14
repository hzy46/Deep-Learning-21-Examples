# fast-neural-style-tensorflow

A tensorflow implementation for [Perceptual Losses for Real-Time Style Transfer and Super-Resolution](https://arxiv.org/abs/1603.08155).

This code is based on [Tensorflow-Slim](https://github.com/tensorflow/models/tree/master/slim) and [OlavHN/fast-neural-style](https://github.com/OlavHN/fast-neural-style).

## Samples:

| configuration | style | sample |
| :---: | :----: | :----: |
| [wave.yml](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/conf/wave.yml) |![](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/img/results/style_wave.jpg)|  ![](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/img/results/wave.jpg)  |
| [cubist.yml](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/conf/cubist.yml) |![](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/img/results/style_cubist.jpg)|  ![](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/img/results/cubist.jpg)  |
| [denoised_starry.yml](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/conf/denoised_starry.yml) |![](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/img/results/style_denoised_starry.jpg)|  ![](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/img/results/denoised_starry.jpg)  |
| [mosaic.yml](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/conf/mosaic.yml) |![](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/img/results/style_mosaic.jpg)|  ![](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/img/results/mosaic.jpg)  |
| [scream.yml](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/conf/scream.yml) |![](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/img/results/style_scream.jpg)|  ![](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/img/results/scream.jpg)  |
| [feathers.yml](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/conf/feathers.yml) |![](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/img/results/style_feathers.jpg)|  ![](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/img/results/feathers.jpg)  |
| [udnie.yml](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/conf/udnie.yml) |![](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/img/results/style_udnie.jpg)|  ![](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/img/results/udnie.jpg)  |

## Requirements and Prerequisites:
- Python 2.7.x
- <b>Now support Tensorflow >= 1.0</b>

<b>Attention: This code also supports Tensorflow == 0.11. If it is your version, use the commit 5309a2a (git reset --hard 5309a2a).</b>

And make sure you installed pyyaml:
```
pip install pyyaml
```

## Use Trained Models:

You can download all the 7 trained models from [Baidu Drive](https://pan.baidu.com/s/1i4GTS4d).

To generate a sample from the model "wave.ckpt-done", run:

```
python eval.py --model_file <your path to wave.ckpt-done> --image_file img/test.jpg
```

Then check out generated/res.jpg.

## Train a Model:
To train a model from scratch, you should first download [VGG16 model](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz) from Tensorflow Slim. Extract the file vgg_16.ckpt. Then copy it to the folder pretrained/ :
```
cd <this repo>
mkdir pretrained
cp <your path to vgg_16.ckpt>  pretrained/
```

Then download the [COCO dataset](http://msvocds.blob.core.windows.net/coco2014/train2014.zip). Please unzip it, and you will have a folder named "train2014" with many raw images in it. Then create a symbol link to it:
```
cd <this repo>
ln -s <your path to the folder "train2014"> train2014
```

Train the model of "wave":
```
python train.py -c conf/wave.yml
```

(Optional) Use tensorboard:
```
tensorboard --logdir models/wave/
```

Checkpoints will be written to "models/wave/".

View the [configuration file](https://github.com/hzy46/fast-neural-style-tensorflow/blob/master/conf/wave.yml) for details.
