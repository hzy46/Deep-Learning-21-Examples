### 10. 超分辨率：如何让图像变得更清晰

本节的程序来源于项目 https://github.com/affinelayer/pix2pix-tensorflow 。

**10.1.1 去除错误图片**

在地址http://msvocds.blob.core.windows.net/coco2014/train2014.zip 下载COCO数据集，将所有图片文件放在目录~/datasets/super-resolution/mscoco下。使用chapter_10中的delete_broken_img.py脚本删除一些错误图像：

```
python delete_broken_img.py -p ~/datasets/super-resolution/mscoco/
```

**10.1.2 将图像裁剪到统一大小**

接着将图像缩放到统一大小：
```
python tools/process.py \
  --input_dir ~/datasets/super-resolution/mscoco/ \
  --operation resize \
  --output_dir ~/datasets/super-resolution/mscoco/resized
```

**10.1.3 为代码添加新的操作**

遵循 10.1.3 为代码添加新的blur操作，然后对图像进行模糊处理：
```
python tools/process.py --operation blur \
  --input_dir ~/datasets/super-resolution/mscoco_resized/ \
  --output_dir ~/datasets/super-resolution/mscoco_blur/
```

合并图像：
```
python tools/process.py \
  --input_dir ~/datasets/super-resolution/mscoco_resized/ \
  --b_dir ~/datasets/super-resolution/mscoco_blur/ \
  --operation combine \
  --output_dir ~/datasets/super-resolution/mscoco_combined/
```

划分训练集和测试集：
```
python tools/split.py \
  --dir ~/datasets/super-resolution/mscoco_combined/
```

模型训练：
```
python pix2pix.py --mode train \
  --output_dir super_resolution \
  --max_epochs 20 \
  --input_dir ~/datasets/super-resolution/mscoco_combined/train \
  --which_direction BtoA
```

模型测试：
```
python pix2pix.py --mode test \
--output_dir super_resolution_test \
--input_dir ~/datasets/super-resolution/mscoco_combined/val \
--checkpoint super_resolution/
```

结果在super_resolution_test文件夹中。
