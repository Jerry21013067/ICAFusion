#!/bin/bash
# COCO 2017 dataset http://cocodataset.org
# 下载命令: bash data/scripts/get_coco.sh
# 训练命令: python train.py --data coco.yaml
# 默认数据集位置位于 YOLOv5 的同级目录:
#   /parent_folder
#     /coco
#     /yolov5

# 下载/解压标签
d='../' # 解压路径
url=https://github.com/ultralytics/yolov5/releases/download/v1.0/
f='coco2017labels.zip' # or 'coco2017labels-segments.zip', 68 MB
echo 'Downloading' $url$f ' ...'
curl -L $url$f -o $f && unzip -q $f -d $d && rm $f & # download, unzip, remove in background

# 下载/解压图片
d='../coco/images' # 解压路径
url=http://images.cocodataset.org/zips/
f1='train2017.zip' # 19G, 118k images
f2='val2017.zip'   # 1G, 5k images
f3='test2017.zip'  # 7G, 41k images (optional)
for f in $f1 $f2; do
  echo 'Downloading' $url$f '...'
  curl -L $url$f -o $f && unzip -q $f -d $d && rm $f & # download, unzip, remove in background
done
wait # finish background tasks
