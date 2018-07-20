#! /bin/bash

python eval_image_classifier.py \
  --eval_dir=tianchi/eval_dir \
  --dataset_name=tianchi \
  --dataset_split_name=validation \
  --dataset_dir=/home/fangsh/tianchi/tianchi_dataset/tfrecord \
  --model_name=inception_v3 \
  --checkpoint_path=tianchi/train_dir \
  --batch_size=32

 
