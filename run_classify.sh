#! /bin/bash

python classify_image_inception_v3.py \
--model_path /home/fangsh/Downloads/frozen_graph.ph \
--label_path label.txt \
--image_file /home/fangsh/tianchi/tianchi_dataset/data_megred/test
