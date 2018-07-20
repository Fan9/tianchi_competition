#! /bin/bash

python freeze_graph.py --input_graph slim/tianchi/inception_v3_inf_graph.pb --input_checkpoint /home/fangsh/tianchi/tianchi_competition/tianchi_competition/slim/tianchi/train_dir/model.ckpt-48362 --input_binary true --output_node_names InceptionV3/Predictions/Reshape_1 --output_graph /home/fangsh/Downloads/frozen_graph.ph

