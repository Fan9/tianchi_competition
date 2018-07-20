#! /bin/bash

python export_inference_graph.py \
--alsologtostdeer \
--model_name=inception_v3 \
--output_file=tianchi/inception_v3_inf_graph.pb \
--dataset_name tianchi



