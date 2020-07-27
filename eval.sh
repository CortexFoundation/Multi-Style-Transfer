#!/bin/bash


CUDA_VISIBLE_DEVICES=0 python3 tools/eval.py --config_path ./configs/single_gpu.yaml
