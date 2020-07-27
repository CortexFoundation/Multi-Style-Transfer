#!/bin/bash


CUDA_VISIBLE_DEVICES=7 python3 tools/train_single_gpu.py --config_path ./configs/single_gpu.yaml
