#!/bin/bash

python make_hdf5.py --dataset ImageNet --resolution 128 --batch_size 256 --data_root data
python calculate_inception_moments.py --dataset ImageNet --resolution 128 --data_root data