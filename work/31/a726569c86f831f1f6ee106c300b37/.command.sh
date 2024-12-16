#!/bin/bash -ue
CUDA_VISIBLE_DEVICES=-1 python /home/shiyinghong/nextflow/WorkFlow_PWB/check_and_pred.py         --work_path /home/shiyinghong/nextflow/WorkFlow_PWB
echo Predicting...... > result.txt
