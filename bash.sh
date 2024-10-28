#!/bin/bash
#loop for different auxiliary anomalous category
for j in $(seq 3 9)
do
    CUDA_VISIBLE_DEVICE=0 python main1.py --normal_digit 0 --gpu 0 --n_epochs 300  --batch_size 400 --auxiliary_digit $j --latent_dim 100  --name mnist --gamma_p 0.2 --gamma_l 0.01 --k 1 --dataset MNIST --dir /mnist/ --gamma_a 0.05 --nk 3
done
exit 0
