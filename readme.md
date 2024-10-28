# Code



You can use following command:

> python main1.py --normal_digit 0 --gpu 0 --n_epochs 300  --batch_size 400 --auxiliary_digit $j --latent_dim 100  --name mnist --gamma_p 0.2 --gamma_l 0.01 --k 1 --dataset MNIST --dir /mnist/ --gamma_a 0.05 --nk 3



The meaning of partial arguments in command are shown as below:

> --gamma_p   the parameter controlling the ratio of contamination

> --gamma_l    the parameter controlling the ratio of observed anomalous data

> --gamma_a   the parameter controlling the ratio of observed normal data

> --k                the parameter controlling the number of types of observed anomalous

> --nk              the parameter controlling the number of types of normal data



You can also modify and run `bash.sh` to run experiment automatically.

> option choices: dataset =[CIFAR,F-MNIST,MNIST], latent_dim = [128 (CIFAR), 100 (F-MNIST,MNIST)]

### Reference

We build our code based on [AA-BiGAN](https://github.com/tbw162/AA-BiGAN)

