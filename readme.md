# Cloth net

this repository contains code to train and test neural networks to perform cloth simulations.
The loss is based on the SLUG paper.

## Training
To train a net, you can call for example:

> python train.py --clip_grad_value=1 --stiffness=5000 --min_stiffness=100 --shearing=100 --min_shearing=0.01 --bending=100 --min_bending=0.01 --net=SMP_param_a_gated --lr=0.001 --dataset_size=5000 --batch_size=300

Meaning of parameters:  
--clip_grad_value: gradient clipping  
--stiffness: maximum value of stiffness range during training  
--min_stiffness: minimum value of stiffness range during training  
--shearing: maximum value of shearing range during training  
--min_shearing: minimum value of shearing range during training  
--bending: maximum value of bending range during training  
--min_bending: minimum value of bending range during training  
--net: different network architectures (preffered at the moment: SMP_param_a_gated)  
--lr: learning rate of Adam optimizer  
--dataset_size: number of randomized domains during training  
--batch_size: batch size  

More Infos:  
--help  
(or look into get_param.py)

## Visualization:
To visualize results after training:

> python test_visualize.py --net=SMP_param_a_gated --average_sequence_length=1000
