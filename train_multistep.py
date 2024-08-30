import matplotlib.pyplot as plt
#from setups_multistep_1_channel import Dataset
from setups_multistep import Dataset
from setups_transform import DatasetToSingleChannel
from cloth_net import get_Net
#from loss_terms import L_stiffness,L_shearing,L_bending,L_a_ext,L_inertia
from Logger import Logger
import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
from get_param import params,toCuda,toCpu,get_hyperparam,get_load_hyperparam 

torch.manual_seed(0)
torch.set_num_threads(4)
np.random.seed(0)

print(f"Parameters: {vars(params)}")

grad_net = toCuda(get_Net(params))
grad_net.train()

optimizer = AdamW(grad_net.parameters(),lr=params.lr)
scheduler = MultiStepLR(optimizer, milestones=[25,50,75], gamma=0.2)


logger = Logger(get_hyperparam(params),use_csv=False,use_tensorboard=params.log)
if params.load_latest or params.load_date_time is not None or params.load_index is not None:
	load_logger = Logger(get_load_hyperparam(params),use_csv=False,use_tensorboard=False)
	if params.load_optimizer:
		params.load_date_time, params.load_index = load_logger.load_state(grad_net,optimizer,params.load_date_time,params.load_index)
	else:
		params.load_date_time, params.load_index = load_logger.load_state(grad_net,None,params.load_date_time,params.load_index)
	params.load_index=int(params.load_index)
	print(f"loaded: {params.load_date_time}, {params.load_index}")
params.load_index = 0 if params.load_index is None else params.load_index

original_dataset = Dataset(params.height,params.width,params.batch_size,params.dataset_size,params.average_sequence_length,iterations_per_timestep=params.iterations_per_timestep)
#dataset = original_dataset
dataset = DatasetToSingleChannel(original_dataset)



for epoch in range(int(params.load_index),params.n_epochs):
	print(f"epoch: {epoch} / {params.n_epochs}")
	
	for step in range(params.n_batches_per_epoch):
		
		grads, hidden_states = dataset.ask()
		
		update_steps, new_hidden_states = grad_net(grads, hidden_states)
		
		loss = dataset.tell(update_steps, new_hidden_states)
		
		if step%10 == 0:
			logger.log(f"L",loss,epoch*params.n_batches_per_epoch+step)
		print(f"({step} / {params.n_batches_per_epoch}): L: {loss}")
		
		optimizer.zero_grad()
		loss.backward()
		
		# optional: clip gradients
		if params.clip_grad_value is not None:
			torch.nn.utils.clip_grad_value_(cloth_net.parameters(),params.clip_grad_value)
		if params.clip_grad_norm is not None:
			torch.nn.utils.clip_grad_norm_(cloth_net.parameters(),params.clip_grad_norm)
		
		optimizer.step()
		
		if params.plot:
			index = original_dataset.indices[0]
			
			x = original_dataset.x[index].cpu()
			a = original_dataset.a[index].cpu()
			
			plt.clf()
			fig, ax = plt.subplots(1,1,subplot_kw={"projection": "3d"},num=1)
			surf = ax.plot_surface(x[0], x[1], x[2], linewidth=0.1, antialiased=False,edgecolors='k') # cloth surface
			ax.scatter(x[0,[0,-1],0],x[1,[0,-1],0],x[2,[0,-1],0],marker='o',color='g',depthshade=0) # boundarys conditions
			
			
			q_stride, q_l=8,10 # gradients
			"""
			ax.quiver(x[0,::q_stride,::q_stride], x[1,::q_stride,::q_stride], x[2,::q_stride,::q_stride], \
				q_l*grads[0,0,::q_stride,::q_stride], q_l*grads[0,1,::q_stride,::q_stride], q_l*grads[0,2,::q_stride,::q_stride],color='r')
			"""
			ax.quiver(x[0,::q_stride,::q_stride], x[1,::q_stride,::q_stride], x[2,::q_stride,::q_stride], \
				q_l*a[0,::q_stride,::q_stride], q_l*a[1,::q_stride,::q_stride], q_l*a[2,::q_stride,::q_stride],color='g')
			
			
			ax.set_zlim(-100, 1.01)
			ax.set_xlim(-50, 50)
			ax.set_ylim(-50, 50)
			plt.title(f"i: {original_dataset.iterations[index]}; T: {original_dataset.T[index]}")
			
			plt.draw()
			plt.pause(0.01)
	
	# save state
	logger.save_state(grad_net.cpu(),optimizer,epoch+1)
	grad_net = toCuda(grad_net)
# beispiel command:
# python train_multistep.py --log=f --net=Grad_net_tiny --cuda=f --batch_size=1 --dataset_size=1 --n_batches_per_epoch=1000
# python train_multistep.py --log=f --net=Grad_net_scale_inv --cuda=f --batch_size=10 --dataset_size=100 --n_batches_per_epoch=1000 --plot=t --iterations_per_timestep=10
# python train_multistep.py --log=f --net=Grad_net_scale_inv --cuda=f --batch_size=10 --dataset_size=100 --n_batches_per_epoch=1000 --plot=t --iterations_per_timestep=10
# python train_multistep.py --net=Grad_net_scale_inv --batch_size=10 --dataset_size=100 --n_batches_per_epoch=1000 --iterations_per_timestep=10 --cuda=f



