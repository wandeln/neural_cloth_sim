import matplotlib.pyplot as plt
from setups import Dataset
from cloth_net import get_Net
#from loss_terms import L_stiffness,L_shearing,L_bending,L_a_ext,L_inertia
from Logger import Logger
import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
from get_param import params,toCuda,toCpu,get_hyperparam,get_load_hyperparam
from ema_pytorch import EMA

torch.manual_seed(0)
torch.set_num_threads(4)
np.random.seed(0)

print(f"Parameters: {vars(params)}")

cloth_net = toCuda(get_Net(params))
cloth_net.train()

ema_net = EMA(
	cloth_net,
	beta = params.ema_beta,								# exponential moving average factor
	update_after_step = params.ema_update_after_step,	# only after this number of .update() calls will it start updating
	update_every = params.ema_update_every,				# how often to actually update, to save on compute (updates every 10th .update() call)
	power = 3.0/4.0,
	include_online_model = True
)

#optimizer = Adam(cloth_net.parameters(),lr=params.lr)
optimizer = AdamW(cloth_net.parameters(),lr=params.lr)
scheduler = MultiStepLR(optimizer, milestones=[25,50,75], gamma=0.2)

logger = Logger(get_hyperparam(params),use_csv=False,use_tensorboard=params.log)
if params.load_latest or params.load_date_time is not None or params.load_index is not None:
	load_logger = Logger(get_load_hyperparam(params),use_csv=False,use_tensorboard=False)
	if params.load_optimizer:
		params.load_date_time, params.load_index = load_logger.load_state(cloth_net,optimizer,params.load_date_time,params.load_index)
	else:
		params.load_date_time, params.load_index = load_logger.load_state(cloth_net,None,params.load_date_time,params.load_index)
	params.load_index=int(params.load_index)
	print(f"loaded: {params.load_date_time}, {params.load_index}")
params.load_index = 0 if params.load_index is None else params.load_index

dataset = Dataset(params.height,params.width,params.batch_size,params.dataset_size,params.average_sequence_length,stiffness_range=params.stiffness_range,shearing_range=params.shearing_range,bending_range=params.bending_range,a_ext_range=params.g,a_ext_noise_range=params.a_ext_noise_range)
n_vertices = params.height*params.width

for epoch in range(int(params.load_index),params.n_epochs):
	print(f"epoch {epoch} / {params.n_epochs}")
	
	for step in range(params.n_batches_per_epoch):
		
		x_v, stiffnesses, shearings, bendings, a_ext, M, bc = dataset.ask()
		x_v, stiffnesses, shearings, bendings, a_ext, M = toCuda([x_v, stiffnesses, shearings, bendings, a_ext, M])
		#print(f"stiffnesses: {stiffnesses} / {shearings} / {bendings} / {a_ext}")
		
		warmup_iterations = 5#3#10#
		if epoch==0 and step<500:
			warmup_iterations = 10
		if epoch==0 and step<100:
			warmup_iterations = 30
		
		for i in range(warmup_iterations):
			a = cloth_net(x_v, stiffnesses, shearings, bendings, a_ext)
			
			# integrate accelerations
			v_new = x_v[:,3:] + params.dt*a
			x_new = x_v[:,:3] + params.dt*v_new
			
			# apply boundary conditions
			x_new,v_new = bc(x_new,v_new)
			
			# compute loss
			#loss_weights = 1000./stiffnesses
			dx_i = x_new[:,:,1:]-x_new[:,:,:-1]
			dx_n_i = torch.nn.functional.normalize(dx_i,dim=1)
			dx_j = x_new[:,:,:,1:]-x_new[:,:,:,:-1]
			dx_n_j = torch.nn.functional.normalize(dx_j,dim=1)
			
			# stiffness loss
			stiffness_i = torch.mean((torch.sqrt(torch.sum(dx_i[:,:3]**2,1))-params.L_0)**2,[1,2])
			stiffness_j = torch.mean((torch.sqrt(torch.sum(dx_j[:,:3]**2,1))-params.L_0)**2,[1,2])
			L_stiff = stiffnesses*(stiffness_i + stiffness_j)
			
			# simple version of shearing / bending loss
			# shearing loss
			"""
			angle_1 = torch.einsum('abcd,abcd->acd',dx_n_i[:,:,:,:-1],dx_n_j[:,:,:-1])
			angle_2 = torch.einsum('abcd,abcd->acd',dx_n_i[:,:,:,:-1],dx_n_j[:,:,1:])
			angle_3 = torch.einsum('abcd,abcd->acd',dx_n_i[:,:,:,1:],dx_n_j[:,:,:-1])
			angle_4 = torch.einsum('abcd,abcd->acd',dx_n_i[:,:,:,1:],dx_n_j[:,:,1:])
			L_shear = shearings*(torch.mean(angle_1**2,[1,2])+torch.mean(angle_2**2,[1,2])+torch.mean(angle_3**2,[1,2])+torch.mean(angle_4**2,[1,2]))
			
			# bending loss
			bend_1 = torch.einsum('abcd,abcd->acd',dx_n_i[:,:,1:],dx_n_i[:,:,:-1])
			bend_2 = torch.einsum('abcd,abcd->acd',dx_n_j[:,:,:,1:],dx_n_j[:,:,:,:-1])
			L_bend = -bendings*(torch.mean(bend_1,[1,2])+torch.mean(bend_2,[1,2])-2)
			"""
			
			# Davids version of shearing / bending loss
			eps = 1e-7
			# shearing loss
			angle_1 = torch.arccos(torch.einsum('abcd,abcd->acd',dx_n_i[:,:,:,:-1],dx_n_j[:,:,:-1]).clamp(eps-1,1-eps))
			angle_2 = torch.arccos(torch.einsum('abcd,abcd->acd',dx_n_i[:,:,:,:-1],dx_n_j[:,:,1:]).clamp(eps-1,1-eps) )
			angle_3 = torch.arccos(torch.einsum('abcd,abcd->acd',dx_n_i[:,:,:,1:] ,dx_n_j[:,:,:-1]).clamp(eps-1,1-eps))
			angle_4 = torch.arccos(torch.einsum('abcd,abcd->acd',dx_n_i[:,:,:,1:] ,dx_n_j[:,:,1:]).clamp(eps-1,1-eps) )
			L_shear = shearings*(torch.sum((angle_1 - torch.pi/2)**2,[1,2])
								+torch.sum((angle_2 - torch.pi/2)**2,[1,2])
								+torch.sum((angle_3 - torch.pi/2)**2,[1,2])
								+torch.sum((angle_4 - torch.pi/2)**2,[1,2])) / n_vertices

			# bending loss
			bend_1 = torch.arccos(torch.einsum('abcd,abcd->acd',dx_n_i[:,:,1:]  ,dx_n_i[:,:,:-1]).clamp(eps-1,1-eps)  )
			bend_2 = torch.arccos(torch.einsum('abcd,abcd->acd',dx_n_j[:,:,:,1:],dx_n_j[:,:,:,:-1]).clamp(eps-1,1-eps))
			L_bend = bendings*(torch.sum((bend_1 - 0)**2,[1,2])+torch.sum((bend_2 - 0)**2,[1,2])) / n_vertices
			
			# bending loss based on face angles
			# TODO
			
			
			# external forces loss
			L_ext = -torch.mean(torch.einsum('abcd,abcd->acd',a,a_ext),[1,2])*params.dt**2
			
			# inertia loss
			L_inert = 0.5*torch.mean(torch.sum(M*a**2,dim=1),[1,2])*params.dt**2
			
			# total loss
			#loss_weights = (L_stiff + L_shear + L_bend + L_ext + L_inert + 1e-3).detach()
			loss_weights = (L_stiff + L_shear + L_bend + L_inert + 1e-3).detach()
			L = torch.mean(1.0/loss_weights*(L_stiff + L_shear + L_bend + L_ext + L_inert))
			
			# optimize Network
			optimizer.zero_grad()
			L.backward()
			
			# optional: clip gradients
			if params.clip_grad_value is not None:
				torch.nn.utils.clip_grad_value_(cloth_net.parameters(),params.clip_grad_value)
			if params.clip_grad_norm is not None:
				torch.nn.utils.clip_grad_norm_(cloth_net.parameters(),params.clip_grad_norm)
			
			optimizer.step()
		ema_net.update()
		
		# log training metrics
		if step%10 == 0:
			L = toCpu(L).detach().numpy()
			L_stiff = toCpu(torch.mean(L_stiff)).detach().numpy()
			L_shear = toCpu(torch.mean(L_shear)).detach().numpy()
			L_bend = toCpu(torch.mean(L_bend)).detach().numpy()
			L_ext = toCpu(torch.mean(L_ext)).detach().numpy()
			L_inert = toCpu(torch.mean(L_inert)).detach().numpy()
			logger.log(f"L",L,epoch*params.n_batches_per_epoch+step)
			logger.log(f"L_stiff",L_stiff,epoch*params.n_batches_per_epoch+step)
			logger.log(f"L_shear",L_shear,epoch*params.n_batches_per_epoch+step)
			logger.log(f"L_bend",L_bend,epoch*params.n_batches_per_epoch+step)
			logger.log(f"L_ext",L_ext,epoch*params.n_batches_per_epoch+step)
			logger.log(f"L_inert",L_inert,epoch*params.n_batches_per_epoch+step)
			
			print(f"( {step} / {params.n_batches_per_epoch} ) L: {L}; L_stiff: {L_stiff}; L_shear: {L_shear}; L_bend: {L_bend}; L_ext: {L_ext}; L_inert: {L_inert}")
		
		
		# feed new x and v back to dataset
		x_v_new = toCpu(torch.cat([x_new.detach(),v_new.detach()],dim=1)).detach()
		dataset.tell(x_v_new)
		
		# reset environments, where loss gets too high
		for i in range(params.batch_size):
			if loss_weights[i]>1000:
				dataset.reset_env(dataset.indices[i])
		
		
		if params.plot and step%1==0:
			plt.clf()
			fig, ax = plt.subplots(1,1,subplot_kw={"projection": "3d"},num=1)
			surf = ax.plot_surface(x_v_new[0,0], x_v_new[0,1], x_v_new[0,2], linewidth=1, antialiased=False)
			ax.set_zlim(-120, 1.01)
			ax.set_xlim(-64, 64)
			ax.set_ylim(-64, 64)
			plt.draw()
			plt.pause(0.001)
	
	logger.save_state(ema_net,optimizer,epoch+1)
	scheduler.step()
