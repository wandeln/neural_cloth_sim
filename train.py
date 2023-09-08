import matplotlib.pyplot as plt
from setups import Dataset
from cloth_net import get_Net
from loss_terms import L_stiffness,L_shear,L_bend,L_gravity,L_inertia
from Logger import Logger
import torch
from torch.optim import Adam
import numpy as np
from get_param import params,toCuda,toCpu,get_hyperparam,get_load_hyperparam

torch.manual_seed(0)
torch.set_num_threads(4)
np.random.seed(0)

print(f"Parameters: {vars(params)}")

cloth_net = toCuda(get_Net(params))
cloth_net.train()

optimizer = Adam(cloth_net.parameters(),lr=params.lr)

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

dataset = Dataset(params.height,params.width,params.batch_size,params.dataset_size,params.average_sequence_length)

for epoch in range(params.load_index,params.n_epochs):
	print(f"epoch {epoch} / {params.n_epochs}")
	
	for step in range(params.n_batches_per_epoch):
		print(f"( {step} / {params.n_batches_per_epoch} )")
		
		x_v, M, bc = dataset.ask()
		x_v, M = toCuda([x_v,M])
		
		warmup_iterations = 5
		if epoch==0 and step<500:
			warmup_iterations = 10
		if epoch==0 and step<100:
			warmup_iterations = 30
		
		for i in range(warmup_iterations):
			a = cloth_net(x_v) # codo: pass M as well
			
			# integrate accelerations
			v_new = x_v[:,3:] + params.dt*a
			x_new = x_v[:,:3] + params.dt*v_new
			
			# apply boundary conditions
			x_new,v_new = bc(x_new,v_new)
			
			# compute loss
			L_stiff = L_stiffness(x_new)
			L_sh = L_shear(x_new)
			L_be = L_bend(x_new)
			L_grav = L_gravity(x_new, M)
			L_inert = L_inertia(a, M)
			L = L_stiff + L_sh + L_be + L_grav + L_inert
			print(f"L: {L.detach().cpu().numpy()}; L_stiff: {L_stiff.detach().cpu().numpy()}; L_shear: {L_sh.detach().cpu().numpy()}; L_bend: {L_be.detach().cpu().numpy()}; L_grav: {L_grav.detach().cpu().numpy()}; L_inert: {L_inert.detach().cpu().numpy()}")
			
			# optimize Network
			optimizer.zero_grad()
			L.backward()
			
			# optional: clip gradients
			if params.clip_grad_value is not None:
				torch.nn.utils.clip_grad_value_(cloth_net.parameters(),params.clip_grad_value)
			if params.clip_grad_norm is not None:
				torch.nn.utils.clip_grad_norm_(cloth_net.parameters(),params.clip_grad_norm)
			
			optimizer.step()
		
		# feed new x and v back to dataset
		x_v_new = toCpu(torch.cat([x_new.detach(),v_new.detach()],dim=1)).detach()
		dataset.tell(x_v_new)
		
		
		if params.plot and step%1==0:
			plt.clf()
			fig, ax = plt.subplots(1,1,subplot_kw={"projection": "3d"},num=1)
			surf = ax.plot_surface(x_v_new[0,0], x_v_new[0,1], x_v_new[0,2], linewidth=1, antialiased=False)
			ax.set_zlim(-120, 1.01)
			ax.set_xlim(-64, 64)
			ax.set_ylim(-64, 64)
			plt.draw()
			plt.pause(0.001)

	logger.save_state(cloth_net,optimizer,epoch+1)
