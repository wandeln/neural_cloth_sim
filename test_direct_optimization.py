import matplotlib.pyplot as plt
from setups import Dataset
from cloth_net import Cloth_net
from loss_terms import L_stiffness,L_shear,L_gravity,L_inertia
import torch
from torch.optim import Adam
import numpy as np
from get_param import params,toCuda,toCpu

torch.manual_seed(0)
torch.set_num_threads(4)
np.random.seed(0)

print(f"Parameters: {vars(params)}")

params.dataset_size = 1
params.batch_size = 1
params.average_sequence_length = 1000

dataset = Dataset(params.height,params.width,params.batch_size,params.dataset_size,params.average_sequence_length)


for epoch in range(params.n_epochs):
	print(f"epoch {epoch} / {params.n_epochs}")
	
	for step in range(params.n_batches_per_epoch):
		print(f"( {step} / {params.n_batches_per_epoch} )")
		
		x_v, M, bc = dataset.ask()
		x_v, M = toCuda([x_v, M])
		
		a = toCuda(torch.zeros(params.batch_size,3,params.height,params.width)).requires_grad_()
		o = Adam([a],lr=0.02)
		
		for i in range(100):
			
			# integrate accelerations
			v_new = x_v[:,3:] + params.dt*a
			x_new = x_v[:,:3] + params.dt*v_new
			
			# apply boundary conditions
			x_new,v_new = bc(x_new,v_new)
			
			# compute loss
			L = L_stiffness(x_new) + L_gravity(x_new, M) + L_shear(x_new) + L_inertia(a, M)
			L = L/params.height/params.width#/1e6
			print(f"L: {L.detach().cpu().numpy()}")
			
			# optimize Network
			o.zero_grad()
			L.backward()
			o.step()
		
		# feed new x and v back to dataset
		x_v_new = toCpu(torch.cat([x_new.detach(),v_new.detach()],dim=1)).detach()
		dataset.tell(x_v_new)
		
		
		if params.plot and step%1==0:
			plt.clf()
			fig, ax = plt.subplots(1,1,subplot_kw={"projection": "3d"},num=1)
			surf = ax.plot_surface(x_v_new[0,0], x_v_new[0,1], x_v_new[0,2], linewidth=1, antialiased=False)
			ax.set_zlim(-120, 1.01)
			ax.set_xlim(-64, 64)
			ax.set_ylim(-32, 96)
			plt.draw()
			plt.pause(0.001)
 
