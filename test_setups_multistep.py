from setups_multistep import Dataset
import matplotlib.pyplot as plt
import numpy as np
import torch
from get_param import params

h,w = 64,64

params.dt = 1

dataset = Dataset(h,w,dataset_size=1,batch_size=1,iterations_per_timestep=1000,average_sequence_length=99999999999,dt=params.dt)

index = np.asarray([0])

#lr = 0.005 # learning rate

step_size = 0.0001


old_grads,_ = dataset.ask(index)
old_std = torch.std(old_grads)

update_step = torch.zeros_like(old_grads)

momentum = 0.05
lr = 0.01

for i in range(100000):
	
	grads,_ = dataset.ask(index)
	grads = grads
	std = torch.std(grads)
	
	x = dataset.x[index][0].cpu()
	a = dataset.a[index][0].cpu()
	
	if i%100 == 0:
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
		plt.title(f"i: {dataset.iterations[index]}; T: {dataset.T[index]}")
		
		plt.draw()
		plt.pause(0.01)
	"""
	alignment = torch.mean(old_grads*grads/old_std/std)
	if alignment>0:
		step_size *= 1.05
	else:
		step_size /= 1.05
	print(f"step_size: {step_size}; alignment: {alignment}")
	"""
	
	# simple gradient descent with momentum
	update_step = momentum*(-grads/std) + (1-momentum)*update_step
	dataset.tell(lr*update_step)
	
	old_grads = grads
	old_std = std
	
	


