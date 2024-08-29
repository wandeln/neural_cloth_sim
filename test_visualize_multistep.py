import matplotlib.pyplot as plt
#from setups_multistep import Dataset
from setups_multistep_1_channel import Dataset
from cloth_net import get_Net
from Logger import Logger
import torch
import numpy as np
from get_param import params,toCuda,toCpu,get_hyperparam
import time
import os
#from moviepy.editor import *

logger = Logger(get_hyperparam(params),use_csv=False,use_tensorboard=False)


save = False#True#
if save:
	path = f"plots/{get_hyperparam(params)}"#.replace(' ','_').replace(';','_')}"
	os.makedirs(path,exist_ok=True)
	frame = 0
	fps = 30

grad_net = toCuda(get_Net(params))


date_time,index = logger.load_state(grad_net,None,datetime=params.load_date_time,index=params.load_index)
print(f"loaded: {date_time}, {index}")
grad_net.eval()


#params.dt=0.1
plt.figure(1,figsize=(20,20),dpi=200)

with torch.no_grad():#enable_grad():#
	for epoch in range(100):
		dataset = Dataset(params.height,params.width,1,1,params.average_sequence_length,iterations_per_timestep=params.iterations_per_timestep)
		FPS=0
		start_time = time.time()

		for t in range(params.average_sequence_length):
			print(f"t: {t}")
			
			grads, hidden_states = dataset.ask()
			
			update_steps, new_hidden_states = grad_net(grads, hidden_states)
			
			loss = dataset.tell(update_steps, new_hidden_states)
			
			# TODO: visualize, how gradient scaling changes during update steps
			
			if t%10==0:
				index = 0
				
				x = dataset.x[index].cpu()
				a = dataset.a[index].cpu()
				
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
		
		end_time = time.time()
		print(f"dt = {end_time-start_time}s")
		print(f"FPS: {params.average_sequence_length/(end_time-start_time)}")
		

if save:
	#os.system(f"ffmpeg -r {fps} -i {path}/%04d.png -vcodec mpeg4 -y {path}/movie.mp4")
	clip = ImageSequenceClip([f"{path}/{str(f).zfill(4)}.png" for f in range(frame)], fps = 30)
	clip.write_videofile(f"{path}/movie.mp4")
	
# example command:
# python test_visualize_multistep.py --net=Grad_net_scale_inv --iterations_per_timestep=10 --average_sequence_length=99999999
