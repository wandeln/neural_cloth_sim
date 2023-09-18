import matplotlib.pyplot as plt
from setups import Dataset
from cloth_net import get_Net
from Logger import Logger
import torch
from torch.optim import Adam
import numpy as np
from get_param import params,toCuda,toCpu,get_hyperparam
import time
import os
from moviepy.editor import *

logger = Logger(get_hyperparam(params),use_csv=False,use_tensorboard=False)

save = False#True#
if save:
	path = f"plots/{get_hyperparam(params)}"#.replace(' ','_').replace(';','_')}"
	os.makedirs(path,exist_ok=True)
	frame = 0
	fps = 30

cloth_net = toCuda(get_Net(params))
date_time,index = logger.load_state(cloth_net,None,datetime=params.load_date_time,index=params.load_index)
print(f"loaded: {date_time}, {index}")
cloth_net.eval()


#params.dt=0.1
plt.figure(1,figsize=(10,10))

with torch.no_grad():#enable_grad():#
	for epoch in range(100):
		dataset = Dataset(params.height,params.width,1,1,params.average_sequence_length,stiffness_range=params.stiffness_range,shearing_range=params.shearing_range,bending_range=params.bending_range,grav_range=params.g,mass_range=None)
		FPS=0
		start_time = time.time()

		#x_v, M, bc = dataset.ask()
		#x_v, M = toCuda([x_v,M])
		
		for t in range(params.average_sequence_length):
			print(f"t: {t}")
			
			x_v, stiffnesses, shearings, bendings, gravs, M, bc = dataset.ask()
			x_v, stiffnesses, shearings, bendings, gravs, M = toCuda([x_v, stiffnesses, shearings, bendings, gravs, M])
			a = cloth_net(x_v, stiffnesses, shearings, bendings) # codo: pass M as well
			
			# integrate accelerations
			v_new = x_v[:,3:] + params.dt*a
			x_new = x_v[:,:3] + params.dt*v_new
			
			# apply boundary conditions
			x_new,v_new = bc(x_new,v_new)
			
			x_v_new = torch.cat([x_new.detach(),v_new.detach()],dim=1)
			dataset.tell(toCpu(x_v_new).detach())
			#x_v = x_v_new
			
			
			if t%5==0:
				plt.clf()
				fig, ax = plt.subplots(1,1,subplot_kw={"projection": "3d"},num=1)
				surf = ax.plot_surface(x_v_new[0,0].cpu(), x_v_new[0,1].cpu(), x_v_new[0,2].cpu(), linewidth=0.1, antialiased=False,edgecolors='k')
				ax.scatter(x_v_new[0,0,[0,-1],0].cpu(),x_v_new[0,1,[0,-1],0].cpu(),x_v_new[0,2,[0,-1],0].cpu(),marker='o',color='r',depthshade=0)
				ax.set_zlim(-120, 1.01)
				ax.set_xlim(-64, 64)
				ax.set_ylim(-32, 96)
				plt.title(f"stiff: {stiffnesses[0].cpu().numpy().round(3)}; shear: {shearings[0].cpu().numpy().round(3)}; bend: {bendings[0].cpu().numpy().round(3)}")
				plt.draw()
				plt.pause(0.001)
				if save:
					plt.savefig(f"{path}/{str(frame).zfill(4)}.png", dpi="figure")
					frame += 1
			
		
		end_time = time.time()
		print(f"dt = {end_time-start_time}s")
		print(f"FPS: {params.average_sequence_length/(end_time-start_time)}")
			
if save:
	#os.system(f"ffmpeg -r {fps} -i {path}/%04d.png -vcodec mpeg4 -y {path}/movie.mp4")
	clip = ImageSequenceClip([f"{path}/{str(f).zfill(4)}.png" for f in range(frame)], fps = 30)
	clip.write_videofile(f"{path}/movie.mp4")
	
	
