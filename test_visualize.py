import matplotlib.pyplot as plt
from setups import Dataset
from cloth_net import get_Net
from Logger import Logger
import torch
from torch.optim import Adam
import numpy as np
from get_param import params,toCuda,toCpu,get_hyperparam
from ema_pytorch import EMA
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

ema_net = EMA(
	cloth_net,
	beta = params.ema_beta,								# exponential moving average factor
	update_after_step = params.ema_update_after_step,	# only after this number of .update() calls will it start updating
	update_every = params.ema_update_every,				# how often to actually update, to save on compute (updates every 10th .update() call)
	power = 3.0/4.0,
	include_online_model = True
)

date_time,index = logger.load_state(ema_net,None,datetime=params.load_date_time,index=params.load_index)
print(f"loaded: {date_time}, {index}")
cloth_net = ema_net#.online_model
cloth_net.eval()

custom_setup = True # False # 

#params.dt=0.1
plt.figure(1,figsize=(20,20),dpi=200)

with torch.no_grad():#enable_grad():#
	for epoch in range(100):
		dataset = Dataset(params.height,params.width,1,1,params.average_sequence_length,stiffness_range=params.stiffness_range,shearing_range=params.shearing_range,bending_range=params.bending_range,a_ext_range=params.g)
		FPS=0
		start_time = time.time()

		if custom_setup:
			x_v, stiffnesses, shearings, bendings, a_ext, M, bc = dataset.ask()
			x_v, stiffnesses, shearings, bendings, a_ext, M = toCuda([x_v, stiffnesses, shearings, bendings, a_ext, M])
			a_ext[:]=0
			a_ext[:,2]=-0.125#1#
		
			#print(f"a_ext: {a_ext[0,:,0,0]}")
		
		for t in range(params.average_sequence_length):
			print(f"t: {t}")
			
			if custom_setup:
				shearings[0:1] = bendings[0:1] = 10#0.1#np.exp(np.cos(t/100)*3-1)#1#
				stiffnesses[0:1] = 10000#np.cos(t/100)*450+550#
			else:
				x_v, stiffnesses, shearings, bendings, a_ext, M, bc = dataset.ask()
				x_v, stiffnesses, shearings, bendings, a_ext, M = toCuda([x_v, stiffnesses, shearings, bendings, a_ext, M])
			
			a = cloth_net(x_v, stiffnesses, shearings, bendings, a_ext)
			
			# integrate accelerations
			v_new = x_v[:,3:] + params.dt*a
			x_new = x_v[:,:3] + params.dt*v_new
			
			# apply boundary conditions
			x_new,v_new = bc(x_new,v_new)
			
			x_v_new = torch.cat([x_new.detach(),v_new.detach()],dim=1)
			if custom_setup:
				x_v = x_v_new
			else:
				dataset.tell(toCpu(x_v_new).detach())
			
			
			if t%5==0:
				plt.clf()
				fig, ax = plt.subplots(1,1,subplot_kw={"projection": "3d"},num=1)
				surf = ax.plot_surface(x_v_new[0,0].cpu(), x_v_new[0,1].cpu(), x_v_new[0,2].cpu(), linewidth=0.1, antialiased=False,edgecolors='k')
				ax.scatter(x_v_new[0,0,[0,-1],0].cpu(),x_v_new[0,1,[0,-1],0].cpu(),x_v_new[0,2,[0,-1],0].cpu(),marker='o',color='g',depthshade=0)
				q_stride, q_l=8,10
				ax.quiver(x_v_new[0,0,::q_stride,::q_stride].cpu(), x_v_new[0,1,::q_stride,::q_stride].cpu(), x_v_new[0,2,::q_stride,::q_stride].cpu(), \
					q_l*a_ext[0,0,::q_stride,::q_stride].cpu(), q_l*a_ext[0,1,::q_stride,::q_stride].cpu(), q_l*a_ext[0,2,::q_stride,::q_stride].cpu(),color='r')
				ax.set_zlim(-120, 1.01)
				ax.set_xlim(-64, 64)
				ax.set_ylim(-32, 96)
				plt.title("stiff: {:.3f}; shear: {:.3f}; bend: {:.3f}".format(stiffnesses[0].cpu().numpy(),shearings[0].cpu().numpy(),bendings[0].cpu().numpy()))
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
	
	
