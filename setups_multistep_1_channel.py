import torch
import numpy as np
from get_param import params,toCuda,toCpu
from itertools import chain

# deprecated!!!!!!!!!!!!!
# use instead setups_multistep.Dataset in combination with setups_transform.DatasetToSingleChannel

eps = 1e-7
device = 'cuda' if params.cuda else 'cpu'

"""
ask-tell interface:
ask(): ask for batch of gradients of velocities wrt certain loss function
tell(): tell update step for velocities (positions are updated internally) => return loss to update NN parameters
"""
#Attention: x/y are swapped (x-dimension=1; y-dimension=0)

# CODO: spatially varying stiffness / shearing / bending parameters

def concat(list_of_lists):
	return [inner for outer in list_of_lists for inner in outer]

def rotation_matrix(dyaw=0.0,dpitch=0.0,droll=0.0,device=None):
	"""
	:return: matrix to rotate by dpitch/dyaw/droll
	"""
	def tensor(x):
		if type(x) is not torch.Tensor:
			return torch.tensor(x,device=device)
		return x
	dpitch,dyaw,droll = tensor(dpitch),tensor(dyaw),tensor(droll)
	trafo_pitch_matrix = torch.eye(3,device=device)
	if dpitch != 0 or dpitch.requires_grad:
		trafo_pitch_matrix[1,1] = torch.cos(dpitch)
		trafo_pitch_matrix[1,2] = -torch.sin(dpitch)
		trafo_pitch_matrix[2,1] = torch.sin(dpitch)
		trafo_pitch_matrix[2,2] = torch.cos(dpitch)
	trafo_yaw_matrix = torch.eye(3,device=device)
	if dyaw != 0 or dyaw.requires_grad:
		trafo_yaw_matrix[0,0] = torch.cos(dyaw)
		trafo_yaw_matrix[0,2] = torch.sin(dyaw)
		trafo_yaw_matrix[2,0] = -torch.sin(dyaw)
		trafo_yaw_matrix[2,2] = torch.cos(dyaw)
	trafo_roll_matrix = torch.eye(3,device=device)
	if droll != 0 or droll.requires_grad:
		trafo_roll_matrix[0,0] = torch.cos(droll)
		trafo_roll_matrix[0,1] = -torch.sin(droll)
		trafo_roll_matrix[1,0] = torch.sin(droll)
		trafo_roll_matrix[1,1] = torch.cos(droll)
	trafo_matrix = torch.matmul(torch.matmul(trafo_yaw_matrix,trafo_pitch_matrix),trafo_roll_matrix)
	return trafo_matrix

n_vertices = params.height*params.width
L_0 = params.L_0

def loss(x_old,v_old,acc,force,bc_masks,bc_positions,M,stiffnesses,shearings,bendings,dt):
	"""
	:return:
		:loss: loss values for samples in batch (shape: batch_size)
		:E_int: internal energies for samples in batch (shape: batch_size)
	"""
	
	# integrate velocity and positions
	v_new = v_old + dt*acc
	x_new = x_old + dt*v_new
	
	# apply boundary conditions
	x_new = bc_masks * bc_positions + (1-bc_masks) * x_new
	v_new = (1-bc_masks) * v_new
	
	# compute energy terms
	dx_i = x_new[:,:,1:]-x_new[:,:,:-1]
	dx_n_i = torch.nn.functional.normalize(dx_i,dim=1)
	dx_j = x_new[:,:,:,1:]-x_new[:,:,:,:-1]
	dx_n_j = torch.nn.functional.normalize(dx_j,dim=1)

	# stiffness energy
	stiffness_i = torch.mean((torch.sqrt(torch.sum(dx_i[:,:3]**2,1))-L_0)**2,[1,2])
	stiffness_j = torch.mean((torch.sqrt(torch.sum(dx_j[:,:3]**2,1))-L_0)**2,[1,2])
	E_stiff = stiffnesses*(stiffness_i + stiffness_j)
	
	# Davids version of shearing energy
	angle_1 = torch.arccos(torch.einsum('abcd,abcd->acd',dx_n_i[:,:,:,:-1],dx_n_j[:,:,:-1]).clamp(eps-1,1-eps))
	angle_2 = torch.arccos(torch.einsum('abcd,abcd->acd',dx_n_i[:,:,:,:-1],dx_n_j[:,:,1:]).clamp(eps-1,1-eps) )
	angle_3 = torch.arccos(torch.einsum('abcd,abcd->acd',dx_n_i[:,:,:,1:] ,dx_n_j[:,:,:-1]).clamp(eps-1,1-eps))
	angle_4 = torch.arccos(torch.einsum('abcd,abcd->acd',dx_n_i[:,:,:,1:] ,dx_n_j[:,:,1:]).clamp(eps-1,1-eps) )
	E_shear = shearings*(torch.sum((angle_1 - torch.pi/2)**2,[1,2])
						+torch.sum((angle_2 - torch.pi/2)**2,[1,2])
						+torch.sum((angle_3 - torch.pi/2)**2,[1,2])
						+torch.sum((angle_4 - torch.pi/2)**2,[1,2])) / n_vertices
	
	# Davids version of bending energy
	bend_1 = torch.arccos(torch.einsum('abcd,abcd->acd',dx_n_i[:,:,1:]  ,dx_n_i[:,:,:-1]).clamp(eps-1,1-eps)  )
	bend_2 = torch.arccos(torch.einsum('abcd,abcd->acd',dx_n_j[:,:,:,1:],dx_n_j[:,:,:,:-1]).clamp(eps-1,1-eps))
	E_bend = bendings*(torch.sum((bend_1 - 0)**2,[1,2])+torch.sum((bend_2 - 0)**2,[1,2])) / n_vertices
	
	# compute inertia term
	L_inert = 0.5*torch.mean(torch.sum(M*acc**2,dim=1),[1,2])*dt**2
	
	# compute external forces term
	L_ext = -torch.mean(torch.einsum('abcd,abcd->acd',acc,force*M),[1,2])*dt**2
	
	# total loss
	#loss_weights = (E_stiff + E_shear + E_bend + L_inert + 1e-3).detach()
	#L = torch.mean(1.0/loss_weights*(E_stiff + E_shear + E_bend + L_ext + L_inert))
	
	E_int = E_stiff + E_shear + E_bend
	L = E_int + L_ext + L_inert
	
	return L, E_int

def log_range_params(range_params,default_param=1):# useful to sample parameters from "exponential distribution"
	range_params = default_param if range_params is None else range_params
	range_params = [range_params,range_params] if type(range_params) is not list else range_params
	range_params = np.log(range_params)
	return range_params[0],range_params[1]-range_params[0]

def range_params(r_params,default_param=1):# useful to sample parameters from "exponential distribution"
	r_params = default_param if r_params is None else r_params
	r_params = [r_params,r_params] if type(r_params) is not list else r_params
	return r_params[0],r_params[1]-r_params[0]

class Dataset:
	def __init__(self,h,w,batch_size=100,dataset_size=1000,average_sequence_length=5000,interactive=False,dt=1,L_0=1,stiffness_range=None,shearing_range=None,bending_range=None,a_ext_range=None,a_ext_noise_range=0,iterations_per_timestep=5):
		# dataset parameters
		self.h,self.w = h,w
		self.batch_size = batch_size
		self.dataset_size = dataset_size
		self.average_sequence_length = average_sequence_length
		
		# grid utility
		x_space = torch.linspace(0,L_0*(w-1),w)
		y_space = torch.linspace(-L_0*(h-1)/2,L_0*(h-1)/2,h)
		y_grid,x_grid = torch.meshgrid(y_space,x_space,indexing="ij")
		self.y_mesh,self.x_mesh = toCuda(torch.meshgrid([torch.arange(0,self.h),torch.arange(0,self.w)]))
		
		# cloth state values
		self.x_0 = torch.cat([x_grid.unsqueeze(0),y_grid.unsqueeze(0),torch.zeros(1,h,w)],dim=0)
		self.v_0 = toCuda(torch.zeros(3,h,w))
		self.x = toCuda(torch.zeros(self.dataset_size,3,self.h,self.w)) # positions and velocities
		self.v = toCuda(torch.zeros(self.dataset_size,3,self.h,self.w)) # positions and velocities
		self.a = toCuda(torch.zeros(dataset_size,3,h,w)) # accelerations (start at zero and get updated for every iteration until next timestep)
		self.T = toCuda(torch.zeros(self.dataset_size,1)) # timestep
		self.iterations = toCuda(torch.zeros(dataset_size)) # iterations for the individual training pool samples
		self.iterations_per_timestep = iterations_per_timestep # number of iterations per timestep
		
		#self.hidden_states = [None for _ in range(dataset_size)]
		self.hidden_states = [[None,None,None] for _ in range(dataset_size)]
		
		# simulation / cloth parameters
		self.dt = dt
		self.L_0 = L_0 # resting length of cloth grid edges
		
		self.M = torch.ones(1,1,h,w,device=device) # Mass matrix
		self.M[:,:,0] = self.M[:,:,-1] = self.M[:,:,:,0] = self.M[:,:,:,-1] = 0.5
		self.M[:,:,0,0] = self.M[:,:,0,-1] = self.M[:,:,-1,0] = self.M[:,:,-1,-1] = 0.25
		
		self.stiffness_range = log_range_params(stiffness_range)
		self.shearing_range = log_range_params(shearing_range)
		self.bending_range = log_range_params(bending_range)
		
		self.g_vect = toCuda(torch.tensor([0,0,-1.])).unsqueeze(0).repeat(self.dataset_size,1).unsqueeze(2).unsqueeze(3) # gravity vector. CODO: radnom directions / strengths of gravity
		self.a_ext_range = range_params(a_ext_range)
		self.a_exts = toCuda(torch.ones(self.dataset_size,3,self.h,self.w))*self.g_vect# external forces
		self.a_exts_damping = 0.999
		self.da_exts_dt = toCuda(torch.zeros(self.dataset_size,3,self.h,self.w))# derivatives of external forces
		self.da_exts_dt_damping = 0.95
		self.a_ext_noise_range = a_ext_noise_range
		
		
		self.rot_speed = toCuda(torch.zeros(self.dataset_size,3,3)) # delta rotation matrix that is recurrently multiplied onto rotations
		self.translation_freq = toCuda(torch.zeros(self.dataset_size,3)) # delta rotation matrix that is recurrently multiplied onto rotations
		self.pinch_freq = toCuda(torch.zeros(self.dataset_size,1)) # delta rotation matrix that is recurrently multiplied onto rotations
		self.translation_amp = toCuda(torch.zeros(self.dataset_size,3)) # delta rotation matrix that is recurrently multiplied onto rotations
		self.rotations = toCuda(torch.zeros(self.dataset_size,3,3))
		self.bc_positions = toCuda(torch.zeros(self.dataset_size,3,self.h,self.w)) # left / right point & (x,y,z) coordinates
		self.bc_masks = toCuda(torch.zeros(self.dataset_size,1,self.h,self.w))
		self.stiffnesses = toCuda(torch.zeros(self.dataset_size))
		self.shearings = toCuda(torch.zeros(self.dataset_size))
		self.bendings = toCuda(torch.zeros(self.dataset_size))
		#self.a_exts = torch.ones(self.dataset_size,3,self.h,self.w)*torch.tensor([0,0,-1]).unsqueeze(0).unsqueeze(2).unsqueeze(3) # init with gravity. CODO: radnom directions / strengths of gravity
		
		
		for i in range(self.dataset_size):
			self.reset_env(i)
		
		self.interactive = interactive
		self.step = 0 # number of tell()-calls
		self.reset_i = 0
		
	def reset_env(self,index):
		self.stiffnesses[index] = 1000#torch.exp(self.stiffness_range[0]+torch.rand(1)*self.stiffness_range[1])
		self.shearings[index] = 10#0#torch.exp(self.shearing_range[0]+torch.rand(1)*self.shearing_range[1])
		self.bendings[index] = 0#10#torch.exp(self.bending_range[0]+torch.rand(1)*self.bending_range[1])
		#self.a_exts[index] = torch.exp(self.a_ext_range[0]+torch.rand(1)*self.a_ext_range[1]) # TODO: init with gravity
		g_scale = self.a_ext_range[0]+torch.rand(1,device=device)*self.a_ext_range[1]
		#self.hidden_states[index] = None
		self.hidden_states[index] = [None,None,None] # NEW: single channel dataset
		
		# reset state
		self.x[index] = self.x_0.clone()
		self.v[index] = self.v_0.clone()
		self.a[index] = 0
		
		self.T[index] = 0
		yaw = 0#(torch.rand(1)-0.5)*2*2*3.14#0#
		pitch = 0#(torch.rand(1)-0.5)*2*2*3.14#0#
		roll = 0#(torch.rand(1)-0.5)*2*2*3.14#0#
		dyaw = 0#(torch.rand(1)-0.5)*2*2*3.14*0.01
		dpitch = 0#(torch.rand(1)-0.5)*2*2*3.14*0.01
		droll = 0#(torch.rand(1)-0.5)*2*2*3.14*0.01 # keep only roll for rotation
		self.rot_speed[index] = rotation_matrix(dyaw,dpitch,droll,device=device)
		self.rotations[index] = rotation_matrix(yaw,pitch,roll,device=device)
		self.translation_freq[index] = 0#(torch.rand(3)-0.5)*2*0.2
		self.translation_amp[index] = 0#(torch.rand(3)-0.5)*2*10
		self.pinch_freq[index] = 0#(torch.rand(1)-0.5)*2*0.2
		self.x[index,:] = torch.einsum("ab,bcd->acd",self.rotations[index],self.x[index,:])
		#print(f"reset {index}")
		
		# boundary conditions
		self.bc_masks[index] = 0
		self.bc_masks[index,:,0,0] = 1
		self.bc_masks[index,:,-1,0] = 1
		self.bc_positions[index] = self.x[index].clone()
		
		# external forces
		self.g_vect[index,:,0,0] = torch.tensor([0,0,-1.0],device=device)*g_scale#torch.einsum("ab,b->a",rotation_matrix((torch.rand(1)-0.5)*2*2*3.14,(torch.rand(1)-0.5)*2*2*3.14,(torch.rand(1)-0.5)*2*2*3.14,device=device),torch.tensor([0,0,-1.0],device=device)*g_scale)
		self.a_exts[index,:,:,:] = self.g_vect[index]
		#print(f"rot mat: {rotation_matrix((torch.rand(1)-0.5)*2*2*3.14,(torch.rand(1)-0.5)*2*2*3.14,(torch.rand(1)-0.5)*2*2*3.14)}")
		#print(f"g_vect: {self.g_vect[index,:,0,0]}")
		self.da_exts_dt[index,:,:,:] = 0
	
	def update_env(self,index):
		# update state
		
		self.v[index] += self.a[index]*self.dt
		self.x[index] += self.v[index]*self.dt
		self.a[index] = 0
		
		# apply boundary conditions
		self.x[index] = self.bc_masks[index] * self.bc_positions[index] + (1-self.bc_masks[index]) * self.x[index]
		self.v[index] = (1-self.bc_masks[index]) * self.v[index]
		
		
		# update external forces and boundary conditions
		
		# TODO update external forces
		
		# update external forces (CODO: clip min/max forces) ...not very efficient (slows down test_cv2_interactive by approx 10%)
		"""
		self.a_exts[self.indices,:,:,:] = self.a_exts_damping*self.a_exts[self.indices,:,:,:]+(1-self.a_exts_damping)*self.g_vect[self.indices]+0.01*self.da_exts_dt[self.indices,:,:,:]
		if torch.rand(1)<0.3:
			gaussian_w = (torch.rand(1)*30)**2
			gaussian = torch.exp(-((self.x_mesh-torch.rand(1,1)*self.w)**2+(self.y_mesh-torch.rand(1,1)*self.h)**2)/gaussian_w).unsqueeze(0).unsqueeze(1)
			gaussian = gaussian*torch.randn(1,3,1,1)
		else:
			gaussian = 0
		self.da_exts_dt[self.indices,:,:,:] = self.da_exts_dt_damping*self.da_exts_dt[self.indices,:,:,:]+0.1*torch.randn(1,3,1,1)+gaussian
		
		# add random noise to a_exts
		a_ext_noise = self.a_ext_noise_range*torch.rand(self.batch_size).unsqueeze(1).unsqueeze(2).unsqueeze(3)*torch.randn(self.batch_size,3,self.h,self.w)
		"""
		
		self.a_exts[self.indices,:,:,:] = self.g_vect[self.indices]
		
		
		# TODO update boundary conditions => include this functionality above...
		"""
		def BoundaryConditions(x,v): # maybe use least squares loss instead! => move this to loss computation
			f = self.rot_speed[self.indices]
			
			x[:,:3,0,0] = self.conditions[self.indices,0,:]*(torch.cos(self.T[self.indices]*self.pinch_freq[self.indices])*0.4+0.6) + torch.sin(self.T[self.indices]*self.translation_freq[self.indices])*self.translation_amp[self.indices]
			x[:,:3,-1,0] = self.conditions[self.indices,1,:]*(torch.cos(self.T[self.indices]*self.pinch_freq[self.indices])*0.4+0.6) + torch.sin(self.T[self.indices]*self.translation_freq[self.indices])*self.translation_amp[self.indices]
			v[:,:,0,0] = v[:,:,-1,0] = 0
			return x,v # CODO: be more careful with BC... (e.g. pass them also to neural net)
		"""
		
		pass
	
	def ask(self,indices=None):
		"""
		:indices: optional indices of samples to draw from training pool (shape: batch_size)
		:return: 
			gradients for accelerations (shape: batch_size x 3 x h x w)
			hidden_states for optimizer
		"""
		self.indices = np.random.choice(self.dataset_size,self.batch_size) if indices is None else indices
		
		with torch.enable_grad():
			# compute gradients wrt accelerations
			asked_grads = torch.zeros(self.batch_size,3,self.h,self.w,device=device,requires_grad=True)
			
			l,_ = loss(self.x[self.indices],
				self.v[self.indices],
				self.a[self.indices] + asked_grads,
				self.a_exts[self.indices],
				self.bc_masks[self.indices],self.bc_positions[self.indices],
				self.M,self.stiffnesses[self.indices],self.shearings[self.indices],self.bendings[self.indices],self.dt)
			
			l = torch.sum(l) # input grads should be independent of batch size => use sum instead of mean
			
			l.backward()
		
		#return asked_grads.grad, [self.hidden_states[i] for i in self.indices]
		return asked_grads.grad.reshape(3*self.batch_size,1,self.h,self.w), concat([self.hidden_states[i] for i in self.indices]) # NEW: single channel dataset
	
	def tell(self,step, hidden_states=None):
		"""
		:step: update step for accelerations for gradients given by ask
		:return: loss to optimize neural update-step-model => TODO: shape sollte batch_size sein, damit loss / gradienten rescaled werden können!
		"""
		#hidden_states = [None for _ in self.indices] if hidden_states is None else hidden_states
		hidden_states = [[None,None,None] for _ in self.indices] if hidden_states is None else hidden_states # NEW: single channel dataset
		step = step.reshape(self.batch_size,3,self.h,self.w) # NEW: single channel dataset
		
		self.iterations[self.indices] = self.iterations[self.indices] + 1
		
		acc = self.a[self.indices] + step
		self.a[self.indices] = acc.detach()
		
		# compute loss => CODO: scaling of loss?
		l, E_int = loss(self.x[self.indices],self.v[self.indices],acc,self.a_exts[self.indices],self.bc_masks[self.indices],self.bc_positions[self.indices],self.M,self.stiffnesses[self.indices],self.shearings[self.indices],self.bendings[self.indices],self.dt)
		
		#print(f"E_int: {E_int}")
		
		# update step if iterations_per_timestep is reached
		for i,index in enumerate(self.indices):
			if E_int[i] > 2000:
				self.reset_env(index)
			else:
				#self.hidden_states[index] = hidden_states[i]
				self.hidden_states[index] = hidden_states[i*3:(i+1)*3] # NEW: single channel dataset
				if self.iterations[index] % self.iterations_per_timestep == 0:
					self.T[index] = self.T[index] + self.dt
					
					self.update_env(index)
		
		# reset environments eventually TODO: check that / reset environment, if E_int becomes too large!
		self.step += 1
		if self.step % (self.average_sequence_length/self.batch_size) == 0:#ca x*batch_size steps until env gets reset
			self.reset_env(int(self.reset_i)) # => this resets also hidden state
			self.reset_i = (self.reset_i+1)%self.dataset_size
			
		return torch.mean(l)
