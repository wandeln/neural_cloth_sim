import torch
from get_param import params

# CODO: loss could be computed more efficiently by reusing dx_i / dx_j
# CODO: stiffness / shearing / bending parameters could vary locally...
# CODO: resting angles for shearing and bending could be different from 0 and vary locally

def L_stiffness(x_new,stiffness=params.stiffness,L_0=params.L_0):
	dx_i = x_new[:,:,1:]-x_new[:,:,:-1]
	dx_j = x_new[:,:,:,1:]-x_new[:,:,:,:-1]
	stiffness_i = torch.mean((torch.sqrt(torch.sum(dx_i[:,:3]**2,1))-L_0)**2,[1,2])
	stiffness_j = torch.mean((torch.sqrt(torch.sum(dx_j[:,:3]**2,1))-L_0)**2,[1,2])
	return torch.mean(stiffness*(stiffness_i + stiffness_j))

def L_shearing(x_new,shearing=params.shearing):
	dx_i = x_new[:,:,1:]-x_new[:,:,:-1]
	dx_n_i = torch.nn.functional.normalize(dx_i,dim=1)
	dx_j = x_new[:,:,:,1:]-x_new[:,:,:,:-1]
	dx_n_j = torch.nn.functional.normalize(dx_j,dim=1)
	angle_1 = torch.einsum('abcd,abcd->acd',dx_n_i[:,:,:,:-1],dx_n_j[:,:,:-1])
	angle_2 = torch.einsum('abcd,abcd->acd',dx_n_i[:,:,:,:-1],dx_n_j[:,:,1:])
	angle_3 = torch.einsum('abcd,abcd->acd',dx_n_i[:,:,:,1:],dx_n_j[:,:,:-1])
	angle_4 = torch.einsum('abcd,abcd->acd',dx_n_i[:,:,:,1:],dx_n_j[:,:,1:])
	return torch.mean(shearing*(torch.mean(angle_1**2,[1,2])+torch.mean(angle_2**2,[1,2])+torch.mean(angle_3**2,[1,2])+torch.mean(angle_4**2,[1,2])))

def L_bending(x_new,bending=params.bending):
	dx_i = x_new[:,:,1:]-x_new[:,:,:-1]
	dx_n_i = torch.nn.functional.normalize(dx_i,dim=1)
	dx_j = x_new[:,:,:,1:]-x_new[:,:,:,:-1]
	dx_n_j = torch.nn.functional.normalize(dx_j,dim=1)
	bend_1 = torch.einsum('abcd,abcd->acd',dx_n_i[:,:,1:],dx_n_i[:,:,:-1])
	bend_2 = torch.einsum('abcd,abcd->acd',dx_n_j[:,:,:,1:],dx_n_j[:,:,:,:-1])
	return -torch.mean(bending*(torch.mean(bend_1,[1,2])+torch.mean(bend_2,[1,2])))

def L_a_ext(a,a_ext):
	return -torch.mean(torch.einsum('abcd,abcd->acd',a,a_ext))*params.dt**2

def L_gravity(x_new,M,g=params.g):
	return torch.mean(g*torch.mean(M*x_new[:,2:3],[1,2,3]))

def L_inertia(a,M):
	return 0.5*torch.mean(torch.sum(M*a**2,dim=1))*params.dt**2
