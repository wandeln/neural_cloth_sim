import torch
from get_param import params

# CODO: loss could be computed more efficiently by reusing dx_i / dx_j

def L_stiffness(x_new):
	dx_i = x_new[:,:,1:]-x_new[:,:,:-1]
	dx_j = x_new[:,:,:,1:]-x_new[:,:,:,:-1]
	stiffness_i = torch.mean((torch.sqrt(torch.sum(dx_i[:,:3]**2,1))-params.L_0)**2)
	stiffness_j = torch.mean((torch.sqrt(torch.sum(dx_j[:,:3]**2,1))-params.L_0)**2)
	return params.stiffness*(stiffness_i + stiffness_j)

def L_shear(x_new):
	dx_i = x_new[:,:,1:]-x_new[:,:,:-1]
	dx_n_i = torch.nn.functional.normalize(dx_i,dim=1)
	dx_j = x_new[:,:,:,1:]-x_new[:,:,:,:-1]
	dx_n_j = torch.nn.functional.normalize(dx_j,dim=1)
	angle_1 = torch.einsum('abcd,abcd->acd',dx_n_i[:,:,:,:-1],dx_n_j[:,:,:-1])
	angle_2 = torch.einsum('abcd,abcd->acd',dx_n_i[:,:,:,:-1],dx_n_j[:,:,1:])
	angle_3 = torch.einsum('abcd,abcd->acd',dx_n_i[:,:,:,1:],dx_n_j[:,:,:-1])
	angle_4 = torch.einsum('abcd,abcd->acd',dx_n_i[:,:,:,1:],dx_n_j[:,:,1:])
	return params.shearing*(torch.mean(angle_1**2)+torch.mean(angle_2**2)+torch.mean(angle_3**2)+torch.mean(angle_4**2))

def L_bend(x_new):
	dx_i = x_new[:,:,1:]-x_new[:,:,:-1]
	dx_n_i = torch.nn.functional.normalize(dx_i,dim=1)
	dx_j = x_new[:,:,:,1:]-x_new[:,:,:,:-1]
	dx_n_j = torch.nn.functional.normalize(dx_j,dim=1)
	bend_1 = torch.einsum('abcd,abcd->acd',dx_n_i[:,:,1:],dx_n_i[:,:,:-1])
	bend_2 = torch.einsum('abcd,abcd->acd',dx_n_j[:,:,:,1:],dx_n_j[:,:,:,:-1])
	return -params.bending*(torch.mean(bend_1**2)+torch.mean(bend_2**2))

def L_gravity(x_new,M):
	return torch.mean(params.g*M*x_new[:,2:3])

def L_inertia(a,M):
	return 0.5*torch.mean(torch.sum(M*a**2,dim=1))*params.dt**2
