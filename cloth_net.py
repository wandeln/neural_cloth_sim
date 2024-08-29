import segmentation_models_pytorch as smp 
import torch
from torch import nn
from torch.nn.parameter import Parameter
from get_param import params
from utils import normalize_grads
from unet_parts import *
device = 'cuda' if params.cuda else 'cpu'

def get_Net(params):
	if params.net == "UNet":
		net = Cloth_Unet(params.hidden_size)
	elif params.net == "UNet_param_a":
		net = Cloth_Unet_param_a(params.hidden_size)
	elif params.net == "SMP":
		net = Cloth_net(params.SMP_model_type,params.SMP_encoder_name)
	elif params.net == "SMP_param":
		net = Cloth_net_param(params.SMP_model_type,params.SMP_encoder_name)
	elif params.net == "SMP_param_a":
		net = Cloth_net_param_a(params.SMP_model_type,params.SMP_encoder_name)
	elif params.net == "SMP_param_a_gated":
		net = Cloth_net_param_a_gated(params.SMP_model_type,params.SMP_encoder_name)
	elif params.net == "SMP_param_a_gated2":
		net = Cloth_net_param_a_gated2(params.SMP_model_type,params.SMP_encoder_name)
	elif params.net == "SMP_param_a_gated3":
		net = Cloth_net_param_a_gated3(params.SMP_model_type,params.SMP_encoder_name)
	elif params.net == "Grad_net":
		net = Grad_net(params.SMP_model_type,params.SMP_encoder_name)
	elif params.net == "Grad_net_tiny":
		net = Grad_net_tiny()
	elif params.net == "Grad_net_scale_inv":
		net = Grad_net_scale_inv(params.hidden_size)
	return net

class Grad_net(nn.Module):
	
	def __init__(self, model_type, encoder_name):
		"""
		:model_type: ... see diverse models below
		:encoder_name: e.g. tu-mobilevitv2_100 or resnet34 ... more examples: https://github.com/qubvel/segmentation_models.pytorch
		"""
		
		super(Grad_net, self).__init__()
		if model_type=="Unet":
			SMP_model_type = smp.Unet
		elif model_type=="UnetPlusPlus":
			SMP_model_type = smp.UnetPlusPlus
		elif model_type=="MAnet":
			SMP_model_type = smp.MAnet
		elif model_type=="Linknet":
			SMP_model_type = smp.Linknet
		elif model_type=="FPN":
			SMP_model_type = smp.FPN
		elif model_type=="PSPNet":
			SMP_model_type = smp.PSPNet
		elif model_type=="PAN":
			SMP_model_type = smp.PAN
		elif model_type=="DeepLabV3":
			SMP_model_type = smp.DeepLabV3
		elif model_type=="DeepLabV3Plus":
			SMP_model_type = smp.DeepLabV3Plus
		else:
			raise Exception("invalid SMP_model_type!")
		self.model = SMP_model_type(encoder_name=encoder_name,in_channels=3,classes=3)
	
	def forward(self, grads, hidden_states=None):
		
		# CODO: use hidden state for concepts such as momentum / scaling
		
		std = grads.std([1,2,3]).detach().unsqueeze(1).unsqueeze(2).unsqueeze(3)
		
		step = self.model(grads/std)*std
		
		# TODO: scaling?
		
		return step

class Grad_net_tiny(nn.Module):
	
	def __init__(self):
		"""
		tiny vanilla gradient descent network with momentum
		"""
		
		super(Grad_net_tiny, self).__init__()
		self.momentum = Parameter(torch.ones(1)*0.01) # learnable momentum parameter ... could be also taken constant
		self.lr = Parameter(torch.ones(1)) # learnable learning rate Parameter
	
	def forward(self, grads, hidden_states=None):
		
		# CODO: use hidden state for concepts such as momentum / scaling
		
		hidden_states = [torch.zeros_like(grads[0:1]) if h is None else h for h in hidden_states]
		
		std = grads.std([1,2,3]).detach().unsqueeze(1).unsqueeze(2).unsqueeze(3)
		
		lp_grads = torch.cat(hidden_states,0) # momentum (low pass filtered grads)
		lp_grads = self.momentum*(grads/std) + (1-self.momentum)*lp_grads
		
		new_hidden_states = [lp_grads[i:i+1].detach() for i in range(lp_grads.shape[0])]
		
		step = (-self.lr) * lp_grads
		
		return step, new_hidden_states

class MixedUnet(nn.Module):
	# U-Net that outputs scalar as well as field values
	
	def __init__(self, in_channels, out_channels, out_scalar_channels,  hidden_size=64,bilinear=True):
		super(MixedUnet, self).__init__()
		self.hidden_size = hidden_size
		self.bilinear = bilinear

		self.inc = DoubleConv(in_channels, hidden_size)
		self.down1 = Down(hidden_size, 2*hidden_size)
		self.down2 = Down(2*hidden_size, 4*hidden_size)
		self.down3 = Down(4*hidden_size, 8*hidden_size)
		factor = 2 if bilinear else 1
		self.down4 = Down(8*hidden_size, 16*hidden_size // factor)
		self.up1 = Up(16*hidden_size, 8*hidden_size // factor, bilinear)
		self.up2 = Up(8*hidden_size, 4*hidden_size // factor, bilinear)
		self.up3 = Up(4*hidden_size, 2*hidden_size // factor, bilinear)
		self.up4 = Up(2*hidden_size, hidden_size, bilinear)
		self.outc = OutConv(hidden_size, out_channels)
		self.out_scalar = nn.Linear(16*hidden_size // factor,out_scalar_channels) # TODO

	def forward(self,inputs):
		x = inputs
		x1 = self.inc(x)
		x2 = self.down1(x1)
		x3 = self.down2(x2)
		x4 = self.down3(x3)
		x5 = self.down4(x4)
		x = self.up1(x5, x4)
		x = self.up2(x, x3)
		x = self.up3(x, x2)
		x = self.up4(x, x1)
		x = self.outc(x)
		
		x_scalar = self.out_scalar(torch.mean(x5,dim=[2,3]))
		return x, x_scalar



class Grad_net_scale_inv(nn.Module):
	
	def __init__(self,hidden_size=64,bilinear=True):
		
		super(Grad_net_scale_inv, self).__init__()
		self.initial_scale = 0.1 # ?
		self.nn = MixedUnet(3*3,3,1,hidden_size,bilinear)
	
	def forward(self, grads, hidden_states=None):
		
		# CODO: use hidden state for concepts such as momentum / scaling
		
		# hidden states for last gradients / last update step / scale
		hidden_states = [[torch.zeros_like(grads[0:1]),torch.zeros_like(grads[0:1]),torch.ones(1,1,1,1,device=device)*self.initial_scale] if h is None else h for h in hidden_states]
		
		last_grads = torch.cat([h[0] for h in hidden_states],0)
		last_steps = torch.cat([h[1] for h in hidden_states],0)
		last_scales = torch.cat([h[2] for h in hidden_states],0)
		
		std = grads.std([1,2,3]).detach().unsqueeze(1).unsqueeze(2).unsqueeze(3)
		
		normalized_grads = grads/std
		normalized_last_grads = last_grads/std
		
		inputs = torch.cat([normalized_grads, normalized_last_grads, last_steps],1)
		
		update_step, d_scale = self.nn(inputs)
		# CODO: more scaling?
		
		# gradient normalization (normalize_grads) => so gradients at different optimization stages get equal weights
		update_step = normalize_grads(update_step)
		d_scale = normalize_grads(d_scale)
		
		update_step = torch.tanh(update_step)
		d_scale = torch.exp(2*torch.tanh(d_scale/2))
		
		scales = last_scales*d_scale.unsqueeze(2).unsqueeze(3)
		step = update_step*scales
		
		#print(f"scales: {scales[:,0,0,0]} / d_scales: {d_scale[:,0]}") # => CODO: visualize that in test script...
		
		new_hidden_states = [[grads[i:i+1].detach(),update_step[i:i+1].detach(),scales[i:i+1].detach()] for i,_ in enumerate(hidden_states)]
		
		return step, new_hidden_states



class Cloth_net(nn.Module):
	
	def __init__(self, model_type, encoder_name):
		"""
		:model_type: ... see diverse models below
		:encoder_name: e.g. tu-mobilevitv2_100 or resnet34 ... more examples: https://github.com/qubvel/segmentation_models.pytorch
		"""
		
		super(Cloth_net, self).__init__()
		if model_type=="Unet":
			SMP_model_type = smp.Unet
		elif model_type=="UnetPlusPlus":
			SMP_model_type = smp.UnetPlusPlus
		elif model_type=="MAnet":
			SMP_model_type = smp.MAnet
		elif model_type=="Linknet":
			SMP_model_type = smp.Linknet
		elif model_type=="FPN":
			SMP_model_type = smp.FPN
		elif model_type=="PSPNet":
			SMP_model_type = smp.PSPNet
		elif model_type=="PAN":
			SMP_model_type = smp.PAN
		elif model_type=="DeepLabV3":
			SMP_model_type = smp.DeepLabV3
		elif model_type=="DeepLabV3Plus":
			SMP_model_type = smp.DeepLabV3Plus
		else:
			raise Exception("invalid SMP_model_type!")
		self.model = SMP_model_type(encoder_name=encoder_name,in_channels=12,classes=3)
	
	def forward(self, x_v, stiffnesses=None, shearings=None, bendings=None, a=None):
		bs,c,h,w = x_v.shape
		device = x_v.device
		di = torch.cat([x_v[:,:,1:]-x_v[:,:,:-1],torch.zeros(bs,c,1,w,device=device)],dim=2)
		dj = torch.cat([x_v[:,:,:,1:]-x_v[:,:,:,:-1],torch.zeros(bs,c,h,1,device=device)],dim=3)
		x = torch.cat([di,dj],dim=1)
		x = self.model(x)
		return 10*torch.tanh(x/10)

class Cloth_net_param(nn.Module):
	
	def __init__(self, model_type, encoder_name):
		"""
		same as Cloth_net but makes use of additional parameters for stiffness, shearing and bending
		:model_type: ... see diverse models below
		:encoder_name: e.g. tu-mobilevitv2_100 or resnet34 ... more examples: https://github.com/qubvel/segmentation_models.pytorch
		"""
		
		super(Cloth_net_param, self).__init__()
		if model_type=="Unet":
			SMP_model_type = smp.Unet
		elif model_type=="UnetPlusPlus":
			SMP_model_type = smp.UnetPlusPlus
		elif model_type=="MAnet":
			SMP_model_type = smp.MAnet
		elif model_type=="Linknet":
			SMP_model_type = smp.Linknet
		elif model_type=="FPN":
			SMP_model_type = smp.FPN
		elif model_type=="PSPNet":
			SMP_model_type = smp.PSPNet
		elif model_type=="PAN":
			SMP_model_type = smp.PAN
		elif model_type=="DeepLabV3":
			SMP_model_type = smp.DeepLabV3
		elif model_type=="DeepLabV3Plus":
			SMP_model_type = smp.DeepLabV3Plus
		else:
			raise Exception("invalid SMP_model_type!")
		self.model = SMP_model_type(encoder_name=encoder_name,in_channels=12+3,classes=3)
	
	def forward(self, x_v, stiffnesses, shearings, bendings, a=None):
		bs,c,h,w = x_v.shape
		device = x_v.device
		ones = torch.ones(1,1,h,w,device=device)
		di = torch.cat([x_v[:,:,1:]-x_v[:,:,:-1],torch.zeros(bs,c,1,w,device=device)],dim=2)
		dj = torch.cat([x_v[:,:,:,1:]-x_v[:,:,:,:-1],torch.zeros(bs,c,h,1,device=device)],dim=3)
		x = torch.cat([di,dj,torch.log(stiffnesses).unsqueeze(1).unsqueeze(2).unsqueeze(3)*ones,torch.log(shearings).unsqueeze(1).unsqueeze(2).unsqueeze(3)*ones,torch.log(bendings).unsqueeze(1).unsqueeze(2).unsqueeze(3)*ones],dim=1)
		x = self.model(x)
		return 10*torch.tanh(x/10)

class Cloth_net_param_a(nn.Module):
	
	def __init__(self, model_type, encoder_name):
		"""
		same as Cloth_net_param but takes additional parameter for external forces / accelerations
		:model_type: ... see diverse models below
		:encoder_name: e.g. tu-mobilevitv2_100 or resnet34 ... more examples: https://github.com/qubvel/segmentation_models.pytorch
		"""
		
		super(Cloth_net_param_a, self).__init__()
		if model_type=="Unet":
			SMP_model_type = smp.Unet
		elif model_type=="UnetPlusPlus":
			SMP_model_type = smp.UnetPlusPlus
		elif model_type=="MAnet":
			SMP_model_type = smp.MAnet
		elif model_type=="Linknet":
			SMP_model_type = smp.Linknet
		elif model_type=="FPN":
			SMP_model_type = smp.FPN
		elif model_type=="PSPNet":
			SMP_model_type = smp.PSPNet
		elif model_type=="PAN":
			SMP_model_type = smp.PAN
		elif model_type=="DeepLabV3":
			SMP_model_type = smp.DeepLabV3
		elif model_type=="DeepLabV3Plus":
			SMP_model_type = smp.DeepLabV3Plus
		else:
			raise Exception("invalid SMP_model_type!")
		self.model = SMP_model_type(encoder_name=encoder_name,in_channels=12+3+3,classes=3)
	
	def forward(self, x_v, stiffnesses, shearings, bendings, a):
		bs,c,h,w = x_v.shape
		device = x_v.device
		ones = torch.ones(1,1,h,w,device=device)
		di = torch.cat([x_v[:,:,1:]-x_v[:,:,:-1],torch.zeros(bs,c,1,w,device=device)],dim=2)
		dj = torch.cat([x_v[:,:,:,1:]-x_v[:,:,:,:-1],torch.zeros(bs,c,h,1,device=device)],dim=3)
		# CODO: normalize a to have 0-mean
		x = torch.cat([di,dj,torch.log(stiffnesses).unsqueeze(1).unsqueeze(2).unsqueeze(3)*ones,torch.log(shearings).unsqueeze(1).unsqueeze(2).unsqueeze(3)*ones,torch.log(bendings).unsqueeze(1).unsqueeze(2).unsqueeze(3)*ones,a],dim=1)
		x = self.model(x)
		return 10*torch.tanh(x/10)
		
class Cloth_net_param_a_gated(nn.Module):
	
	def __init__(self, model_type, encoder_name):
		"""
		same as Cloth_net_param_a but allows to pass external accelerations through gating mechanism
		:model_type: ... see diverse models below
		:encoder_name: e.g. tu-mobilevitv2_100 or resnet34 ... more examples: https://github.com/qubvel/segmentation_models.pytorch
		"""
		
		super(Cloth_net_param_a_gated, self).__init__()
		if model_type=="Unet":
			SMP_model_type = smp.Unet
		elif model_type=="UnetPlusPlus":
			SMP_model_type = smp.UnetPlusPlus
		elif model_type=="MAnet":
			SMP_model_type = smp.MAnet
		elif model_type=="Linknet":
			SMP_model_type = smp.Linknet
		elif model_type=="FPN":
			SMP_model_type = smp.FPN
		elif model_type=="PSPNet":
			SMP_model_type = smp.PSPNet
		elif model_type=="PAN":
			SMP_model_type = smp.PAN
		elif model_type=="DeepLabV3":
			SMP_model_type = smp.DeepLabV3
		elif model_type=="DeepLabV3Plus":
			SMP_model_type = smp.DeepLabV3Plus
		else:
			raise Exception("invalid SMP_model_type!")
		self.model = SMP_model_type(encoder_name=encoder_name,in_channels=12+3+3,classes=3+2)
	
	def forward(self, x_v, stiffnesses, shearings, bendings, a):
		bs,c,h,w = x_v.shape
		device = x_v.device
		ones = torch.ones(1,1,h,w,device=device)
		di = torch.cat([x_v[:,:,1:]-x_v[:,:,:-1],torch.zeros(bs,c,1,w,device=device)],dim=2)
		dj = torch.cat([x_v[:,:,:,1:]-x_v[:,:,:,:-1],torch.zeros(bs,c,h,1,device=device)],dim=3)
		# CODO: normalize a to have 0-mean
		x = torch.cat([di,dj,torch.log(stiffnesses).unsqueeze(1).unsqueeze(2).unsqueeze(3)*ones,torch.log(shearings).unsqueeze(1).unsqueeze(2).unsqueeze(3)*ones,torch.log(bendings).unsqueeze(1).unsqueeze(2).unsqueeze(3)*ones,a],dim=1)
		x = self.model(x)
		return 10*torch.tanh(x[:,0:3]/10)*torch.sigmoid(x[:,3:4])+a*torch.sigmoid(x[:,4:5])

class Cloth_net_param_a_gated2(nn.Module):
	
	def __init__(self, model_type, encoder_name):
		"""
		same as Cloth_net_param_a_gated but passes additionally normalized external forces (improves performance for small external forces)
		:model_type: ... see diverse models below
		:encoder_name: e.g. tu-mobilevitv2_100 or resnet34 ... more examples: https://github.com/qubvel/segmentation_models.pytorch
		"""
		
		super(Cloth_net_param_a_gated2, self).__init__()
		if model_type=="Unet":
			SMP_model_type = smp.Unet
		elif model_type=="UnetPlusPlus":
			SMP_model_type = smp.UnetPlusPlus
		elif model_type=="MAnet":
			SMP_model_type = smp.MAnet
		elif model_type=="Linknet":
			SMP_model_type = smp.Linknet
		elif model_type=="FPN":
			SMP_model_type = smp.FPN
		elif model_type=="PSPNet":
			SMP_model_type = smp.PSPNet
		elif model_type=="PAN":
			SMP_model_type = smp.PAN
		elif model_type=="DeepLabV3":
			SMP_model_type = smp.DeepLabV3
		elif model_type=="DeepLabV3Plus":
			SMP_model_type = smp.DeepLabV3Plus
		else:
			raise Exception("invalid SMP_model_type!")
		self.model = SMP_model_type(encoder_name=encoder_name,in_channels=12+3+3+3,classes=3+2)
	
	def forward(self, x_v, stiffnesses, shearings, bendings, a):
		bs,c,h,w = x_v.shape
		device = x_v.device
		ones = torch.ones(1,1,h,w,device=device)
		di = torch.cat([x_v[:,:,1:]-x_v[:,:,:-1],torch.zeros(bs,c,1,w,device=device)],dim=2)
		dj = torch.cat([x_v[:,:,:,1:]-x_v[:,:,:,:-1],torch.zeros(bs,c,h,1,device=device)],dim=3)
		# CODO: normalize a to have 0-mean
		a_norm = torch.nn.functional.normalize(a,1)
		x = torch.cat([di,dj,torch.log(stiffnesses).unsqueeze(1).unsqueeze(2).unsqueeze(3)*ones,torch.log(shearings).unsqueeze(1).unsqueeze(2).unsqueeze(3)*ones,torch.log(bendings).unsqueeze(1).unsqueeze(2).unsqueeze(3)*ones,a,a_norm],dim=1)
		x = self.model(x)
		return 10*torch.tanh(x[:,0:3]/10)*torch.sigmoid(x[:,3:4])+a*torch.sigmoid(x[:,4:5])
		
class Cloth_net_param_a_gated3(nn.Module):
	
	def __init__(self, model_type, encoder_name):
		"""
		same as Cloth_net_param_a_gated2 but doesn't pass dx but dx-L. This way only the deviations from the resting length are passed through the network => hopefully, this helps the network to learn better dynamics.
		:model_type: ... see diverse models below
		:encoder_name: e.g. tu-mobilevitv2_100 or resnet34 ... more examples: https://github.com/qubvel/segmentation_models.pytorch
		"""
		
		super(Cloth_net_param_a_gated3, self).__init__()
		if model_type=="Unet":
			SMP_model_type = smp.Unet
		elif model_type=="UnetPlusPlus":
			SMP_model_type = smp.UnetPlusPlus
		elif model_type=="MAnet":
			SMP_model_type = smp.MAnet
		elif model_type=="Linknet":
			SMP_model_type = smp.Linknet
		elif model_type=="FPN":
			SMP_model_type = smp.FPN
		elif model_type=="PSPNet":
			SMP_model_type = smp.PSPNet
		elif model_type=="PAN":
			SMP_model_type = smp.PAN
		elif model_type=="DeepLabV3":
			SMP_model_type = smp.DeepLabV3
		elif model_type=="DeepLabV3Plus":
			SMP_model_type = smp.DeepLabV3Plus
		else:
			raise Exception("invalid SMP_model_type!")
		self.model = SMP_model_type(encoder_name=encoder_name,in_channels=12+3+3+3+6,classes=3+2)
	
	def forward(self, x_v, stiffnesses, shearings, bendings, a):
		bs,c,h,w = x_v.shape
		device = x_v.device
		ones = torch.ones(1,1,h,w,device=device)
		di = torch.cat([x_v[:,:,1:]-x_v[:,:,:-1],torch.zeros(bs,c,1,w,device=device)],dim=2)
		dj = torch.cat([x_v[:,:,:,1:]-x_v[:,:,:,:-1],torch.zeros(bs,c,h,1,device=device)],dim=3)
		dxi = di[:,:3]
		dxi = dxi - params.L_0*torch.nn.functional.normalize(dxi,1)
		dxj = dj[:,:3]
		dxj = dxj - params.L_0*torch.nn.functional.normalize(dxj,1)
		
		a_norm = torch.nn.functional.normalize(a,1)
		x = torch.cat([dxi,di,dxj,dj,torch.log(stiffnesses).unsqueeze(1).unsqueeze(2).unsqueeze(3)*ones,torch.log(shearings).unsqueeze(1).unsqueeze(2).unsqueeze(3)*ones,torch.log(bendings).unsqueeze(1).unsqueeze(2).unsqueeze(3)*ones,a,a_norm],dim=1)
		x = self.model(x)
		return 10*torch.tanh(x[:,0:3]/10)*torch.sigmoid(x[:,3:4])+a*torch.sigmoid(x[:,4:5])

class Cloth_Unet(nn.Module):
	
	def __init__(self, hidden_size):
		"""
		:hidden_size: hidden_size of UNet
		"""
		
		super(Cloth_Unet, self).__init__()
		self.hidden_size = hidden_size
		self.conv1 = nn.Conv2d( 12, self.hidden_size,kernel_size=[3,3],padding=[1,1])
		self.conv2 = nn.Conv2d( self.hidden_size, self.hidden_size,kernel_size=5,padding=0,stride=2)
		self.conv3 = nn.Conv2d( self.hidden_size, self.hidden_size,kernel_size=5,padding=0,stride=2)
		self.conv4 = nn.Conv2d( self.hidden_size, self.hidden_size,kernel_size=5,padding=0,stride=2)
		self.conv5 = nn.Conv2d( self.hidden_size, self.hidden_size,kernel_size=5,padding=0,stride=2)
		self.deconv1 = nn.ConvTranspose2d( self.hidden_size, self.hidden_size, kernel_size=5, stride = 2, padding=0)
		self.deconv2 = nn.ConvTranspose2d( 2*self.hidden_size, self.hidden_size, kernel_size=5, stride = 2, padding=0)
		self.deconv3 = nn.ConvTranspose2d( 2*self.hidden_size, self.hidden_size, kernel_size=5, stride = 2, padding=0)
		self.deconv4 = nn.ConvTranspose2d( 2*self.hidden_size, self.hidden_size, kernel_size=5, stride = 2, padding=0)
		self.deconv5 = nn.ConvTranspose2d( 2*self.hidden_size, self.hidden_size, kernel_size=5, stride = 2, padding=0)
		self.conv6 = nn.Conv2d( 2*self.hidden_size,3,kernel_size=[3,3],padding=[1,1])
	
	def forward(self,x_v, stiffnesses=None, shearings=None, bendings=None, a=None):
		bs,c,h,w = x_v.shape
		device = x_v.device
		di = torch.cat([x_v[:,:,1:]-x_v[:,:,:-1],torch.zeros(bs,c,1,w,device=device)],dim=2)
		dj = torch.cat([x_v[:,:,:,1:]-x_v[:,:,:,:-1],torch.zeros(bs,c,h,1,device=device)],dim=3)
		x = torch.cat([di,dj],dim=1)
		
		x1 = torch.sigmoid(self.conv1(x))
		x2 = torch.sigmoid(self.conv2(x1))
		x3 = torch.sigmoid(self.conv3(x2))
		x4 = torch.sigmoid(self.conv4(x3))
		x = torch.sigmoid(self.conv5(x4))
		x = torch.sigmoid(self.deconv1(x, output_size = [x4.shape[2],x4.shape[3]]))
		x = torch.cat([x,x4],dim=1)
		x = torch.sigmoid(self.deconv2(x, output_size = [x3.shape[2],x3.shape[3]]))
		x = torch.cat([x,x3],dim=1)
		x = torch.sigmoid(self.deconv4(x, output_size = [x2.shape[2],x2.shape[3]]))
		x = torch.cat([x,x2],dim=1)
		x = torch.sigmoid(self.deconv5(x, output_size = [x1.shape[2],x1.shape[3]]))
		x = torch.cat([x,x1],dim=1)
		x = self.conv6(x)
		
		return 10*torch.tanh(x/10)

class Cloth_Unet_param_a(nn.Module):
	
	def __init__(self, hidden_size):
		"""
		:hidden_size: hidden_size of UNet
		"""
		
		super(Cloth_Unet_param_a, self).__init__()
		self.hidden_size = hidden_size
		self.conv1 = nn.Conv2d( 12+3+3, self.hidden_size,kernel_size=[3,3],padding=[1,1])
		self.conv2 = nn.Conv2d( self.hidden_size, self.hidden_size,kernel_size=5,padding=0,stride=2)
		self.conv3 = nn.Conv2d( self.hidden_size, self.hidden_size,kernel_size=5,padding=0,stride=2)
		self.conv4 = nn.Conv2d( self.hidden_size, self.hidden_size,kernel_size=5,padding=0,stride=2)
		self.conv5 = nn.Conv2d( self.hidden_size, self.hidden_size,kernel_size=5,padding=0,stride=2)
		self.deconv1 = nn.ConvTranspose2d( self.hidden_size, self.hidden_size, kernel_size=5, stride = 2, padding=0)
		self.deconv2 = nn.ConvTranspose2d( 2*self.hidden_size, self.hidden_size, kernel_size=5, stride = 2, padding=0)
		self.deconv3 = nn.ConvTranspose2d( 2*self.hidden_size, self.hidden_size, kernel_size=5, stride = 2, padding=0)
		self.deconv4 = nn.ConvTranspose2d( 2*self.hidden_size, self.hidden_size, kernel_size=5, stride = 2, padding=0)
		self.deconv5 = nn.ConvTranspose2d( 2*self.hidden_size, self.hidden_size, kernel_size=5, stride = 2, padding=0)
		self.conv6 = nn.Conv2d( 2*self.hidden_size,3,kernel_size=[3,3],padding=[1,1])
	
	def forward(self,x_v, stiffnesses, shearings, bendings, a):
		bs,c,h,w = x_v.shape
		device = x_v.device
		ones = torch.ones(1,1,h,w,device=device)
		di = torch.cat([x_v[:,:,1:]-x_v[:,:,:-1],torch.zeros(bs,c,1,w,device=device)],dim=2)
		dj = torch.cat([x_v[:,:,:,1:]-x_v[:,:,:,:-1],torch.zeros(bs,c,h,1,device=device)],dim=3)
		x = torch.cat([di,dj,torch.log(stiffnesses).unsqueeze(1).unsqueeze(2).unsqueeze(3)*ones,torch.log(shearings).unsqueeze(1).unsqueeze(2).unsqueeze(3)*ones,torch.log(bendings).unsqueeze(1).unsqueeze(2).unsqueeze(3)*ones,a],dim=1)
		
		x1 = torch.sigmoid(self.conv1(x))
		x2 = torch.sigmoid(self.conv2(x1))
		x3 = torch.sigmoid(self.conv3(x2))
		x4 = torch.sigmoid(self.conv4(x3))
		x = torch.sigmoid(self.conv5(x4))
		x = torch.sigmoid(self.deconv1(x, output_size = [x4.shape[2],x4.shape[3]]))
		x = torch.cat([x,x4],dim=1)
		x = torch.sigmoid(self.deconv2(x, output_size = [x3.shape[2],x3.shape[3]]))
		x = torch.cat([x,x3],dim=1)
		x = torch.sigmoid(self.deconv4(x, output_size = [x2.shape[2],x2.shape[3]]))
		x = torch.cat([x,x2],dim=1)
		x = torch.sigmoid(self.deconv5(x, output_size = [x1.shape[2],x1.shape[3]]))
		x = torch.cat([x,x1],dim=1)
		x = self.conv6(x)
		
		return 10*torch.tanh(x/10)
