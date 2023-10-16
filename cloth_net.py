import segmentation_models_pytorch as smp 
import torch
from torch import nn
from get_param import params

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
	return net

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
