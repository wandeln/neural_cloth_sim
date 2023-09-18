import segmentation_models_pytorch as smp 
import torch
from torch import nn

def get_Net(params):
	if params.net == "UNet":
		net = Cloth_Unet(params.hidden_size)
	elif params.net == "SMP":
		net = Cloth_net(params.SMP_model_type,params.SMP_encoder_name)
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
	
	def forward(self,x_v):
		bs,c,h,w = x_v.shape
		device = x_v.device
		di = torch.cat([x_v[:,:,1:]-x_v[:,:,:-1],torch.zeros(bs,c,1,w,device=device)],dim=2)
		dj = torch.cat([x_v[:,:,:,1:]-x_v[:,:,:,:-1],torch.zeros(bs,c,h,1,device=device)],dim=3)
		x = torch.cat([di,dj],dim=1)
		x = self.model(x)
		return 10*torch.tanh(x/10)

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
	
	def forward(self,x_v):
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