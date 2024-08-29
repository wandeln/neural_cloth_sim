from utils import *

scale = torch.tensor([1.,2,3])

# test scale_grads
if False:
	x = torch.ones(3,requires_grad=True)


	print(f"x: {x} / scale: {scale}")

	y = scale_grads(x,scale)

	print(f"y: {y}")

	loss = 0.5*torch.sum(y**2)

	loss.backward()

	print(f"grads: {x.grad}")



# test normalize_grads
if True:
	x = torch.randn(3,1,10,10,requires_grad=True)

	y = scale.unsqueeze(1).unsqueeze(2).unsqueeze(3)*normalize_grads(x)
	#y = scale.unsqueeze(1).unsqueeze(2).unsqueeze(3)*x

	print(f"y: {y.shape}")


	loss = torch.sum(y**2)
	loss.backward()

	print(f"normalized grads: {torch.norm(x.grad,dim=(2,3))}")

