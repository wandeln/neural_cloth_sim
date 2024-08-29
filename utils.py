import torch
from torch.autograd import Function
from torch.nn.functional import normalize

# "pseudo function" that doesn't affect the outputs but only scales the gradients

class ScaleGrads(Function):
	@staticmethod
	def forward(input, scale):
		return input
	
	@staticmethod
	def setup_context(ctx, inputs, output):
		input, scale = inputs
		ctx.save_for_backward(scale)
	
	@staticmethod
	def backward(ctx, grad_output):
		scale = ctx.saved_tensors[0]
		return scale*grad_output, None # no gradients for gradient scaling

scale_grads = ScaleGrads.apply

# "pseudo function" that doesn't affect the outputs but only normalizes the gradients
class NormalizeGrads(Function):
	@staticmethod
	def forward(input):
		return input
	
	@staticmethod
	def setup_context(ctx, inputs, output):
		pass
		
	@staticmethod
	def backward(ctx, grad_output):
		# achtung, normalization kann sehr hohe / niedrige werte zurückgeben, wenn fast alles = 0 ist => min / max clampen
		return normalize(grad_output,dim=[i+1 for i in range(len(grad_output.shape)-1)]).clamp(min=-10,max=10)

normalize_grads = NormalizeGrads.apply
