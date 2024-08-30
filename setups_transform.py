import torch
import numpy as np
from get_param import params,toCuda,toCpu

# this script contains multistep dataset functionality to:
# transform datasets with multiple channels into dataset with one channel (this is needed to concatenate multiple datasets with different numbers of channels)
# concatenate multiple datasets

class DatasetToSingleChannel:
	# transform datasets with multiple channels into dataset with one channel 
	# (this is needed to concatenate multiple datasets with different numbers of channels)
	
	def __init__(self,multi_channel_dataset):
		"""
		:multi_channel_dataset: dataset with multiple channels (e.g. cloth with x/y/z or fluid with a/p channels)
		"""
		self.multi_channel_dataset = multi_channel_dataset
	
	def ask(self,indices=None):
		"""
		ask for a batch from multi_channel_dataset. The multi channel samples are split into single channel samples.
		:indices: optional indices of samples to draw from training pool (shape: batch_size)
		:return: 
			:grads: gradients for accelerations (shape: batch_size*n_channels x 1 x h x w)
			:hidden_states: list of length batch_size * n_channels that contains the hidden_states 
							(list entries are None if corresponding hidden_states are not yet set)
		"""
		grads, hidden_states = self.multi_channel_dataset.ask(indices)
		self.bs, self.c, self.h, self.w = grads.shape
		split_grads = grads.reshape(self.bs*self.c,1,self.h,self.w)
		hidden_states = [[None for _ in range(self.c)] if hs is None else hs for hs in hidden_states]
		split_hidden_states = [hs_split for hs_merged in hidden_states for hs_split in hs_merged]
		return split_grads, split_hidden_states
	
	def tell(self,step, hidden_states=None):
		"""
		The single channel update steps and hidden_states are merged into multi channel updates.
		:step: update step for gradients given by ask(). (shape: batch_size*n_channels x 1 x h x w)
		:hidden_states: list of length batch_size * n_channels that contains the hidden_states that should be stored.
						This is useful to store e.g. momentum / variance / last update steps etc. 
						If None: no hidden_states are stored.
		:return: loss to optimize neural update-step-model
		"""
		merge_step = step.reshape(self.bs,self.c,self.h,self.w)
		merge_hidden_states = None if hidden_states is None else [hidden_states[i*self.c:(i+1)*self.c] for i in range(self.bs)]
		l = self.multi_channel_dataset.tell(merge_step,merge_hidden_states)
		return l


class DatasetConcat: # TODO
	# transform datasets with multiple channels into dataset with one channel 
	# (this is needed to concatenate multiple datasets with different numbers of channels)
	
	def __init__(self,datasets):
		self.datasets = datasets
		
	
	def ask(self,indices=None):
		"""
		:indices: optional indices of samples to draw from training pool (shape: batch_size)
		:return: 
			gradients for accelerations (shape: batch_size x 3 x h x w)
			hidden_states for optimizer
		"""
		results = [ds.ask() for ds in self.datasets]
		# TODO: extract batch sizes so everything can be properly put together afterwards again....
		grads = torch.cat([r[0] for r in results],0)
		#hidden_states = [*r[1] for r in results] # wird wrsl nicht tun...
		
		return grads, hidden_states
	
	def tell(self,step, hidden_states=None):
		"""
		:step: update step for gradients given by ask()
		:return: loss to optimize neural update-step-model
		"""
		# TODO
		return torch.mean(l)
