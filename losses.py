""" Defines different losses for training """
import torch.nn as nn
import torch


def compute_inner_pdt_mtx(z: torch.Tensor, z_aug: torch.Tensor, temp: float) -> torch.Tensor:
	"""
	compute Temp normalized - cross similarity (inner product) scores
	:param z:
	:param z_aug:
	:param temp:
	"""
	batch_size, _ = z.shape
	
	# project onto hypersphere
	z = nn.functional.normalize(z, dim=1)
	z_aug = nn.functional.normalize(z_aug, dim=1)
	
	# calculate similarities block-wise - the resulting vectors have shape (batch_size, batch_size)
	inner_pdt_00 = torch.einsum('nc,mc->nm', z, z) / temp
	inner_pdt_01 = torch.einsum('nc,mc->nm', z, z_aug) / temp
	inner_pdt_10 = torch.t(inner_pdt_01)
	inner_pdt_11 = torch.einsum('nc,mc->nm', z_aug, z_aug) / temp
	
	# remove similarities between same views of the same image
	diag_mask = torch.eye(batch_size, device=z.device, dtype=torch.bool)
	inner_pdt_00 = inner_pdt_00[~diag_mask].view(batch_size, -1)
	inner_pdt_11 = inner_pdt_11[~diag_mask].view(batch_size, -1)
	
	# concatenate blocks : o/p shape (2*batch_size, 2*batch_size) - diagonals (self sim) zero
	# [ Block 00 ] | [ Block 01 ]
	# [ Block 10 ] | [ Block 11 ]
	inner_pdt_0001 = torch.cat([inner_pdt_00, inner_pdt_01], dim=1)
	inner_pdt_1011 = torch.cat([inner_pdt_10, inner_pdt_11], dim=1)
	inner_pdt_mtx = torch.cat([inner_pdt_0001, inner_pdt_1011], dim=0)
	
	return inner_pdt_mtx


class SelfSupConLoss(nn.Module):
	def __init__(self, temperature: float = 0.5):
		super(SelfSupConLoss, self).__init__()
		self.temperature = temperature
		self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")
	
	def forward(self, z: torch.Tensor, z_aug: torch.Tensor, *kwargs) -> torch.Tensor:
		"""
		:param z: features
		:param z_aug: augmentations
		:return: loss value, scalar
		"""
