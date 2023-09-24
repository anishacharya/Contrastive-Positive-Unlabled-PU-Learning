""" Defines different losses for training """
import torch.nn as nn
import torch


class SelfSupConLoss(nn.Module):
	"""
	Adopted from lightly.loss.NTXentLoss :
	https://github.com/lightly-ai/lightly/blob/master/lightly/loss/ntx_ent_loss.py
	"""
	
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
		# get inner product matrix - diag zero
		batch_size, _ = z.shape
		
		# project onto hypersphere
		z = nn.functional.normalize(z, dim=1)
		z_aug = nn.functional.normalize(z_aug, dim=1)
		
		# calculate similarities block-wise - the resulting vectors have shape (batch_size, batch_size)
		inner_pdt_00 = torch.einsum('nc,mc->nm', z, z) / self.temperature
		inner_pdt_01 = torch.einsum('nc,mc->nm', z, z_aug) / self.temperature
		inner_pdt_10 = torch.einsum("nc,mc->nm", z_aug, z) / self.temperature
		inner_pdt_11 = torch.einsum('nc,mc->nm', z_aug, z_aug) / self.temperature
		
		# remove similarities between same views of the same image
		diag_mask = torch.eye(batch_size, device=z.device, dtype=torch.bool)
		inner_pdt_00 = inner_pdt_00[~diag_mask].view(batch_size, -1)
		inner_pdt_11 = inner_pdt_11[~diag_mask].view(batch_size, -1)
		
		# concatenate blocks : o/p shape (2*batch_size, 2*batch_size) - diagonals (self sim) zero
		# [ Block 00 ] | [ Block 01 ]
		# [ Block 10 ] | [ Block 11 ]
		inner_pdt_0100 = torch.cat([inner_pdt_01, inner_pdt_00], dim=1)
		inner_pdt_1011 = torch.cat([inner_pdt_10, inner_pdt_11], dim=1)
		logits = torch.cat([inner_pdt_0100, inner_pdt_1011], dim=0)
		
		labels = torch.arange(batch_size, device=z.device, dtype=torch.long)
		labels = labels.repeat(2)
		loss = self.cross_entropy(logits, labels)
		
		return loss


class SupConLoss(nn.Module):
	"""
	Adopted from lightly.loss.NTXentLoss :
	https://github.com/lightly-ai/lightly/blob/master/lightly/loss/ntx_ent_loss.py
	"""
	
	def __init__(self, temperature: float = 0.5):
		super(SupConLoss, self).__init__()
		self.temperature = temperature
	
	def forward(self, z: torch.Tensor, z_aug: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
		"""
		
		:param z: features => bs * shape
		:param z_aug: augmentations => bs * shape
		:param labels: ground truth labels of size => bs
		:return: loss value => scalar
		"""
		batch_size, _ = z.shape
		
		# project onto hypersphere
		z = nn.functional.normalize(z, dim=1)
		z_aug = nn.functional.normalize(z_aug, dim=1)
		
		# calculate similarities block-wise - the resulting vectors have shape (batch_size, batch_size)
		inner_pdt_00 = torch.einsum('nc,mc->nm', z, z) / self.temperature
		inner_pdt_01 = torch.einsum('nc,mc->nm', z, z_aug) / self.temperature
		inner_pdt_10 = torch.einsum("nc,mc->nm", z_aug, z) / self.temperature
		inner_pdt_11 = torch.einsum('nc,mc->nm', z_aug, z_aug) / self.temperature
		
		# concatenate blocks : o/p shape (2*batch_size, 2*batch_size) - diagonals (self sim) zero
		# [ Block 00 ] | [ Block 01 ]
		# [ Block 10 ] | [ Block 11 ]
		inner_pdt_0001 = torch.cat([inner_pdt_00, inner_pdt_01], dim=1)
		inner_pdt_1011 = torch.cat([inner_pdt_10, inner_pdt_11], dim=1)
		inner_pdt_mtx = torch.cat([inner_pdt_0001, inner_pdt_1011], dim=0)
		
		max_inner_pdt, _ = torch.max(inner_pdt_mtx, dim=1, keepdim=True)
		inner_pdt_mtx = inner_pdt_mtx - max_inner_pdt.detach()  # for numerical stability
		
		# compute negative log-likelihoods
		nll_mtx = torch.exp(inner_pdt_mtx)
		# mask out self contrast
		diag_mask = torch.ones_like(inner_pdt_mtx).fill_diagonal_(0)
		nll_mtx = nll_mtx * diag_mask
		nll_mtx /= torch.sum(nll_mtx, dim=1, keepdim=True)
		nll_mtx[nll_mtx != 0] = - torch.log(nll_mtx[nll_mtx != 0])
		
		# mask out contributions from samples not from same class as i
		mask_label = torch.unsqueeze(labels, dim=-1)
		eq_mask = torch.eq(mask_label, torch.t(mask_label))
		eq_mask = torch.tile(eq_mask, (2, 2))
		similarity_scores = nll_mtx * eq_mask
		
		# compute the loss -by averaging over multiple positives
		loss = similarity_scores.sum(dim=1) / (eq_mask.sum(dim=1) - 1)
		loss = torch.mean(loss)
		
		return loss

