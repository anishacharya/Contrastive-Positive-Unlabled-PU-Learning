""" Defines different losses for training """
import torch.nn as nn
import torch
from typing import Dict
from lightly.loss import NTXentLoss


def get_loss(framework_config: Dict) -> nn.Module:
	"""
	:param framework_config:
	"""
	loss_fn = framework_config.get('loss')
	temp = framework_config.get('temp', 0.5)
	prior = framework_config.get('prior', 0)
	
	if loss_fn == 'ce':
		return nn.CrossEntropyLoss()
	elif loss_fn in ['uPU', 'nnPU']:
		return PULoss(loss_fn=loss_fn, prior=prior)
	elif loss_fn == 'ssCL':
		return SelfSupConLoss(temperature=temp, reduction='mean')
	elif loss_fn == 'sCL':
		return SupConLoss(temperature=temp, reduction='mean')
	elif loss_fn == 'puCL':
		# ICLR 2024 Submission
		return PUConLoss(temperature=temp)
	elif loss_fn == 'puNCE':
		raise NotImplementedError
	else:
		raise NotImplementedError


class SelfSupConLoss(nn.Module):
	"""
	Self Sup Con Loss: https://arxiv.org/abs/2002.05709
	Adopted from lightly.loss.NTXentLoss :
	https://github.com/lightly-ai/lightly/blob/master/lightly/loss/ntx_ent_loss.py
	"""
	
	def __init__(self, temperature: float = 0.5, reduction="mean"):
		super(SelfSupConLoss, self).__init__()
		self.temperature = temperature
		self.reduction = reduction
		
		self.neg_mask = None
		self.self_aug_mask = None
		self.bs = None
		
		self.criterion = NTXentLoss(
			temperature=self.temperature,
		)
	
	def forward(self, z, z_aug, *kwargs):
		"""
		:param z:
		:param z_aug:
		:param kwargs:
		:return:
		"""
		# # compute matrix with <z_i , z_j> / temp
		# inner_pdt_mtx = compute_inner_pdt_mtx(z=z, z_aug=z_aug, temp=self.temperature)
		# # softmax row wise -- w/o diagonal i.e. inner_pdt / Z
		# similarity_mtx = compute_sfx_mtx(inner_pdt_mtx=inner_pdt_mtx)
		#
		# # get masks - runs once for a set of images of same shape
		# if self.bs != z.shape[0] or self.self_aug_mask is None:
		# 	self.bs = z.shape[0]
		# 	self.self_aug_mask, _ = get_self_aug_mask(z=z)
		#
		# loss = (similarity_mtx * self.self_aug_mask)
		#
		# return torch.mean(loss) if self.reduction == 'mean' else loss
		return self.criterion(z, z_aug)


class SupConLoss(nn.Module):
	"""
	Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
	Attractive force between self augmentation and all other samples from same class
	"""
	
	def __init__(self, temperature: float = 0.5, reduction="mean"):
		super(SupConLoss, self).__init__()
		self.temperature = temperature
		self.reduction = reduction
	
	def forward(self, z: torch.Tensor, z_aug: torch.Tensor, labels: torch.Tensor, *kwargs) -> torch.Tensor:
		"""
		
		:param z: features => bs * shape
		:param z_aug: augmentations => bs * shape
		:param labels: ground truth labels of size => bs
		:return: loss value => scalar
		"""
		
		# compute matrix with <z_i , z_j> / temp
		inner_pdt_mtx = compute_inner_pdt_mtx(z=z, z_aug=z_aug, temp=self.temperature)
		# softmax row wise -- w/o diagonal i.e. inner_pdt / Z
		similarity_mtx = compute_sfx_mtx(inner_pdt_mtx=inner_pdt_mtx)
		
		# mask out contributions from samples not from same class as i
		mask_label = torch.unsqueeze(labels, dim=-1)
		eq_mask = torch.eq(mask_label, torch.t(mask_label))
		eq_mask = torch.tile(eq_mask, (2, 2))
		similarity_scores = similarity_mtx * eq_mask
		
		# compute the loss by averaging over multiple positives
		loss = similarity_scores.sum(dim=1) / (eq_mask.sum(dim=1) - 1)
		
		return torch.mean(loss) if self.reduction == 'mean' else loss


class PUConLoss(nn.Module):
	"""
    Proposed PUConLoss : leveraging available positives only
    """
	
	def __init__(self, temperature: float = 0.5):
		super(PUConLoss, self).__init__()
		# per sample unsup and sup loss : since reduction is None
		self.sscl = SelfSupConLoss(temperature=temperature, reduction='none')
		self.scl = SupConLoss(temperature=temperature, reduction='none')
	
	def forward(self, z: torch.Tensor, z_aug: torch.Tensor, labels: torch.Tensor, *kwargs) -> torch.Tensor:
		"""
        @param z: Anchor
        @param z_aug: Mirror
        @param labels: annotations
        """
		# get per sample sup and unsup loss
		sup_loss = self.scl(z=z, z_aug=z_aug, labels=labels)
		unsup_loss = self.sscl(z=z, z_aug=z_aug)
		
		# label for M-viewed batch with M=2
		labels = labels.repeat(2).to(z.device)
		
		# get the indices of P and  U samples in the multi-viewed batch
		p_ix = torch.where(labels == 1)[0]
		u_ix = torch.where(labels == 0)[0]
		
		# if no positive labeled it is simply SelfSupConLoss
		num_labeled = len(p_ix)
		if num_labeled == 0:
			return torch.mean(unsup_loss)
		
		# compute expected similarity
		# -------------------------
		risk_p = sup_loss[p_ix]
		risk_u = unsup_loss[u_ix]
		
		loss = torch.cat([risk_p, risk_u], dim=0)
		return torch.mean(loss)


class PULoss(nn.Module):
	def __init__(self, prior: float, loss_fn: str):
		super(PULoss, self).__init__()
		if not 0 < prior < 1:
			raise ValueError("The class prior should be in [0, 1]")
		self.prior, self.loss_fn = prior, loss_fn
		self.meta_loss = nn.CrossEntropyLoss()
	
	def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
		"""

		:param logits:
		:param targets:
		:return:
		"""
		# logits: shape [Batch Size \times Num of Classes] - un-normalized raw linear combination (w_i * x_i + b)
		# noinspection PyTypeChecker
		ix_positive = torch.where(targets == 1)[0]
		# noinspection PyTypeChecker
		ix_unlabeled = torch.where(targets == 0)[0]
		
		pos_logits = torch.index_select(input=logits, dim=0, index=ix_positive)
		unlabeled_logits = torch.index_select(input=logits, dim=0, index=ix_unlabeled)
		
		targets_pos = torch.ones(len(ix_positive), dtype=targets.dtype)
		targets_pos_inverse = torch.zeros(len(ix_positive), dtype=targets.dtype)
		targets_unlabeled = torch.zeros(len(ix_unlabeled), dtype=targets.dtype)
		
		# compute empirical estimates
		# R_p+
		loss_positive = self.meta_loss(pos_logits.to(logits.device), targets_pos.to(targets.device)) \
			if ix_positive.nelement() != 0 else 0
		# R_u-
		loss_unlabeled = self.meta_loss(unlabeled_logits.to(logits.device), targets_unlabeled.to(targets.device)) \
			if ix_unlabeled.nelement() != 0 else 0
		# R_p-
		loss_pos_inv = self.meta_loss(pos_logits.to(logits.device), targets_pos_inverse.to(targets.device)) \
			if ix_positive.nelement() != 0 else 0
		# (1-pi) Rn- = R_u- - prior * R_p-
		loss_negative = loss_unlabeled - self.prior * loss_pos_inv
		
		if self.prior == 0:
			prior = ix_positive.nelement() / (ix_positive.nelement() + ix_unlabeled.nelement())
			# i.e. fully supervised equivalent to PN strategy
			return prior * loss_unlabeled + (1 - prior) * loss_positive
		elif self.loss_fn == 'nnPU':
			return - loss_negative if loss_negative < 0 else self.prior * loss_positive + loss_negative
		elif self.loss_fn == 'uPU':
			return self.prior * loss_positive + loss_negative
		else:
			ValueError('Unsupported Loss')


def compute_inner_pdt_mtx(z: torch.Tensor, z_aug: torch.Tensor, temp: float) -> torch.Tensor:
	"""
	returns a Temp normalized - cross similarity (inner product) scores.
	diagonals are set to 0.
	"""
	z = torch.nn.functional.normalize(z, dim=1)
	z_aug = torch.nn.functional.normalize(z_aug, dim=1)
	
	# calculate similarities block-wise - the resulting vectors have shape (batch_size, batch_size)
	inner_pdt_00 = torch.einsum('nc,mc->nm', z, z) / temp
	inner_pdt_01 = torch.einsum('nc,mc->nm', z, z_aug) / temp
	inner_pdt_10 = torch.t(inner_pdt_01)
	inner_pdt_11 = torch.einsum('nc,mc->nm', z_aug, z_aug) / temp
	
	# concatenate blocks : o/p shape (2*batch_size, 2*batch_size)
	# [ Block 00 ] | [ Block 01 ]
	# [ Block 10 ] | [ Block 11 ]
	inner_pdt_0001 = torch.cat([inner_pdt_00, inner_pdt_01], dim=1)
	inner_pdt_1011 = torch.cat([inner_pdt_10, inner_pdt_11], dim=1)
	inner_pdt_mtx = torch.cat([inner_pdt_0001, inner_pdt_1011], dim=0)
	
	max_inner_pdt, _ = torch.max(inner_pdt_mtx, dim=1, keepdim=True)
	inner_pdt_mtx = inner_pdt_mtx - max_inner_pdt.detach()  # for numerical stability
	
	return inner_pdt_mtx


def compute_sfx_mtx(inner_pdt_mtx: torch.Tensor) -> torch.Tensor:
	"""
	:param inner_pdt_mtx:
	:returns: softmax per row w/o the diagonal
	"""
	# max_inner_pdt, _ = torch.max(inner_pdt_mtx, dim=1, keepdim=True)
	# inner_pdt_mtx = inner_pdt_mtx - max_inner_pdt.detach()  # for numerical stability
	
	sfx_mtx = torch.exp(inner_pdt_mtx)
	# softmax w/o the diagonal entry
	diag_mask = torch.ones_like(inner_pdt_mtx).fill_diagonal_(0)
	sfx_mtx = sfx_mtx * diag_mask
	sfx_mtx /= torch.sum(sfx_mtx, dim=1, keepdim=True)
	
	return sfx_mtx


def get_self_aug_mask(z: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
	""" [i,j] = 1 if x_j is aug of x_i else 0 """
	aug_mask_00 = torch.zeros((z.shape[0], z.shape[0]), device=z.device)
	aug_mask_01 = torch.zeros((z.shape[0], z.shape[0]), device=z.device)
	aug_mask_01.fill_diagonal_(1)
	aug_mask_10 = aug_mask_01
	aug_mask_11 = aug_mask_00
	aug_mask_0001 = torch.cat([aug_mask_00, aug_mask_01], dim=1)
	aug_mask_1011 = torch.cat([aug_mask_10, aug_mask_11], dim=1)
	aug_mask = torch.cat([aug_mask_0001, aug_mask_1011], dim=0)
	
	neg_aug_mask = aug_mask.clone()
	neg_aug_mask[aug_mask == 0] = 1
	neg_aug_mask[aug_mask == 1] = 0
	neg_aug_mask.fill_diagonal_(0)
	
	return aug_mask, neg_aug_mask
