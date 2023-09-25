""" Defines different losses for training """
import torch.nn as nn
import torch
from typing import Dict


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
		return SelfSupConLoss(temperature=temp, reduction='mean')
	elif loss_fn == 'puCL':
		return PUConLoss(temperature=temp)
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
		self.cross_entropy = nn.CrossEntropyLoss(reduction=reduction)
	
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
		# [ Block 01 ] | [ Block 00 ]
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
		diag_mask = torch.ones_like(inner_pdt_mtx, device=z.device, dtype=torch.bool).fill_diagonal_(0)
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
		if self.reduction == 'mean':
			loss = torch.mean(loss)
		return loss


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
