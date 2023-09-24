"""
Return appropriate optimizer and LRS
"""
from typing import Dict
import warnings
import numpy as np
import torch
from torch import optim
from lightly.utils.lars import LARS
# from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torch.optim import Optimizer
# from torch.optim.lr_scheduler import LambdaLR
# import math
# from transformers import get_cosine_schedule_with_warmup


def get_optimizer(
		params,
		optimizer_config: Dict = None
):
	"""
	wrapper to return appropriate optimizer class
	"""
	opt_alg = optimizer_config.get('optimizer', 'SGD')
	optimizer_config = {} if optimizer_config is None else optimizer_config
	if opt_alg == 'SGD':
		return optim.SGD(
			params=params,
			lr=optimizer_config.get('lr0', 1),
			momentum=optimizer_config.get('momentum', 0.9),
			weight_decay=optimizer_config.get('reg', 0.005),
			nesterov=optimizer_config.get('nesterov', True)
		)
	elif opt_alg == 'Adam':
		return optim.Adam(
			params=params,
			lr=optimizer_config.get('lr0', 1),
			betas=optimizer_config.get('betas', (0.5, 0.99)),
			eps=optimizer_config.get('eps', 1e-08),
			weight_decay=optimizer_config.get('reg', 0.005),
			amsgrad=optimizer_config.get('amsgrad', False)
		)
	elif opt_alg == 'AdamW':
		return optim.AdamW(
			params=params,
			lr=optimizer_config.get('lr0', 1),
			betas=optimizer_config.get('betas', (0.9, 0.999)),
			eps=optimizer_config.get('eps', 1e-08),
			weight_decay=optimizer_config.get('reg', 0.005),
			amsgrad=optimizer_config.get('amsgrad', False)
		)
	elif opt_alg == 'LARS':
		return LARS(
			params=params,
			lr=optimizer_config.get('lr0', 1),
			momentum=optimizer_config.get('momentum', 0.9),
			weight_decay=optimizer_config.get('reg', 0),
			nesterov=optimizer_config.get('nesterov', True)
		)
	else:
		raise NotImplementedError


def get_scheduler(
		optimizer,
		lrs_config: Dict = None
):
	"""
	Get appropriate Learning Rate Scheduler
	"""
	lrs = lrs_config.get('lrs')
	if lrs == 'step':
		return optim.lr_scheduler.StepLR(
			optimizer=optimizer,
			step_size=lrs_config.get('step_size'),
			gamma=lrs_config.get('gamma')
		)
	elif lrs == 'multi_step':
		return optim.lr_scheduler.MultiStepLR(
			optimizer=optimizer,
			milestones=lrs_config.get('milestones'),
			gamma=lrs_config.get('gamma')
		)
	elif lrs == 'cosine':
		# return get_cosine_schedule_with_warmup(
		# 	optimizer=optimizer,
		# 	num_warmup_steps=lrs_config.get('warmup'),
		# 	num_training_steps=lrs_config.get('T_max'),
		# )
		return optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
		                                            T_max=lrs_config.get('T_max'),
		                                            eta_min=lrs_config.get('eta_min', 0))
	# return LinearWarmupCosineAnnealingLR(
	# 	optimizer=optimizer,
	# 	warmup_start_lr=0,
	# 	warmup_epochs=lrs_config.get('warmup', 10),
	# 	max_epochs=lrs_config.get('T_max'),
	# )
	# return CosineWarmupScheduler(
	# 	optimizer=optimizer,
	# 	warmup_epochs=lrs_config.get('warmup', 10),
	# 	max_epochs=lrs_config.get('T_max'),
	# 	start_value=lrs_config.get('lr0', 1),
	# 	end_value=lrs_config.get('eta_min', 0.001),
	# 	verbose=True
	# )
	else:
		return None


def get_cosine_schedule_with_warmup(
		optimizer: Optimizer,
		num_warmup_steps: int,
		num_training_steps: int,
		num_cycles: float = 0.5,
		last_epoch: int = -1
):
	"""
	:param optimizer:
	:param num_warmup_steps:
	:param num_training_steps:
	:param num_cycles:
	:param last_epoch:
	:return:
	"""

	def lr_lambda(current_step):
		"""
		:param current_step:
		:return:
		"""
		if current_step < num_warmup_steps:
			return float(current_step + 1) / float(max(1, num_warmup_steps))
		progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
		return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))

	return LambdaLR(optimizer, lr_lambda, last_epoch)


class CosineWarmupScheduler(torch.optim.lr_scheduler.LambdaLR):
	"""
    Cosine warmup scheduler for learning rate.
    Args:
        optimizer:
            Optimizer object to schedule the learning rate.
        warmup_epochs:
            Number of warmup epochs or steps.
        max_epochs:
            Total number of training epochs or steps.
        last_epoch:
            The index of last epoch or step. Default: -1
        start_value:
            Starting learning rate scale. Default: 1.0
        end_value:
            Target learning rate scale. Default: 0.001
        verbose:
            If True, prints a message to stdout for each update. Default: False.
    """

	def __init__(
			self,
			optimizer: torch.optim.Optimizer,
			warmup_epochs: int,
			max_epochs: int,
			last_epoch: int = -1,
			start_value: float = 1.0,
			end_value: float = 0.001,
			verbose: bool = True,
	) -> None:
		self.warmup_epochs = warmup_epochs
		self.max_epochs = max_epochs
		self.start_value = start_value
		self.end_value = end_value
		super().__init__(
			optimizer=optimizer,
			lr_lambda=self.scale_lr,
			last_epoch=last_epoch,
			verbose=verbose,
		)

	def cosine_schedule(self, step: int, max_steps: int) -> float:
		"""
		Use cosine decay to gradually modify start_value to reach target end_value during iterations.
		@param step
		@param max_steps
		Returns: Cosine decay value.
		"""
		if step < 0:
			raise ValueError("Current step number can't be negative")
		if max_steps < 1:
			raise ValueError("Total step number must be >= 1")
		if step > max_steps:
			warnings.warn(
				f"Current step number {step} exceeds max_steps {max_steps}.",
				category=RuntimeWarning,
			)
		if max_steps == 1:
			# Avoid division by zero
			decay = self.end_value
		elif step == max_steps:
			# Special case for Pytorch Lightning which updates LR scheduler also for epoch
			# after last training epoch.
			decay = self.end_value
		else:
			decay = (
					self.end_value
					- (self.end_value - self.start_value)
					* (np.cos(np.pi * step / (max_steps - 1)) + 1)
					/ 2
			)
		return decay / self.start_value

	def scale_lr(self, epoch: int) -> float:
		"""
        Scale learning rate according to the current epoch number.
        Args:
            epoch:
                Current epoch number.
        Returns:
            Scaled learning rate scaling:
        """
		if epoch < self.warmup_epochs:
			decay = (epoch + 1) / self.warmup_epochs
		else:
			decay = self.cosine_schedule(
				step=epoch - self.warmup_epochs,
				max_steps=self.max_epochs - self.warmup_epochs
			)
		return decay
