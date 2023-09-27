"""
Return appropriate optimizer and LRS
"""
from typing import Dict
from torch import optim
from lightly.utils.lars import LARS
import transformers


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
		return optim.lr_scheduler.CosineAnnealingLR(
			optimizer=optimizer,
			T_max=lrs_config.get('T_max'),
			eta_min=lrs_config.get('eta_min', 0)
		)
	elif lrs == 'cosine_warmup':
		return transformers.get_cosine_schedule_with_warmup(
			optimizer=optimizer,
			num_warmup_steps=lrs_config.get('warmup', 10),
			num_training_steps=lrs_config.get('T_max')
		)
	else:
		return None

