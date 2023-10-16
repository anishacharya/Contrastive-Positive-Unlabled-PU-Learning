"""
Return appropriate optimizer and LRS
"""
from typing import Dict
from torch import optim
from lightly.utils.lars import LARS
import transformers
import torch.nn as nn
import torch.nn.functional as F


# ---- Optimizer ------
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


# ---- Scheduler ------
def get_scheduler(
		optimizer,
		lrs_config: Dict = None,
		verbose: bool = False
):
	"""
	Get appropriate Learning Rate Scheduler
	"""
	lrs = lrs_config.get('lrs')
	if lrs == 'step':
		return optim.lr_scheduler.StepLR(
			optimizer=optimizer,
			step_size=lrs_config.get('step_size'),
			gamma=lrs_config.get('gamma'),
			verbose=verbose
		)
	elif lrs == 'multi_step':
		return optim.lr_scheduler.MultiStepLR(
			optimizer=optimizer,
			milestones=lrs_config.get('milestones'),
			gamma=lrs_config.get('gamma'),
			verbose=verbose
		)
	elif lrs == 'cosine':
		return optim.lr_scheduler.CosineAnnealingLR(
			optimizer=optimizer,
			T_max=lrs_config.get('T_max'),
			eta_min=lrs_config.get('eta_min', 0),
			verbose=verbose
		)
	elif lrs == 'cosine_warmup':
		return transformers.get_cosine_schedule_with_warmup(
			optimizer=optimizer,
			num_warmup_steps=lrs_config.get('warmup', 10),
			num_training_steps=lrs_config.get('T_max'),
		)
	else:
		return None


# ---- Models ------
def conv3x3(in_planes, out_planes, stride=1):
	"""

	:param in_planes:
	:param out_planes:
	:param stride:
	:return:
	"""
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
	expansion = 1
	
	def __init__(self, in_planes, planes, stride=1):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(in_planes, planes, stride)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = nn.BatchNorm2d(planes)
		
		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != self.expansion * planes:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(self.expansion * planes)
			)
	
	def forward(self, x):
		"""

		:param x:
		:return:
		"""
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.bn2(self.conv2(out))
		out += self.shortcut(x)
		out = F.relu(out)
		return out


class Bottleneck(nn.Module):
	expansion = 4
	
	def __init__(self, in_planes, planes, stride=1):
		super(Bottleneck, self).__init__()
		self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(self.expansion * planes)
		
		self.shortcut = nn.Sequential()
		if stride != 1 or in_planes != self.expansion * planes:
			self.shortcut = nn.Sequential(
				nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(self.expansion * planes)
			)
	
	def forward(self, x):
		"""

		:param x:
		:return:
		"""
		out = F.relu(self.bn1(self.conv1(x)))
		out = F.relu(self.bn2(self.conv2(out)))
		out = self.bn3(self.conv3(out))
		out += self.shortcut(x)
		out = F.relu(out)
		return out


class CIFARResNet(nn.Module):
	def __init__(self, block, num_blocks):
		super(CIFARResNet, self).__init__()
		self.in_planes = 64
		
		self.conv1 = conv3x3(3, 64)
		self.bn1 = nn.BatchNorm2d(64)
		self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
		self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
		self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
		self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
	
	# self.linear = nn.Linear(512*block.expansion, num_classes)
	
	def _make_layer(self, block, planes, num_blocks, stride):
		strides = [stride] + [1] * (num_blocks - 1)
		layers = []
		for stride in strides:
			layers.append(block(self.in_planes, planes, stride))
			self.in_planes = planes * block.expansion
		return nn.Sequential(*layers)
	
	def forward(self, x):
		"""

		:param x:
		:return:
		"""
		out = F.relu(self.bn1(self.conv1(x)))
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = F.avg_pool2d(out, 4)
		# out = out.view(out.size(0), -1)
		# out = self.linear(out)
		return out


def cifarresnet18():
	"""

	:return:
	"""
	return CIFARResNet(BasicBlock, [2, 2, 2, 2])


def cifarresnet34():
	"""

	:return:
	"""
	return CIFARResNet(BasicBlock, [3, 4, 6, 3])


def cifarresnet50():
	"""

	:return:
	"""
	return CIFARResNet(Bottleneck, [3, 4, 6, 3])


def cifarresnet101():
	"""

	:return:
	"""
	return CIFARResNet(Bottleneck, [3, 4, 23, 3])


def cifarresnet152():
	"""

	:return:
	"""
	return CIFARResNet(Bottleneck, [3, 8, 36, 3])


class CIFARCNN(nn.Module):
	""" Ref: https://openreview.net/pdf?id=NH29920YEmj. Code in .zip submission """
	
	def __init__(self):
		super(CIFARCNN, self).__init__()
		
		self.conv1 = nn.Conv2d(3, 96, 3)
		self.conv2 = nn.Conv2d(96, 96, 3, stride=2)
		self.conv3 = nn.Conv2d(96, 192, 1)
		self.conv4 = nn.Conv2d(192, 10, 1)
		
		self.fc1 = nn.Linear(1960, 1000)
		self.fc2 = nn.Linear(1000, 1000)
	
	# self.fc3 = nn.Linear(1000, num_classes)
	
	def forward(self, x):
		"""

		:param x:
		:return:
		"""
		h = self.conv1(x)
		h = F.relu(h)
		h = self.conv2(h)
		h = F.relu(h)
		h = self.conv3(h)
		h = F.relu(h)
		h = self.conv4(h)
		h = F.relu(h)
		
		h = h.view(h.size(0), -1)
		h = self.fc1(h)
		h = F.relu(h)
		h = self.fc2(h)
		# h = F.relu(h)
		# h = self.fc3(h)
		return h


class STLCNN(nn.Module):
	""" Ref: https://openreview.net/pdf?id=NH29920YEmj. Code in .zip submission """
	
	def __init__(self):
		super(STLCNN, self).__init__()
		
		self.relu = nn.ReLU()
		self.conv1 = nn.Conv2d(3, 6, 3)
		self.conv2 = nn.Conv2d(6, 6, 3)
		self.mp = nn.MaxPool2d(2, 2)
		self.conv3 = nn.Conv2d(6, 16, 5)
		self.conv4 = nn.Conv2d(16, 32, 5)
		
		self.fc1 = nn.Linear(32 * 8 * 8, 120)
		self.fc2 = nn.Linear(120, 84)
		
		self.layer1 = nn.Sequential(self.conv1, self.relu, self.mp)
		self.layer2 = nn.Sequential(self.conv2, self.relu)
		self.layer3 = nn.Sequential(self.conv3, self.relu, self.mp)
		self.layer4 = nn.Sequential(self.conv4, self.relu, self.mp)
		
		self.conv_layers = nn.ModuleList([self.layer1, self.layer2, self.layer3, self.layer4])
		self.fc_layers = nn.Sequential(self.fc1, self.relu, self.fc2)
	
	def forward(self, x):
		"""

		:param x:
		:return:
		"""
		h = x
		for i, layer_module in enumerate(self.conv_layers):
			h = layer_module(h)
		h = h.view(h.size(0), -1)
		h = self.fc_layers(h)
		return h


class FMNISTLeNet(nn.Module):
	""" Ref: https://openreview.net/pdf?id=NH29920YEmj. Code in .zip submission """
	
	def __init__(self):
		super(FMNISTLeNet, self).__init__()
		
		self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
		self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
		self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
		self.bn_conv1 = nn.BatchNorm2d(6)
		self.bn_conv2 = nn.BatchNorm2d(16)
		self.mp = nn.MaxPool2d(2)
		self.relu = nn.ReLU()
		
		self.fc1 = nn.Linear(120, 84)
		self.bn_fc1 = nn.BatchNorm1d(84)
		
		self.layer1 = nn.Sequential(self.conv1, self.mp, self.relu)
		self.layer2 = nn.Sequential(self.conv2, self.mp, self.relu)
		self.layer3 = nn.Sequential(self.conv3, self.relu)
		
		# self.conv_layers = nn.ModuleList([self.layer1, self.layer2, self.layer3])
		self.fc_layers = nn.Sequential(self.fc1, self.bn_fc1)
	
	def forward(self, x):
		"""

		:param x:
		:return:
		"""
		h = x
		# for i, layer_module in enumerate(self.conv_layers):
		#     h = layer_module(h)
		h = self.layer1(h)
		h = self.layer2(h)
		h = self.layer3(h)
		
		h = h.view(h.size(0), -1)
		
		h = self.fc_layers(h)
		
		return h
