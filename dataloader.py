"""
Data Loader Module
"""
import os
from typing import Dict, List
from sklearn.utils import shuffle
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import lightly.data as data
from lightly.transforms import SimCLRTransform

root_dir = os.path.join(os.path.dirname(__file__), './data/')


class DataManager:
	"""
    Defines all data related things.
    """
	
	def __init__(
			self,
			data_config: Dict,
			gpu_strategy: str = "auto"
	):
		# ---- config ----
		self.data_config = data_config
		self.data_set = self.data_config.get('data_set')
		self.train_batch_size = self.data_config.get('train_batch_size', 256)
		self.test_batch_size = self.data_config.get('test_batch_size', 1000)
		#  attributes specific to dataset
		self.setting = self.data_config.get('setting', 'supervised')
		self.num_labeled = self.data_config.get('num_labeled', None)
		self.num_unlabeled = self.data_config.get('num_unlabeled', None)
		self.dataset_prior = self.data_config.get('dataset_prior', None)
		self.neg_classes, self.pos_classes = None, None  # PU specific
		self.num_classes = None
		self.num_channels, self.height, self.width = None, None, None
		self.tr_dataset, self.te_dataset = None, None
		self.sv_transform, self.mv_transform, self.basic_transform = None, None, None
		
		# ---- DDP -- divide batch into gpus -----
		if gpu_strategy == "ddp":
			print("DDP ~ so updating the batch sizes accordingly to split data into gpus")
			n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
			self.train_batch_size //= n_gpus
			self.test_batch_size //= n_gpus
		self.num_worker = self.data_config.get('num_worker', 1)
	
	def get_transforms(self):
		"""
		:return:
		"""
		if self.data_set in ['cifar10.dog_cat']:
			# update dataset attributes
			mean, std = [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
			model_ip_shape = 32
			mv_transform = SimCLRTransform(
				input_size=model_ip_shape,
				cj_strength=0.5,
				gaussian_blur=0.0
			)
		else:
			raise NotImplementedError
		
		basic_transform = self.BasicTransform(
			mean=mean,
			std=std,
			input_shape=model_ip_shape
		)
		return mv_transform, basic_transform
	
	class BasicTransform:
		"""
		Basic Transform
		"""
		
		def __init__(
				self,
				mean: List,
				std: List,
				input_shape: int
		):
			self.mean, self.std = mean, std
			self.transform = transforms.Compose([
				transforms.Resize(input_shape),
				transforms.ToTensor(),
				transforms.Normalize(mean=self.mean, std=self.std),
			])
		
		def __call__(self, x):
			return self.transform(x)
	
	def get_data(self) -> [DataLoader, DataLoader, DataLoader]:
		"""
        Returns:
        	train and test dataset
        """
		binary_class_mapping = {
			# Cifar10 Class Mapping
			# 0:'airplane', 1:'automobile', 2:'bird', 3:'cat', 4:'deer',
			# 5:'dog', 6:'frog', 7:'horse', 8:'ship', 9:'truck'
			'cifar10.dog_cat': {
				'pos_classes': [5],
				'neg_classes': [3]
			},
			'cifar10.1': {
				'pos_classes': [0, 1, 8, 9],
				'neg_classes': [2, 3, 4, 5, 6, 7]
			},
			'cifar10.2': {
				'pos_classes': [2, 3, 4, 5, 6, 7],
				'neg_classes': [0, 1, 8, 9]
			},
		}
		dataset_map = {
			"binary_cifar": BinaryCIFAR10,
		}
		if self.data_set in ['cifar10.dog_cat', 'cifar.1', 'cifar.2']:
			# get attributes
			self.num_classes = 2
			self.num_channels, self.height, self.width = 3, 32, 32
			self.pos_classes = binary_class_mapping[self.data_set]['pos_classes']
			self.neg_classes = binary_class_mapping[self.data_set]['neg_classes']
			root_dataset = 'binary_cifar'
		
		else:
			raise NotImplementedError
		
		# obtain datasets
		dataset_train_ssl = dataset_map[root_dataset](
			pos_class=self.pos_classes,
			neg_class=self.neg_classes,
			setting=self.setting,
			num_labeled=self.num_labeled,
			num_unlabeled=self.num_unlabeled,
			prior=self.dataset_prior
		)
		dataset_train_sv = dataset_map[root_dataset](
			pos_class=self.pos_classes,
			neg_class=self.neg_classes,
			setting=self.setting,
			num_labeled=self.num_labeled,
			num_unlabeled=self.num_unlabeled,
			prior=self.dataset_prior
		)
		dataset_train_val = dataset_map[root_dataset](
			pos_class=self.pos_classes,
			neg_class=self.neg_classes,
			setting='supervised',
		)
		dataset_test = dataset_map[root_dataset](
			pos_class=self.pos_classes,
			neg_class=self.neg_classes,
			setting='supervised',
			train=False,
		)
		
		# define validation and test transform
		self.mv_transform, self.basic_transform = self.get_transforms()
		
		dataset_train_ssl.transform = self.mv_transform
		dataset_train_sv.transform = self.basic_transform  # for Linear Probing / FineTuning
		dataset_train_val.transform = self.basic_transform  # for kNN
		dataset_test.transform = self.basic_transform
		
		dataset_train_ssl = data.LightlyDataset.from_torch_dataset(dataset_train_ssl)
		dataset_train_sv = data.LightlyDataset.from_torch_dataset(dataset_train_sv)
		dataset_train_val = data.LightlyDataset.from_torch_dataset(dataset_train_val)
		dataset_test = data.LightlyDataset.from_torch_dataset(dataset_test)
		
		dataloader_train_mv = DataLoader(
			dataset_train_ssl,
			batch_size=self.train_batch_size,
			shuffle=True,
			drop_last=True,
			num_workers=self.num_worker,
		)
		dataloader_train_sv = DataLoader(
			dataset_train_sv,
			batch_size=self.train_batch_size,
			shuffle=True,
			drop_last=True,
			num_workers=self.num_worker,
		)
		dataloader_train_val = DataLoader(
			dataset_train_val,
			batch_size=self.train_batch_size,
			shuffle=False,
			num_workers=self.num_worker
		)
		dataloader_test = DataLoader(
			dataset_test,
			batch_size=self.test_batch_size,
			shuffle=False,
			num_workers=self.num_worker
		)
		return dataloader_train_mv, dataloader_train_sv, dataloader_train_val, dataloader_test


class BinaryCIFAR10(datasets.CIFAR10):
	"""
		Binarize CIFAR10
	"""
	
	def __init__(
			self,
			pos_class: List,
			neg_class: List = None,
			root=root_dir,
			train: bool = True,
			download: bool = True,
			setting: str = None,
			num_labeled: int = None,
			num_unlabeled: int = None,
			prior: float = None
	):
		super().__init__(root=root, train=train, download=download)
		self.data, self.targets = np.array(self.data), np.array(self.targets)
		self.data, self.targets = binarize_dataset(
			features=self.data,
			targets=self.targets,
			pos_class=pos_class,
			neg_class=neg_class,
			setting=setting,
			num_labeled=num_labeled,
			num_unlabeled=num_unlabeled,
			prior=prior
		)


def binarize_dataset(
		features: np.array,
		targets: np.array,
		pos_class: List,
		neg_class: List = None,
		setting: str = None,
		num_labeled: int = 0,
		num_unlabeled: int = None,
		prior: float = None
) -> [np.array, np.array, Dict]:
	"""
		Binarize and flip labels as needed to obtain
		Case control PU, Single dataset PU or Binary PN data
	"""
	p_data_idx = np.where(np.isin(targets, pos_class))[0]
	n_data_idx = np.where(np.isin(targets, neg_class) if neg_class else
	                      np.isin(targets, pos_class, invert=True))[0]
	
	if setting in ['pu_case_control', 'pu_single_data']:
		"""
        'pu_case_control': PU setting Xp ~ p(x | y=1), Xu ~ p(x)
        'pu_single_data' : X ~ p(x), some P samples are labeled based on propensity score
        """
		if num_labeled == 0:
			targets = np.zeros(len(features), dtype=targets.dtype)
		else:
			# Obtain P data
			# p_ix = np.random.choice(a=p_data_idx, size=num_labeled, replace=True)
			# # balanced P data ~ equal data from each class in P
			num_pos_labeled_per_cls = int(num_labeled / len(pos_class))
			p_ix = []
			for cls in pos_class:
				p_cls = np.where(np.isin(targets, cls))[0]
				p_ix.extend(np.random.choice(a=p_cls, size=num_pos_labeled_per_cls, replace=False))
			p_data = features[p_ix]
			p_labels = np.ones(len(p_data), dtype=targets.dtype)
			
			# Obtain U data
			if setting == 'pu_case_control':
				if num_unlabeled and prior:
					n_pu = int(prior * num_unlabeled)
					n_nu = num_unlabeled - n_pu
					pu_ix = np.random.choice(a=p_data_idx, size=n_pu,
					                         replace=False if n_pu <= len(p_data_idx) else True)
					nu_ix = np.random.choice(a=n_data_idx, size=n_nu,
					                         replace=False if n_nu <= len(p_data_idx) else True)
					u_ix = np.concatenate((pu_ix, nu_ix), axis=0)
				else:
					u_ix = np.concatenate((p_data_idx, n_data_idx), axis=0)
				u_data = features[u_ix]
				u_labels = np.zeros(len(u_data), dtype=targets.dtype)
			else:
				remaining_p_ix = np.setdiff1d(ar1=p_data_idx, ar2=p_ix)
				u_ix = np.concatenate((remaining_p_ix, n_data_idx), axis=0)
				u_data = features[u_ix]
				u_labels = np.zeros(len(u_data), dtype=targets.dtype)
			# create PU data
			features = np.concatenate((p_data, u_data), axis=0)
			targets = np.concatenate((p_labels, u_labels), axis=0)
	elif setting == 'unsupervised':
		"""
		Fully Unsupervised setting X
		"""
		p_data = features[p_data_idx]
		n_data = features[n_data_idx]
		features = np.concatenate((p_data, n_data), axis=0)
		targets = np.zeros(len(features), dtype=targets.dtype)
	elif setting == 'supervised':
		"""
        standard binary (PN) setting Xp ~ p(x | y=1) , Xn ~ p(x | y=0)
        """
		p_data = features[p_data_idx]
		n_data = features[n_data_idx]
		p_labels = np.ones(len(p_data), dtype=targets.dtype)
		n_labels = np.zeros(len(n_data), dtype=targets.dtype)
		features = np.concatenate((p_data, n_data), axis=0)
		targets = np.concatenate((p_labels, n_labels), axis=0)
	else:
		raise NotImplementedError
	
	features, targets = shuffle(features, targets, random_state=0)
	return features, targets
