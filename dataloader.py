"""
Data Loader Module
"""
import os
from typing import Dict, List

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import lightly.data as data
from lightly.transforms import SimCLRTransform  # we will stick with this for all the experiments

root_dir = os.path.join(os.path.dirname(__file__), './data/')


class DataManager:
	"""
    @param data_config = config
    """
	
	def __init__(
			self,
			data_config: Dict,
			gpu_strategy: str = "auto"
	):
		# config
		self.data_config = data_config
		self.data_set = self.data_config.get('data_set')
		self.num_classes = self.data_config.get('num_classes')
		self.train_batch_size = self.data_config.get('train_batch_size', 256)
		self.test_batch_size = self.data_config.get('test_batch_size', 1000)
		
		# DDP -- divide batch into gpus
		if gpu_strategy == "ddp":
			print("DDP ~ so updating the batch sizes accordingly to split data into gpus")
			n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
			self.train_batch_size //= n_gpus
			self.test_batch_size //= n_gpus
		self.num_worker = self.data_config.get('num_worker', 1)
		
		# initialize attributes specific to dataset
		self.num_channels, self.height, self.width = None, None, None
		self.tr_dataset, self.te_dataset = None, None
		self.sv_transform, self.mv_transform, self.basic_transform = None, None, None
	
	def get_transforms(self):
		"""
		:return:
		"""
		if self.data_set == 'cifar10':
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
	
	def get_contrastive_learning_data(self) -> [DataLoader, DataLoader, DataLoader]:
		"""
        Returns:
        	train and test dataset
        """
		if self.data_set == 'cifar10':
			# update dataset attributes
			self.num_channels, self.height, self.width = 3, 32, 32
			dataset_train_ssl = datasets.CIFAR10(
				"datasets/cifar10",
				download=True
			)
			dataset_train_val = datasets.CIFAR10(
				"datasets/cifar10",
				download=True
			)
			dataset_test = datasets.CIFAR10(
				"datasets/cifar10",
				download=True,
				train=False
			)
		else:
			raise NotImplementedError
		# define validation and test transform
		self.mv_transform, self.basic_transform = self.get_transforms()
		dataset_train_ssl.transform = self.mv_transform
		dataset_train_val.transform = self.basic_transform
		dataset_test.transform = self.basic_transform
		dataset_train_ssl = data.LightlyDataset.from_torch_dataset(dataset_train_ssl)
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
		return dataloader_train_mv, dataloader_train_sv, dataloader_test
