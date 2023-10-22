"""
Defines contrastive framework SimCLR: https://arxiv.org/abs/2002.05709
"""
from typing import Dict, Optional, List

import torch
import torch.nn.functional as F
from lightly.models.modules import heads
from lightly.utils.benchmarking.knn import knn_predict
from pytorch_lightning import LightningModule
from torch import Tensor
from torch import distributed as torch_dist
from torch.utils.data import DataLoader

from losses import get_loss
from utils import (
	get_optimizer,
	get_scheduler,
	cifarresnet18,
	cifarresnet50,
	CIFARCNN,
)


class BaseFramework(LightningModule):
	def __init__(
			self,
			framework_config: Dict,
			training_config: Dict,
			data_config: Dict,
			val_dataloader: DataLoader,
			num_classes: int,
			gather_distributed: bool = False
	):
		super().__init__()
		self.framework_config = framework_config
		self.training_config = training_config
		self.data_config = data_config
		self.val_dataloader = val_dataloader
		self.num_classes = num_classes
		self.gather_distributed = gather_distributed
		
		self.temp = self.framework_config.get("temp", 0.5)
		# sanity checks -----
		if abs(self.temp) < 1e-8:
			raise ValueError(
				"Illegal temperature: abs({}) < 1e-8".format(self.temp)
			)
		if gather_distributed and not torch_dist.is_available():
			raise ValueError(
				"gather_distributed is True but torch.distributed is not available. "
				"Please set gather_distributed=False or install a torch version with "
				"distributed support."
			)
		
		# --- Load Models
		(self.backbone, self.projection_head, self.feat_dim,
		 self.proj_hidden_dim, self.proj_dim, self.proj_num_layers) = None, None, None, None, None, None
		self.init_model(encoder_arch=self.framework_config.get('encoder_arch', 'cifar-resnet18'))
		
		# Get objective
		self.criterion = get_loss(framework_config=framework_config)
		
		# kNN validation
		self._train_features: Optional[Tensor] = None
		self._train_targets: Optional[Tensor] = None
		self._val_predicted_labels: List[Tensor] = []
		self._val_targets: List[Tensor] = []
		self.knn_k = self.framework_config.get("knn_k", 200)
		self.knn_t = self.framework_config.get("knn_t", 0.1)
		self.max_accuracy = 0.0
	
	def configure_optimizers(self):
		params = (
				list(self.backbone.parameters()) +
				list(self.projection_head.parameters())
		)
		opt = get_optimizer(
			params=params,
			optimizer_config=self.training_config
		)
		scheduler = get_scheduler(
			optimizer=opt,
			lrs_config=self.training_config,
			verbose=False
		)
		# return {"optimizer": opt, "lr_schedulers": scheduler}
		return [opt], [{"scheduler": scheduler, "interval": "epoch"}]
	
	def init_model(self, encoder_arch: str):
		"""
		initialize encoder and projection n/w
		"""
		# encoder
		if encoder_arch == 'cifar-resnet18':
			self.backbone = cifarresnet18()
			self.feat_dim = 512
			self.proj_hidden_dim = 512
			self.proj_dim = 128
			self.proj_num_layers = 2
		
		elif encoder_arch == 'cifar-resnet50':
			self.backbone = cifarresnet50()
			self.feat_dim = 2048
			self.proj_hidden_dim = 512
			self.proj_dim = 128
			self.proj_num_layers = 2
		
		elif encoder_arch == 'cifar-cnn':
			self.backbone = CIFARCNN()
			self.feat_dim = 1000
			self.proj_hidden_dim = 512
			self.proj_dim = 128
			self.proj_num_layers = 2
		
		else:
			raise NotImplementedError
		
		# projection
		# an MLP with num_layers layers, i/p is encoder o/p
		self.projection_head = heads.SimCLRProjectionHead(
			input_dim=self.feat_dim,
			hidden_dim=self.framework_config.get('proj_hidden_dim', 512),
			output_dim=self.framework_config.get('proj_dim', 128),
			num_layers=self.framework_config.get('proj_num_layers', 2),
		)
	
	def extract_features(self, dataloader: DataLoader) -> [torch.Tensor, torch.Tensor]:
		"""
		Given a dataloader return the extracted features and labels
		:param dataloader:
		:return:
		"""
		features = []
		labels = []
		with torch.no_grad():
			for mini_batch in dataloader:
				img, target, _ = mini_batch
				img = img.to(self.device)
				target = target.to(self.device)
				feature = self.backbone(img).squeeze()
				feature = F.normalize(feature, dim=1)
				features.append(feature)
				labels.append(target)
		extracted_features = torch.cat(features, dim=0).t().contiguous()
		extracted_labels = torch.cat(labels, dim=0).t().contiguous()
		return extracted_features, extracted_labels
	
	def on_validation_epoch_start(self) -> None:
		self._train_features, self._train_targets = self.extract_features(dataloader=self.val_dataloader)
	
	def validation_step(self, batch, batch_idx: int) -> None:
		# we can only do kNN predictions once we have a feature bank
		if self._train_features is not None and self._train_targets is not None:
			images, targets, _ = batch
			feature = self.backbone(images).squeeze()
			feature = F.normalize(feature, dim=1)
			predicted_labels = knn_predict(
				feature,
				self._train_features,
				self._train_targets,
				self.num_classes,
				self.knn_k,
				self.knn_t,
			)
			self._val_predicted_labels.append(predicted_labels.cpu())
			self._val_targets.append(targets.cpu())
	
	def on_validation_epoch_end(self) -> None:
		if self._val_predicted_labels and self._val_targets:
			predicted_labels = torch.cat(self._val_predicted_labels, dim=0)
			targets = torch.cat(self._val_targets, dim=0)
			top1 = (predicted_labels[:, 0] == targets).float().sum()
			acc = top1 / len(targets)
			if acc > self.max_accuracy:
				self.max_accuracy = acc.item()
			self.log(
				"val_accuracy",
				acc * 100.0,
				prog_bar=True,
				sync_dist=True
			)
			self.log(
				"max_accuracy",
				self.max_accuracy * 100.0,
				prog_bar=True,
				sync_dist=True
			)
		self._val_predicted_labels.clear()
		self._val_targets.clear()


class SimCLR(BaseFramework):
	"""
	SimCLR: https://arxiv.org/abs/2002.05709
	"""
	
	def __init__(
			self,
			framework_config: Dict,
			training_config: Dict,
			data_config: Dict,
			val_dataloader: DataLoader,
			num_classes: int,
			gather_distributed: bool = False,
	):
		super().__init__(
			framework_config=framework_config,
			training_config=training_config,
			data_config=data_config,
			val_dataloader=val_dataloader,
			num_classes=num_classes,
			gather_distributed=gather_distributed,
		)
		self.projection_head = heads.SimCLRProjectionHead(
			input_dim=self.feat_dim,
			hidden_dim=512,
			output_dim=128,
			num_layers=2
		)
	
	def forward(self, x) -> torch.Tensor:
		"""
		:param x: feature input
		:return: representation = projection o encoder (x)
		"""
		x = self.backbone(x).flatten(start_dim=1)
		z = self.projection_head(x)
		return z
	
	def training_step(self, batch, batch_index):
		"""
		:param batch:
		:param batch_index:
		:return:
		"""
		(x0, x1), labels, _ = batch
		z0 = self.forward(x0)
		z1 = self.forward(x1)
		loss = self.criterion(z0, z1, labels)
		self.log(
			"train-loss",
			loss,
			on_step=True,
			on_epoch=True,
			prog_bar=True
		)
		return loss
