""" Attach a Linear Head """
from typing import Dict, Tuple
from lightly.models.utils import activate_requires_grad, deactivate_requires_grad
from lightly.utils.benchmarking.topk import mean_topk_accuracy
from pytorch_lightning import LightningModule
import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss, Linear, Module
from utils import get_optimizer, get_scheduler
import torch.nn.functional as F


# class LinearClassificationHead(LightningModule):
# 	def __init__(
# 			self,
# 			model: Module,
# 			training_config: Dict,
# 			feature_dim: int,
# 			num_classes: int,
# 			freeze_model: bool = True,
# 	) -> None:
# 		super().__init__()
# 		self.model = model
# 		self.feature_dim = feature_dim
# 		self.num_classes = num_classes
# 		self.training_config = training_config
# 		self.freeze_model = freeze_model
#
# 		self.classification_head = Linear(
# 			feature_dim,
# 			num_classes
# 		)
# 		self.criterion = CrossEntropyLoss()
# 		self.max_accuracy = 0.0
#
# 	def training_step(self, batch, batch_idx) -> Tensor:
# 		images, targets = batch[0], batch[1]
# 		predictions = self.classification_head(images)
# 		loss = self.criterion(predictions, targets)
# 		# Convert logits to predicted classes
# 		_, predicted_classes = torch.max(predictions, 1)
# 		# Calculate correct predictions
# 		correct_predictions = (predicted_classes == targets).sum().item()
# 		# Calculate accuracy
# 		accuracy = correct_predictions / targets.size(0)
# 		batch_size = len(targets)
# 		self.log(
# 			"train_loss",
# 			loss,
# 			prog_bar=True,
# 			sync_dist=True,
# 			batch_size=batch_size
# 		)
# 		self.log(
# 			"train_acc",
# 			accuracy,
# 			prog_bar=True,
# 			sync_dist=True,
# 			batch_size=batch_size
# 		)
# 		return loss
#
# 	def forward(self, x: Tensor) -> Tensor:
# 		features = self.model.backbone(x).flatten(start_dim=1)
# 		features = F.relu(features)
# 		logits = self.classification_head(features)
# 		return logits
#
# 	def configure_optimizers(self):
# 		parameters = list(self.classification_head.parameters())
# 		if not self.freeze_model:
# 			print("Encoder not Frozen")
# 			# FineTuning backprop through entire model
# 			parameters += self.model.backbone.parameters()
# 		optimizer = get_optimizer(
# 			params=parameters,
# 			optimizer_config=self.training_config
# 		)
# 		scheduler = get_scheduler(
# 			optimizer=optimizer,
# 			lrs_config=self.training_config
# 		)
# 		return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
#
# 	def on_fit_start(self) -> None:
# 		# Freeze model weights.
# 		if self.freeze_model:
# 			deactivate_requires_grad(model=self.model)
#
# 	def on_fit_end(self) -> None:
# 		# Unfreeze model weights.
# 		if self.freeze_model:
# 			activate_requires_grad(model=self.model)


class LinearClassificationHead(LightningModule):
	def __init__(
			self,
			model: Module,
			training_config: Dict,
			feature_dim: int,
			num_classes: int,
			# topk: Tuple[int, ...] = (1, 2),
			freeze_model: bool = False,
	) -> None:
		"""Linear classifier for benchmarking (LP or FT)
		Args:
			model:
				Model used for feature extraction. Must define a forward(images) method
				that returns a feature tensor.
			feature_dim:
				Dimension of features returned by forward method of model.
			num_classes:
				Number of classes in the dataset.
			topk:
				Tuple of integers defining the top-k accuracy metrics to compute.
			freeze_model:
				If True, the model is frozen and only the classification head is
				trained. This corresponds to the linear eval setting. Set to False for
				finetuning.
		"""
		super().__init__()
		self.save_hyperparameters(ignore="model")
		
		self.model = model
		self.feature_dim = feature_dim
		self.num_classes = num_classes
		# self.topk = topk
		self.freeze_model = freeze_model
		self.training_config = training_config
		
		self.classification_head = Linear(
			feature_dim,
			num_classes
		)
		self.criterion = CrossEntropyLoss()
		self.max_acc = 0.0
	
	def forward(self, images: Tensor) -> Tensor:
		features = self.model.backbone(images).flatten(start_dim=1)
		return self.classification_head(features)
	
	def shared_step(self, batch, batch_idx) -> Tuple[Tensor, float]:
		"""

		:param batch:
		:param batch_idx:
		:return:
		"""
		images, targets = batch[0], batch[1]
		predictions = self.forward(images)
		loss = self.criterion(predictions, targets)
		# _, predicted_labels = predictions.topk(max(self.topk))
		# topk = mean_topk_accuracy(predicted_labels, targets, k=self.topk)
		# return loss, topk
		
		# Convert logits to predicted classes
		_, predicted_classes = torch.max(predictions, 1)
		# Calculate correct predictions
		correct_predictions = (predicted_classes == targets).float().sum()
		# Calculate accuracy
		acc = correct_predictions / len(targets)
		
		return loss, acc
	
	def training_step(self, batch, batch_idx) -> Tensor:
		loss, acc = self.shared_step(batch=batch, batch_idx=batch_idx)
		batch_size = len(batch[1])
		self.log(
			"train_loss",
			loss,
			prog_bar=True,
			sync_dist=True,
			batch_size=batch_size
		)
		self.log(
			"train_acc",
			acc * 100,
			prog_bar=True,
			sync_dist=True,
			batch_size=batch_size
		)
		# log_dict = {f"train_top{k}": acc for k, acc in topk.items()}
		# self.log("train_loss", loss, prog_bar=True, sync_dist=True, batch_size=batch_size)
		# self.log_dict(log_dict, sync_dist=True, batch_size=batch_size)
		return loss
	
	def validation_step(self, batch, batch_idx) -> Tensor:
		loss, acc = self.shared_step(batch=batch, batch_idx=batch_idx)
		batch_size = len(batch[1])
		self.log(
			"val_accuracy",
			acc * 100,
			prog_bar=True,
			sync_dist=True,
			batch_size=batch_size
		)
		if acc > self.max_accuracy:
			self.max_acc = acc
		self.log(
			"max_accuracy",
			self.max_acc * 100.0,
			prog_bar=True,
			sync_dist=True,
			batch_size=batch_size
		)
		# self.log("val_loss", loss, prog_bar=True, sync_dist=True, batch_size=batch_size)
		# self.log_dict(log_dict, prog_bar=True, sync_dist=True, batch_size=batch_size)
		return loss
	
	def configure_optimizers(self):
		parameters = list(self.classification_head.parameters())
		if not self.freeze_model:
			# FineTuning backprop through entire model
			parameters += self.model.backbone.parameters()
		optimizer = get_optimizer(
			params=parameters,
			optimizer_config=self.training_config
		)
		scheduler = get_scheduler(
			optimizer=optimizer,
			lrs_config=self.training_config
		)
		return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]
	
	def on_fit_start(self) -> None:
		# Freeze model weights.
		if self.freeze_model:
			deactivate_requires_grad(model=self.model)
	
	def on_fit_end(self) -> None:
		# Unfreeze model weights.
		if self.freeze_model:
			activate_requires_grad(model=self.model)
