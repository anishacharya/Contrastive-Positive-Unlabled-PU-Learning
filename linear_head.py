""" Attach a Linear Head """
from typing import Dict, Tuple
from lightly.models.utils import activate_requires_grad, deactivate_requires_grad
from lightly.utils.benchmarking.topk import mean_topk_accuracy
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import CrossEntropyLoss, Linear, Module
from training_utils import get_optimizer, get_scheduler


class LinearClassificationHead(LightningModule):
	def __init__(
			self,
			model: Module,
			training_config: Dict,
			feature_dim: int,
			num_classes: int,
			topk: Tuple[int, ...] = (1, 2),
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
		self.topk = topk
		self.freeze_model = freeze_model
		self.training_config = training_config
		
		self.classification_head = Linear(
			feature_dim,
			num_classes
		)
		self.criterion = CrossEntropyLoss()
		self.max_accuracy = 0.0
	
	def forward(self, images: Tensor) -> Tensor:
		features = self.model.backbone(images).flatten(start_dim=1)
		return self.classification_head(features)
	
	def shared_step(self, batch, batch_idx) -> Tuple[Tensor, Dict[int, Tensor]]:
		"""

		:param batch:
		:param batch_idx:
		:return:
		"""
		images, targets = batch[0], batch[1]
		predictions = self.forward(images)
		loss = self.criterion(predictions, targets)
		_, predicted_labels = predictions.topk(max(self.topk))
		topk = mean_topk_accuracy(predicted_labels, targets, k=self.topk)
		return loss, topk
	
	def training_step(self, batch, batch_idx) -> Tensor:
		loss, topk = self.shared_step(batch=batch, batch_idx=batch_idx)
		batch_size = len(batch[1])
		log_dict = {f"train_top{k}": acc for k, acc in topk.items()}
		self.log(
			"train_loss", loss, prog_bar=True, sync_dist=True, batch_size=batch_size
		)
		self.log_dict(log_dict, sync_dist=True, batch_size=batch_size)
		return loss
	
	def validation_step(self, batch, batch_idx) -> Tensor:
		loss, topk = self.shared_step(batch=batch, batch_idx=batch_idx)
		batch_size = len(batch[1])
		log_dict = {f"val_top{k}": acc for k, acc in topk.items()}
		if topk[1] > self.max_accuracy:
			self.max_accuracy = topk[1]
		self.log("val_loss", loss, prog_bar=True, sync_dist=True, batch_size=batch_size)
		self.log_dict(log_dict, prog_bar=True, sync_dist=True, batch_size=batch_size)
		return loss
	
	def configure_optimizers(self):
		parameters = list(self.classification_head.parameters())
		if not self.freeze_model:
			# FineTuning backprop through entire model
			parameters += self.model.backbone.parameters()
		optim = get_optimizer(
			params=parameters,
			optimizer_config=self.training_config
		)
		scheduler = get_scheduler(
			optimizer=optim,
			lrs_config=self.training_config
		)
		return [optim], [scheduler]
	
	def on_fit_start(self) -> None:
		# Freeze model weights.
		if self.freeze_model:
			deactivate_requires_grad(model=self.model)
	
	def on_fit_end(self) -> None:
		# Unfreeze model weights.
		if self.freeze_model:
			activate_requires_grad(model=self.model)
