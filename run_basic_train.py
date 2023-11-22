"""
Perform Linear Evaluation : Supports FineTuning, Linear Probing etc
"""
import os
from argparse import (
	ArgumentParser,
	Namespace
)
from pathlib import Path
from typing import Dict
import lightly.data as data
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import yaml
from lightly.models.utils import activate_requires_grad, deactivate_requires_grad
from lightly.utils.benchmarking import MetricCallback
from pytorch_lightning import LightningModule
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor
from torch.nn import CrossEntropyLoss, Linear, Module
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import DataManager
from training_framework import SimCLR
from utils import get_optimizer, get_scheduler


def _parse_args(verbose=True):
	parser = ArgumentParser(description="Linear Evaluation Arguments")
	parser.add_argument(
		'--dataset',
		type=str,
		default='cifar10.dog_cat',
		help='Pass Dataset'
	)
	parser.add_argument(
		'--n_repeat',
		type=int,
		default=1,
		help='Specify number of repeat runs'
	)
	# ----- logging related ------ #
	parser.add_argument(
		"--log_dir",
		type=Path,
		default=os.path.join(os.getcwd(), "logs")
	)
	parser.add_argument(
		"--exp_name",
		type=str,
		default="lp-testbed",
		help="logs saved inside exp sub-folder in the logs folder"
	)
	args = parser.parse_args()
	verbose and print(args)
	
	return args


class LinearClassificationHead(LightningModule):
	def __init__(
			self,
			model: Module,
			training_config: Dict,
			feature_dim: int,
			num_classes: int,
			freeze_model: bool = True,
	) -> None:
		super().__init__()
		self.model = model
		self.feature_dim = feature_dim
		self.num_classes = num_classes
		self.training_config = training_config
		self.freeze_model = freeze_model
		
		self.classification_head = Linear(
			feature_dim,
			num_classes
		)
		self.criterion = CrossEntropyLoss()
		self.max_accuracy = 0.0
	
	def training_step(self, batch, batch_idx) -> Tensor:
		images, targets = batch[0], batch[1]
		predictions = self.classification_head(images)
		loss = self.criterion(predictions, targets)
		# Convert logits to predicted classes
		_, predicted_classes = torch.max(predictions, 1)
		# Calculate correct predictions
		correct_predictions = (predicted_classes == targets).sum().item()
		# Calculate accuracy
		accuracy = correct_predictions / targets.size(0)
		batch_size = len(targets)
		self.log(
			"train_loss",
			loss,
			prog_bar=True,
			sync_dist=True,
			batch_size=batch_size
		)
		self.log(
			"train_acc",
			accuracy,
			prog_bar=True,
			sync_dist=True,
			batch_size=batch_size
		)
		return loss
	
	def configure_optimizers(self):
		parameters = list(self.classification_head.parameters())
		if not self.freeze_model:
			print("Encoder not Frozen")
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


def extract_features(encoder, dataloader: DataLoader) -> [torch.Tensor, torch.Tensor]:
	"""
	Given a dataloader return the extracted features and labels
	:param encoder:
	:param dataloader:
	:return:
	"""
	features = []
	labels = []
	encoder.eval()
	with torch.no_grad():
		for mini_batch in tqdm(dataloader):
			img, target, _ = mini_batch
			if torch.cuda.is_available():
				img = img.cuda()
				target = target.cuda()
				encoder = encoder.cuda()
			feature = encoder(img).squeeze()
			feature = F.normalize(feature, dim=1)
			features.append(feature)
			labels.append(target)
	extracted_features = torch.cat(features, dim=0).contiguous()
	extracted_labels = torch.cat(labels, dim=0).contiguous()
	# print(extracted_features.shape)
	# print(extracted_labels.shape)
	return extracted_features, extracted_labels


def run_linear_eval(args: Namespace, config: Dict, freeze_encoder: bool = True, pseudo_label: bool = False) -> None:
	"""
	Runs a linear evaluation on the given model. If no model is given trains one from scratch
	"""
	# --- Device Setup ---
	n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
	if n_gpus >= 2:
		raise NotImplementedError('Not all methods support ddp - run using single gpu')
	
	# --- parse configs ---
	config = config[args.dataset]
	framework_config = config["framework_config"]
	data_config = config["data_config"]
	training_config = config["training_config"]
	
	# --- Run Experiment for different seeds # -----
	val_acc = []
	for seed in range(args.n_repeat):
		torch.set_float32_matmul_precision("high")
		pl.seed_everything(seed)
		# --- Data -----
		data_manager = DataManager(
			dataset=args.dataset,
			data_config=data_config,
			gpu_strategy="ddp" if n_gpus >= 2 else "auto"
		)
		_, dataloader_train_sv, _, dataloader_test = data_manager.get_data()
		# --- Model -------
		if args.checkpoint is not None:
			print('Loading PreTrained Model from Checkpoint {}'.format(args.checkpoint))
			model = SimCLR.load_from_checkpoint(
				args.checkpoint,
				framework_config=framework_config,
				training_config=training_config,
				data_config=data_config,
				val_dataloader=dataloader_test,
				num_classes=data_manager.num_classes,
				gather_distributed=True if n_gpus >= 2 else False
			)
			model.max_accuracy = 0.0
		else:
			raise ValueError("You need to pass model chkpt to perform LP evaluation")
		
		print("feature extraction")
		feat_tr, lbl_tr = extract_features(dataloader=dataloader_train_sv, encoder=model.backbone)
		feat_te, lbl_te = extract_features(dataloader=dataloader_test, encoder=model.backbone)
		
		# create new dataset from feat
		train_dataset = torch.utils.data.TensorDataset(feat_tr, lbl_tr)
		train_dataset.transform = data_manager.basic_transform
		train_dataset = data.LightlyDataset.from_torch_dataset(train_dataset)
		
		test_dataset = torch.utils.data.TensorDataset(feat_te, lbl_te)
		test_dataset.transform = data_manager.basic_transform
		test_dataset = data.LightlyDataset.from_torch_dataset(test_dataset)
		
		train_loader = DataLoader(
			train_dataset,
			batch_size=data_manager.train_batch_size,
			shuffle=True,
			drop_last=True,
			num_workers=data_manager.num_worker,
		)
		test_loader = DataLoader(
			test_dataset,
			batch_size=data_manager.test_batch_size,
			shuffle=True,
			drop_last=True,
			num_workers=data_manager.num_worker,
		)
		
		lin_classifier = LinearClassificationHead(
			model=model,
			training_config=training_config,
			feature_dim=model.feat_dim,
			num_classes=data_manager.num_classes,
			freeze_model=freeze_encoder,
		)
		
		# --- Logging -----
		logger = TensorBoardLogger(
			save_dir=os.path.join(args.log_dir, f"tf_logging/{args.dataset}"),
			name=f"{args.exp_name}",
			version=f"seed={seed}",
		)
		metric_callback = MetricCallback()
		
		# ----- Train linear classifier.
		trainer = Trainer(
			max_epochs=training_config.get('epochs'),
			# --- devices ----
			devices=n_gpus,
			accelerator="gpu" if torch.cuda.is_available() else "mps",
			strategy="ddp" if n_gpus >= 2 else "auto",
			# --- reproducibility
			sync_batchnorm=True if n_gpus >= 2 else False,
			deterministic=True,
			use_distributed_sampler=True if n_gpus >= 2 else False,
			# --- logging ---
			logger=logger,
			default_root_dir=args.log_dir,
			check_val_every_n_epoch=training_config.get('eval_freq', 1),
			log_every_n_steps=1,
			callbacks=[
				LearningRateMonitor(logging_interval='step'),
				metric_callback,
			],
			# precision="16-mixed",
		)
		
		# ---- Kick off Training
		trainer.fit(
			model=lin_classifier,
			train_dataloaders=train_loader,
			val_dataloaders=test_loader,
		)
		val_acc.append(lin_classifier.max_accuracy.cpu())
	
	# ----- delete model and trainer + free up cuda memory ---
	del model
	torch.cuda.reset_peak_memory_stats()
	torch.cuda.empty_cache()
	print('val acc = {} +- {}'.format(np.mean(val_acc), np.var(val_acc)))


if __name__ == '__main__':
	# ----- Parse Arguments ----- #
	arguments = _parse_args()
	
	if arguments.mode == 'lp':
		freeze = True
		print("Running linear probing")
	
	elif arguments.mode == 'ft':
		freeze = False
		print("Running Finetuning")
	
	else:
		raise ValueError(
			'Need to Linear Probe or Finetune'
		)
	
	run_linear_eval(
		args=arguments,
		config=yaml.load(
			open('config_linear_eval.yaml'),
			Loader=yaml.FullLoader
		),
		freeze_encoder=freeze,
		pseudo_label=True if arguments.puPL else False
	)
