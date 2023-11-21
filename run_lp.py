"""
Perform Linear Evaluation : Supports FineTuning, Linear Probing etc
"""
import os
from typing import Dict
from argparse import (
	ArgumentParser,
	Namespace
)
from pathlib import Path
import yaml
import numpy as np
import torch
from lightly.utils.benchmarking import MetricCallback
from lightly.utils.dist import (
	print_rank_zero,
	rank
)
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from dataloader import DataManager, get_pseudo_labels, PseudoLabeledData
from linear_head import LinearClassificationHead
from training_framework import SimCLR
from torchvision.transforms import v2
import lightly.data as data
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils import get_optimizer, get_scheduler
from losses import get_loss
from torch.nn import Linear


def _parse_args(verbose=True):
	parser = ArgumentParser(description="Linear Evaluation Arguments")
	parser.add_argument(
		'--dataset',
		type=str,
		default='cifar10.dog_cat',
		help='Pass Dataset'
	)
	parser.add_argument(
		"--checkpoint",
		type=Path,
		default=None,
	)
	parser.add_argument(
		"--puPL",
		type=bool,
		default=False,
		help="if enabled, pseudo-labeling will happen before training"
	)
	parser.add_argument(
		"--algo",
		type=str,
		default='kMeans',
		help=" kMeans | kMeans++ | PUkMeans++ | DBSCAN "
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


def extract_features(encoder, dataloader: DataLoader) -> [torch.Tensor, torch.Tensor]:
	"""
	Given a dataloader return the extracted features and labels
	:param encoder:
	:param dataloader:
	:return:
	"""
	features = []
	labels = []
	with ((torch.no_grad())):
		for mini_batch in dataloader:
			img, target, _ = mini_batch
			img = img.to(encoder.device)
			target = target.to(encoder.device)
			feature = encoder.backbone(img).squeeze()
			feature = F.normalize(feature, dim=1)
			features.append(feature)
			labels.append(target)
	extracted_features = torch.cat(features, dim=0).t().contiguous()
	extracted_labels = torch.cat(labels, dim=0).t().contiguous()
	return extracted_features, extracted_labels


def run_linear_eval(args: Namespace, config: Dict) -> None:
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
		feat_tr, lbl_tr = model.extract_features(dataloader=dataloader_train_sv)
		feat_te, lbl_te = model.extract_features(dataloader=dataloader_test)
		
		# create new dataset from feat
		train_dataset = torch.utils.data.TensorDataset(feat_tr, lbl_tr)
		test_dataset = torch.utils.data.TensorDataset(feat_te, lbl_te)
		
		train_dataset.transform = data_manager.basic_transform
		test_dataset.transform = data_manager.basic_transform
		
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
		
		# Linear Model
		classification_head = Linear(
			model.feat_dim,
			data_manager.num_classes
		)
		
		# Get objective
		criterion = get_loss(framework_config=framework_config)
		opt = get_optimizer(
			params=classification_head.parameters(),
			optimizer_config=training_config
		)
		
		scheduler = get_scheduler(
			optimizer=opt,
			lrs_config=training_config,
			verbose=False
		)
		
		# --- Logging -----
		logger = TensorBoardLogger(
			save_dir=os.path.join(args.log_dir, f"tf_logging/{args.dataset}"),
			name=f"{args.exp_name}",
			version=f"seed={seed}",
		),
		
		# ----- delete model and trainer + free up cuda memory ---
		del model
		torch.cuda.reset_peak_memory_stats()
		torch.cuda.empty_cache()
		print('val acc = {} +- {}'.format(np.mean(val_acc), np.var(val_acc)))


if __name__ == '__main__':
	# ----- Parse Arguments ----- #
	arguments = _parse_args()
	run_linear_eval(
		args=arguments,
		config=yaml.load(
			open('config_linear_eval.yaml'),
			Loader=yaml.FullLoader
		)
	)
