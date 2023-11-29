"""
Perform Contrastive Representation Learning
"""
import os
import time
from argparse import ArgumentParser
from pathlib import Path
import pytorch_lightning as pl
import torch
import yaml
from lightly.utils.dist import rank
from pytorch_lightning.callbacks import (
	LearningRateMonitor,
	ModelCheckpoint
)
from pytorch_lightning.loggers import TensorBoardLogger
from dataloader import DataManager
from training_framework import SimCLR


def _parse_args(verbose=True):
	parser = ArgumentParser(description="PreTraining Arguments")
	# ----- logging related ------ #
	parser.add_argument(
		"--log_dir",
		type=Path,
		default=os.path.join(os.getcwd(), "logs")
	)
	parser.add_argument(
		"--exp_name",
		type=str,
		default="testbed",
		help="this will determine the path of saved results"
	)
	parser.add_argument(
		"--checkpoint",
		type=Path,
		default=None,
		help="path of the ckpt e.g. os.path.join(os.getcwd(), pl_logs/checkpoints/cifar10/pt-SimCLR-0-v1.ckpt)"
	)
	parser.add_argument(
		'--n_repeat',
		type=int,
		default=1,
		help='Specify number of repeat runs'
	)
	parser.add_argument(
		'--dataset',
		type=str,
		default='cifar10.dog_cat',
		help='Pass Dataset'
	)
	args = parser.parse_args()
	verbose and print(args)
	return args


def run_contrastive_training(args, config):
	"""
	contrastive training
	"""
	n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
	# # TODO: **** remove this condition when sCL with ddp support is implemented *****
	# if n_gpus >= 2:
	# 	raise NotImplementedError('Not all methods support ddp - run using single gpu')
	
	# ---- parse config ----
	config = config[args.dataset]
	framework_config = config["framework_config"]
	data_config = config["data_config"]
	training_config = config["training_config"]
	
	# --- Run Experiment for different seeds # -----
	runs = []
	for seed in range(args.n_repeat):
		torch.set_float32_matmul_precision("high")
		pl.seed_everything(seed)
		# --- Data -----
		data_manager = DataManager(
			dataset=args.dataset,
			data_config=data_config,
			gpu_strategy="ddp" if n_gpus >= 2 else "auto"
		)
		dataloader_train_mv, _, dataloader_train_val, dataloader_test = data_manager.get_data()
		print("Done with Data Loading")
		# --- Model -------
		if args.checkpoint is not None:
			print('Loading PreTrained Model from Checkpoint {}'.format(args.checkpoint))
			model = SimCLR.load_from_checkpoint(
				args.checkpoint,
				framework_config=framework_config,
				training_config=training_config,
				data_config=data_config,
				val_dataloader=dataloader_train_val,
				num_classes=data_manager.num_classes,
				gather_distributed=True if n_gpus >= 2 else False
			)
			# --- evaluate the model - sanity check ---
			with torch.no_grad():
				model.on_validation_epoch_start()
				for batch_ix, batch in enumerate(dataloader_train_val):
					model.validation_step(batch_idx=batch_ix, batch=batch)
					model.on_validation_epoch_end()
		else:
			model = SimCLR(
				framework_config=framework_config,
				training_config=training_config,
				data_config=data_config,
				val_dataloader=dataloader_train_val,
				num_classes=data_manager.num_classes,
				gather_distributed=True if n_gpus >= 2 else False
			)
		# --- Logging -----
		logger = TensorBoardLogger(
			save_dir=os.path.join(args.log_dir, f"tf_logging/{args.dataset}"),
			name=f"{args.exp_name}",
			version=f"seed={seed}",
		)
		checkpoint_callback = ModelCheckpoint(
			dirpath=os.path.join(args.log_dir, f"checkpoints/{args.dataset}/{args.exp_name}/{seed}"),
			filename="{epoch:d}",
			every_n_epochs=training_config.get("save_model_freq", None),
			save_top_k=-1,
		)
		lr_monitor = LearningRateMonitor(logging_interval='step')
		# --- Train ------
		trainer = pl.Trainer(
			max_epochs=training_config.get('epochs'),
			# --- devices ----
			devices=n_gpus,
			accelerator="gpu" if torch.cuda.is_available() else "mps",
			strategy="ddp" if n_gpus >= 2 else "auto",
			# --- reproducibility ---
			sync_batchnorm=True if n_gpus >= 2 else False,
			deterministic=True,
			use_distributed_sampler=True if n_gpus >= 2 else False,
			# --- logging ---
			logger=logger,
			default_root_dir=args.log_dir,
			check_val_every_n_epoch=training_config.get('eval_freq', 1),
			log_every_n_steps=1,
			callbacks=[
				checkpoint_callback,
				lr_monitor
			],
		)
		start = time.time()
		trainer.fit(
			model=model,
			train_dataloaders=dataloader_train_mv,
			val_dataloaders=dataloader_test,
		)
		end = time.time()
		run = {
			"batch_size": data_manager.train_batch_size,
			"epochs": training_config.get('epochs'),
			"max_accuracy": model.max_accuracy,
			"runtime": end - start,
			"gpu_memory_usage": torch.cuda.max_memory_allocated(),
			"seed": seed,
		}
		if rank() == 0:
			runs.append(run)
			logger.log_metrics(metrics=run)
			logger.log_hyperparams(config)
			print(run)
		# ----- delete model and trainer + free up cuda memory ---
		del model
		del trainer
		torch.cuda.reset_peak_memory_stats()
		torch.cuda.empty_cache()


if __name__ == '__main__':
	# ----- Parse Arguments -----
	run_contrastive_training(
		args=_parse_args(),
		config=yaml.load(
			open('config_pretrain.yaml'),
			Loader=yaml.FullLoader
		)
	)
