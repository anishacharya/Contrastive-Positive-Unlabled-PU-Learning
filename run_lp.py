"""
Perform Linear Probe
"""

from argparse import (
	ArgumentParser
)
from pathlib import Path
from torch import optim
import torch.nn as nn
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from dataloader import DataManager
from training_framework import SimCLR
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
import numpy as np


def _parse_args(verbose=True):
	parser = ArgumentParser(description="PreTraining Arguments")
	# ----- logging related ------ #
	parser.add_argument(
		"--checkpoint",
		type=Path,
		default="/Users/aa56927-admin/Desktop/cifar.ckpt",
		help="path of the ckpt e.g. os.path.join(os.getcwd(), pl_logs/checkpoints/cifar10/pt-SimCLR-0-v1.ckpt)"
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


def extract_embeddings(encoder, dataloader: DataLoader) -> [torch.Tensor, torch.Tensor]:
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
	extracted_features = torch.cat(features, dim=0).contiguous().cpu().numpy()
	extracted_labels = torch.cat(labels, dim=0).contiguous().cpu().numpy()
	# print(extracted_features.shape)
	# print(extracted_labels.shape)
	return extracted_features, extracted_labels


# Training and Evaluation Function
def train_and_evaluate(model, criterion, optimizer, train_loader, test_loader, epochs=10):
	"""
	Train and Eval script
	"""
	if torch.cuda.is_available():
		model.cuda()
	
	for epoch in range(epochs):
		# Training
		model.train()
		total_loss = 0
		for inputs, labels in train_loader:
			if torch.cuda.is_available():
				inputs, labels = inputs.cuda(), labels.cuda()
			
			optimizer.zero_grad()
			outputs = model(inputs)
			loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			total_loss += loss.item()
		
		avg_loss = total_loss / len(train_loader)
		print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_loss}")
		
		# Evaluation
		model.eval()
		correct = 0
		total = 0
		with torch.no_grad():
			for inputs, labels in test_loader:
				if torch.cuda.is_available():
					inputs, labels = inputs.cuda(), labels.cuda()
				
				outputs = model(inputs)
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()
		
		accuracy = 100 * correct / total
		print(f'Epoch {epoch + 1}/{epochs}, Test Accuracy: {accuracy}%')


if __name__ == '__main__':
	args = _parse_args()
	config = yaml.load(
		open('config_lp.yaml'),
		Loader=yaml.FullLoader
	)
	torch.set_float32_matmul_precision("high")
	
	seed = 0
	# ---- parse config ----
	config = config[args.dataset]
	framework_config = config["framework_config"]
	data_config = config["data_config"]
	training_config = config["training_config"]
	pl.seed_everything(seed)
	# --- Data -----
	data_manager = DataManager(
		dataset=args.dataset,
		data_config=data_config
	)
	_, dataloader_train_sv, _, dataloader_test = data_manager.get_data()
	
	print('Loading PreTrained Model from Checkpoint {}'.format(args.checkpoint))
	model = SimCLR.load_from_checkpoint(
		args.checkpoint,
		framework_config=framework_config,
		training_config=training_config,
		data_config=data_config,
		val_dataloader=None,
		num_classes=data_manager.num_classes,
		map_location=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
	)
	
	print("Extracting Train Embeddings")
	feat_tr, lbl_tr = extract_embeddings(dataloader=dataloader_train_sv, encoder=model.backbone)
	print("Extracting Test Embeddings")
	feat_te, lbl_te = extract_embeddings(dataloader=dataloader_test, encoder=model.backbone)
	
	tr_bs = data_config.get('train_batch_size', 256)
	te_bs = data_config.get('test_batch_size', 1000)
	num_worker = data_config.get('num_worker', 1)
	
	tr_dataset = TensorDataset(torch.tensor(feat_tr, dtype=torch.float32), torch.tensor(lbl_tr, dtype=torch.long))
	tr_dataloader = DataLoader(tr_dataset, batch_size=tr_bs, shuffle=True, num_workers=num_worker)
	
	te_dataset = TensorDataset(torch.tensor(feat_te, dtype=torch.float32), torch.tensor(lbl_te, dtype=torch.long))
	te_dataloader = DataLoader(te_dataset, batch_size=te_bs, shuffle=False, num_workers=num_worker)
	
	# Move to GPU if available
	if torch.cuda.is_available():
		model.cuda()
	
	# Loss and Optimizer
	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr=0.1)
	
	num_features = feat_tr.shape[1]  # Assuming feat_tr is a 2D array of shape (num_samples, num_features)
	num_classes = np.unique(lbl_tr).size
	lin_model = nn.Linear(num_features, num_classes)
	
	# Now, use the train_and_evaluate function with the dataloaders
	train_and_evaluate(
		lin_model,
		criterion,
		optimizer,
		tr_dataloader,
		te_dataloader
	)
