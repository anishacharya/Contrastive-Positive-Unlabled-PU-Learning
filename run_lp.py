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
from losses import get_loss
from utils import get_optimizer, get_scheduler
from torch.utils.data import default_collate
from torchvision.transforms import v2
from torch.autograd import Variable


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
	parser.add_argument(
		'--mixup',
		type=bool,
		default=False,
		help='Mixup Data Aug'
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


def mixup_data(x, y, alpha=1.0, use_cuda=True):
	"""
	Returns mixed inputs, pairs of targets, and lambda
	"""
	if alpha > 0:
		lam = np.random.beta(alpha, alpha)
	else:
		lam = 1
	
	batch_size = x.size()[0]
	if use_cuda:
		index = torch.randperm(batch_size).cuda()
	else:
		index = torch.randperm(batch_size)
	
	mixed_x = lam * x + (1 - lam) * x[index, :]
	y_a, y_b = y, y[index]
	return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
	"""
	mixed loss
	"""
	return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# Training and Evaluation Function
def train_and_evaluate(lin_mdl, criterion, optimizer, train_loader, test_loader, epochs=10, mixup=False):
	"""
	Train and Eval script
	"""
	if torch.cuda.is_available():
		lin_mdl.cuda()
	best_acc = 0
	for epoch in range(epochs):
		# Training
		lin_mdl.train()
		total_loss = 0
		for inputs, labels in train_loader:
			if torch.cuda.is_available():
				inputs, labels = inputs.cuda(), labels.cuda()
			optimizer.zero_grad()
			if mixup:
				# print("mixup happening")
				inputs, targets_a, targets_b, lam = mixup_data(
					x=inputs,
					y=labels,
					alpha=1,
					use_cuda=True if torch.cuda.is_available() else False
				)
				inputs, targets_a, targets_b = map(Variable, (inputs, targets_a, targets_b))
				outputs = lin_mdl(inputs)
				loss = mixup_criterion(criterion, outputs, targets_a, targets_b, lam)
			else:
				outputs = lin_mdl(inputs)
				loss = criterion(outputs, labels)
			loss.backward()
			optimizer.step()
			total_loss += loss.item()
		
		avg_loss = total_loss / len(train_loader)
		print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_loss}")
		
		# Evaluation
		lin_mdl.eval()
		correct = 0
		total = 0
		with torch.no_grad():
			for inputs, labels in test_loader:
				if torch.cuda.is_available():
					inputs, labels = inputs.cuda(), labels.cuda()
				
				outputs = lin_mdl(inputs)
				_, predicted = torch.max(outputs.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum().item()
		
		accuracy = 100 * correct / total
		if accuracy > best_acc:
			best_acc = accuracy
		print(f'Epoch {epoch + 1}/{epochs}, Test Accuracy: {accuracy}%')
	
	print("\n Best Linear Probe Accuracy: {}\n".format(best_acc))
	return best_acc


def collate_fn(batch):
	"""
	MixUp / CutMix enable
	"""
	mixup = v2.MixUp(num_classes=2)
	return mixup(*default_collate(batch))


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
	te_dataset = TensorDataset(torch.tensor(feat_te, dtype=torch.float32), torch.tensor(lbl_te, dtype=torch.long))
	
	tr_dataloader = DataLoader(tr_dataset, batch_size=tr_bs, shuffle=True, num_workers=num_worker)
	te_dataloader = DataLoader(te_dataset, batch_size=te_bs, shuffle=False, num_workers=num_worker)
	
	num_features = feat_tr.shape[1]  # Assuming feat_tr is a 2D array of shape (num_samples, num_features)
	num_classes = np.unique(lbl_tr).size
	lin_model = nn.Linear(num_features, num_classes)
	# Move to GPU if available
	if torch.cuda.is_available():
		lin_model.cuda()
	
	# Loss and Optimizer
	opt = get_optimizer(
		params=lin_model.parameters(),
		optimizer_config=training_config
	)
	scheduler = get_scheduler(
		optimizer=opt,
		lrs_config=training_config,
		verbose=False
	)
	criterion = get_loss(framework_config=framework_config)
	
	# Now, use the train_and_evaluate function with the dataloaders
	lp_acc = train_and_evaluate(
		lin_mdl=lin_model,
		criterion=criterion,
		optimizer=opt,
		train_loader=tr_dataloader,
		test_loader=te_dataloader,
		epochs=training_config.get("epochs", 10),
		mixup=True if args.mixup is True else False
	)
