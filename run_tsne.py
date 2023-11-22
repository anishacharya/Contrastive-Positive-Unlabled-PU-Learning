"""
Perform tsne eval of encoder
"""

from argparse import (
	ArgumentParser
)
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import DataManager
from training_framework import SimCLR
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt


def _parse_args(verbose=True):
	parser = ArgumentParser(description="PreTraining Arguments")
	# ----- logging related ------ #
	parser.add_argument(
		"--checkpoint",
		type=Path,
		default=None,
		help="path of the ckpt e.g. os.path.join(os.getcwd(), pl_logs/checkpoints/cifar10/pt-SimCLR-0-v1.ckpt)"
	)
	parser.add_argument(
		'--dataset',
		type=str,
		default='cifar10.dog_cat',
		help='Pass Dataset'
	)
	parser.add_argument(
		'--fig_name',
		type=str,
		default='tsne_plots/test.pdf',
		help='Save Figure'
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


if __name__ == '__main__':
	args = _parse_args()
	config = yaml.load(
		open('config_pretrain.yaml'),
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
	_, _, _, dataloader_test = data_manager.get_data()
	print('Loading PreTrained Model from Checkpoint {}'.format(args.checkpoint))
	model = SimCLR.load_from_checkpoint(
		args.checkpoint,
		framework_config=framework_config,
		training_config=training_config,
		data_config=data_config,
		val_dataloader=None,
		num_classes=data_manager.num_classes
	)
	
	print("Extracting Embeddings")
	feat_te, lbl_te = extract_embeddings(dataloader=dataloader_test, encoder=model.backbone)
	
	print("TSNE Visualization")
	tsne = TSNE(n_components=2, verbose=1, random_state=123)
	z = tsne.fit_transform(feat_te)
	plt.figure(figsize=(8, 6), dpi=300)
	ax = sns.scatterplot(
		x=z[:, 0],
		y=z[:, 1],
		hue=lbl_te,  # Class labels
		palette=['cornflowerblue', 'rosybrown'],  # Adjust number of colors based on classes
		s=50,  # Increase scatter point size
		alpha=0.7  # Slight transparency
	)
	handles, labels = ax.get_legend_handles_labels()
	legend_labels = ['Positive (y = +1)', 'Negative (y = -1)']
	custom_legend = plt.legend(handles,
	                           legend_labels,
	                           # title='',
	                           # loc='upper left',
	                           # ncol=2,
	                           # bbox_to_anchor=(0, 1.2)
	                           )
	plt.xlabel('tsne PC-1')
	plt.ylabel('tsne PC-2')
	plt.grid()
	ax.figure.savefig(args.fig_name)
