
import os
import torch
import torch.onnx as onnx
from torch.utils.data import DataLoader
import logging
import argparse
import wandb as wb

from train_utils import train, inference
from data_utils import DRIVE_dataset

seed = 5
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

run_no = input("Enter run number: ")
savePath = '/home/anubhav/DRIVE_segmentation/Results/Run'+run_no+'/'
if not os.path.isdir(savePath):
   os.makedirs(savePath)
logging.basicConfig(filename=savePath+'run_'+run_no+'.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--epochs', default=500, type=int, help='number of epochs')
	parser.add_argument('--batch_size', default=1, type=int, help='batch size')
	parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
	parser.add_argument('--gpu', default=2, type=int, help='gpu number')
	parser.add_argument('--model', default='unet', type=str, help='which model?')
	parser.add_argument('--optimizer', default='adam', type=str, help='which optimizer?')
	parser.add_argument('--loss_fn', default='dice_bce', type=str, help='which loss function?')
	args = parser.parse_args()
		
	args.savePath = savePath
	args.device = torch.device("cuda:"+str(args.gpu) if torch.cuda.is_available() else "cpu")

	train_dl = DataLoader(DRIVE_dataset(split='train'), batch_size=args.batch_size, shuffle=True)
	valid_dl = DataLoader(DRIVE_dataset(split='valid'), batch_size=args.batch_size, shuffle=False)
	test_dl = DataLoader(DRIVE_dataset(split='test'), batch_size=args.batch_size, shuffle=False)

	api_key = '1fc87c4a28d59e9d394d492ea4d21eb4b3c3acd8'
	wb.login(key=api_key)
	
	hyperparams = dict(
		dataset = 'DRIVE',
		architecture = 'CNN',
		epochs = args.epochs,
		batch_size = args.batch_size
	)
	
	wb.init(project='retinal-vessel-segmentation', name=f'run_{run_no}', config=hyperparams)
	train(train_dl, valid_dl, args, logger)
	inference(test_dl, args, logger)
	wb.finish()