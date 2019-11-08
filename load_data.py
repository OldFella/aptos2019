import os
import torch
import torchvision
import torchvision.transforms as transforms
from config import DATA_PATH

# PATH = '/home/gwent/projects/kaggle_comp/aptos2019-blindness-detection/'
# /home/gwent/projects/kaggle_comp/aptos2019-blindness-detection/validation/






normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])

train_transforms = transforms.Compose([
	transforms.Resize((224,224)),
	transforms.RandomVerticalFlip(0.5),
	transforms.RandomHorizontalFlip(0.5),
	transforms.RandomRotation(360),
	transforms.ToTensor(),
	normalize
	])

validation_transforms = transforms.Compose([
	transforms.Resize((224,224)),
	transforms.ToTensor(),
	normalize
	])
# validation_transforms = transforms.Compose([
# 	transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# 	transforms.ToTensor()])

def create_loader(folder, transforms, batch_size = 4):
	dataset = torchvision.datasets.ImageFolder(
		root = DATA_PATH + folder,
		transform = transforms)
	
	loader = torch.utils.data.DataLoader(
		dataset,
		batch_size = batch_size,
		shuffle = True,
		num_workers = 1)

	return loader

