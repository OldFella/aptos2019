import os
import torch
import torchvision
import torchvision.transforms as transforms
from _config import PATH, DATA_PATH

# PATH = '/home/gwent/projects/kaggle_comp/aptos2019-blindness-detection/'
# /home/gwent/projects/kaggle_comp/aptos2019-blindness-detection/validation/







train_transforms = transforms.Compose([
	transforms.Resize((128,128)),
	transforms.RandomVerticalFlip(0.5),
	transforms.RandomHorizontalFlip(0.5),
	transforms.RandomRotation(360),
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])

validation_transforms = transforms.Compose([
	transforms.Resize((128,128)),
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])


def create_loader(folder, transforms, batch_size = 4):
	dataset = torchvision.datasets.ImageFolder(
		root = DATA_PATH + folder,
		transform = transforms)
	
	loader = torch.utils.data.DataLoader(
		dataset,
		batch_size = batch_size,
		shuffle = True,
		num_workers = 2)

	return loader

