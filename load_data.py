import torch
import torchvision
import torchvision.transforms as transforms

PATH = '/home/gwent/projects/kaggle_comp/aptos2019-blindness-detection/'


train_transforms = transforms.Compose([
	transforms.Resize((150,150)),
	transforms.RandomVerticalFlip(0.5),
	transforms.RandomHorizontalFlip(0.5),
	transforms.RandomRotation(360),
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])

validation_transforms = transforms.Compose([
	transforms.Resize((150,150)),
	transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
	])


def create_loader(folder, transforms, batch_size = 4):
	dataset = torchvision.datasets.ImageFolder(
		root = PATH + folder,
		transform = transforms)
	
	loader = torch.utils.data.DataLoader(
		dataset,
		batch_size = batch_size,
		shuffle = True,
		num_workers = 2)

	return loader

