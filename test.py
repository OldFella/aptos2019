import torch
import torch.nn as nn
import torch.nn.functional as F
from _config import MODEL_PATH

from load_data import create_loader, validation_transforms


device = torch.device("cpu")
test_loader = create_loader('validation/', validation_transforms)

def test(test_loader, model, device = device):
	""" 
	Loads the test data set and tests the model on it. 
	Returns achieved loss, correctly guessed samples and accuracy.
	"""
	model.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)
			output = model(data)
			test_loss += F.nll_loss(output, target, reduction = 'sum').item()
			pred = output.argmax(dim = 1, keepdim = True)
			correct += pred.eq(target.view_as(pred)).sum().item()

		test_loss /= len(test_loader.dataset)
		accuracy_percent = 100 * correct/len(test_loader.dataset)
		print('\nTest set: Average loss:  {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(test_loader.dataset), accuracy_percent))

	return test_loss, correct, len(test_loader.dataset), accuracy_percent
