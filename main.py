from config import MODEL_PATH
from torch import nn
from train import train
from test import test
from datetime import datetime
import torch
import argparse
import torchvision.models as models






# parse arguments from command line
parser = argparse.ArgumentParser(description='Process some hyperparameters.')

parser.add_argument('stage_one_epochs', metavar='s1e', type=int,
	help='Number of of training epochs on long tailed distributed dataset.')
parser.add_argument('stage_two_epochs', metavar='s2e', type=int,
	help='Number of training epochs on evenly distributed dataset.')
# parser.add_argument('number_of_tests', metavar='t', type=int,
# 	help='Number of times, the model goes through test dataset')
parser.add_argument('learning_rate', metavar='lr', type=float,
	help='Learning rate')
parser.add_argument('batch_size', metavar='bs', type=int,
	help='Batch size')


# get hyperparameters
args = parser.parse_args()


print(type(args.batch_size))

# init model
Dense_NET = models.densenet161(pretrained = True)

# start timer
start = datetime.now()

# train on long tailed training set
trained_on_long_tailed_dataset = train(
	model = Dense_NET,
	training_data_path = "train/",
	training_epochs = args.stage_one_epochs,
	criterion = nn.CrossEntropyLoss(),
	learning_rate = args.learning_rate,
	batch_size = args.batch_size)


# save models stage_dict after first stage
torch.save(trained_on_long_tailed_dataset[0].state_dict(), MODEL_PATH + "long_tailed_" + trained_on_long_tailed_dataset[1])





# use the model from first stage and load it with the saved stage dict  
model = trained_on_long_tailed_dataset[0]
model.load_state_dict(torch.load(MODEL_PATH + "long_tailed_" + trained_on_long_tailed_dataset[1]))

# now, do finetuning on the evenly distributed training set
finetuned_model = train(
	model = model,
	training_data_path = "train_even/",
	training_epochs = args.stage_two_epochs,
	criterion = nn.CrossEntropyLoss(),	
	learning_rate = args.learning_rate,
	batch_size = args.batch_size)


torch.save(finetuned_model[0], MODEL_PATH + "finetuned_" + finetuned_model[1])



print("\nOverall training and testing time: " + str(datetime.now() - start))
