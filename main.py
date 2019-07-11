import os
import torch
from config import MODEL_PATH

from torch import nn
from train import train
from test import test
from datetime import datetime
import torchvision.models as models


# init model
Dense_NET = models.densenet161(pretrained = True)



# set hyperparameters
TRAINING_EPOCHS = 1
NUMBER_OF_TESTS = 2
LEARNING_RATE = 0.0007
BATCH_SIZE = 4

# start timer
start = datetime.now()

# train
trained_on_long_tailed_dataset = train(
	model = Dense_NET,
	training_data_path = "train/",
	criterion = nn.CrossEntropyLoss(),
	training_epochs = TRAINING_EPOCHS,
	learning_rate = LEARNING_RATE,
	batch_size = BATCH_SIZE)




torch.save(trained_on_long_tailed_dataset[0].state_dict(), MODEL_PATH + "long_tailed_" + trained_on_long_tailed_dataset[1])



model = trained_on_long_tailed_dataset[0]

model.load_state_dict(torch.load(MODEL_PATH + "long_tailed_" + trained_on_long_tailed_dataset[1]))
# model.eval()

finetuned_model = train(
	model = model,
	training_data_path = "train_even/",
	criterion = nn.CrossEntropyLoss(),
	training_epochs = 3,
	learning_rate = LEARNING_RATE,
	batch_size = BATCH_SIZE)



torch.save(finetuned_model[0], MODEL_PATH + "finetuned_" + finetuned_model[1])




print("\nOverall training and testing time: " + str(datetime.now() - start))
