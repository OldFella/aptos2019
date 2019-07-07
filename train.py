import torch
import torch.optim as optim

from torch import nn
from load_data import create_loader, train_transforms, validation_transforms
import torchvision
from _config import MODEL_PATH
import copy
from utils import progress
import numpy as np
# import visualize
from test import test



def train(model = None, criterion = None, training_epochs = 4, batch_size = 32, learning_rate = 0.001):
	""" 
	Runs through the training data, makes a prediction and computes loss, then backpropagates
	the result through the model and adjusts the weights and biases until a local minimum in the loss
	function is reached.
	"""

	# optimizer searches fo a local minimum of in the lossfunction with different input parameters
	optimizer = optim.Adam(model.parameters(), lr = learning_rate)
	graph_loss = []
	graph_accuracy = [(0,0)]
	graph_validation_loss = []
	validation_loader = create_loader('validation/', validation_transforms, batch_size = 16)
	best_model = None

	threshold = 0
	for epoch in range(training_epochs):
		running_loss = 0.0

		training_loader = create_loader('train/', train_transforms, batch_size = batch_size)
		average_loss = 0
		print('')
		number_of_files = len(training_loader.dataset)
		for i, data in enumerate(training_loader, 0):

			# get input for training
			inputs, labels = data

			# init optimizer with 0
			optimizer.zero_grad()

			# rung data trough net
			outputs = model(inputs)

			# compute loss (compare output to label)
			loss = criterion(outputs, labels)

			# backpropagate loss
			loss.backward()

			# tweak parameters
			optimizer.step()

			# add loss to overall loss
			running_loss += loss.item()

			# pretty print progress
			if i % 10 == 9:  # append a the average of the last 10 losses as point to the loss/epoch graph_loss
				average_loss = running_loss/10
				graph_loss.append((epoch + i/(number_of_files/batch_size), average_loss))
				running_loss = 0.0
				

			# Progress bar
			progress(i, number_of_files/batch_size, epoch + 1, '{}/{:.0f} Loss: {:.2f}'.format(i, number_of_files/batch_size, average_loss))


		model_name = 'epoch{}.pt'.format(epoch)
		torch.save(best_model, MODEL_PATH + model_name)
		print("\nmodel: " + model_name + " has been saved.")
		# Validate the result of the epoch
		test_loss, correct, dataset_size, accuracy_percent = test(validation_loader, model)
		graph_accuracy.append((epoch + 1, accuracy_percent/100))
		graph_validation_loss.append((epoch + 1, test_loss))

		
	return best_model