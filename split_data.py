import pandas as pd
import os
import shutil
import numpy as np
from _config import DATA_PATH

train_images = DATA_PATH + 'train_images/'

train_path = DATA_PATH + 'train/'

validation_path = DATA_PATH + 'validation/'

train = pd.read_csv('train.csv')

for x in range(len(train)):
	random_number = np.random.random()
	id_code = train['id_code'][x]
	image_name = id_code + '.png'

	dest = train_path

	if random_number < 0.1:
		dest = validation_path
	dest = dest + str(train['diagnosis'][x]) + '/'
	shutil.copy2(train_images + image_name, dest)
		
	