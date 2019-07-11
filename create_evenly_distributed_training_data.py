from config import DATA_PATH, TEST_PATH, TRAIN_PATH, EVEN_TRAIN_PATH, ORIGINAL_TRAIN_PATH

import pandas as pd
import os
import shutil


label_lookup = pd.read_csv('train.csv')




zero_counter, one_counter, two_counter, three_counter, four_counter = 0,0,0,0,0


category_cap = 199


for i in range(len(label_lookup)):
	id_code = label_lookup['id_code'][i]
	image_name = id_code + '.png'
	diagnosis = label_lookup['diagnosis'][i]

	if (diagnosis == 0) and zero_counter <= category_cap:
		zero_counter += 1
		shutil.copy2(ORIGINAL_TRAIN_PATH + image_name, EVEN_TRAIN_PATH + str(diagnosis))
		print("copy created...")
	elif (diagnosis == 1) and one_counter <= category_cap:
		one_counter += 1
		shutil.copy2(ORIGINAL_TRAIN_PATH + image_name, EVEN_TRAIN_PATH + str(diagnosis))
		print("copy created...")
	elif (diagnosis == 2) and two_counter <= category_cap:
		two_counter += 1
		shutil.copy2(ORIGINAL_TRAIN_PATH + image_name, EVEN_TRAIN_PATH + str(diagnosis))
		print("copy created...")
	elif (diagnosis == 3) and three_counter <= category_cap:
		three_counter += 1
		shutil.copy2(ORIGINAL_TRAIN_PATH + image_name, EVEN_TRAIN_PATH + str(diagnosis))
		print("copy created...")
	elif (diagnosis == 4) and four_counter <= category_cap:
		four_counter += 1
		shutil.copy2(ORIGINAL_TRAIN_PATH + image_name, EVEN_TRAIN_PATH + str(diagnosis))
		print("copy created...")


print(zero_counter, one_counter, two_counter, three_counter, four_counter)
