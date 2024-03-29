import os

PATH = os.path.dirname(os.path.realpath(__file__))
DATA_PATH = os.path.dirname(os.path.realpath(".")) + "/" # directory below the one, where this file is located

ORIGINAL_TRAIN_PATH = DATA_PATH + "train_images/"
MODEL_PATH = DATA_PATH + "models/"
TRAIN_PATH = DATA_PATH + "train/"
VALIDATION_PATH = DATA_PATH + "validation/"
TEST_PATH = DATA_PATH + "test_images/"
EVEN_TRAIN_PATH = DATA_PATH + "train_even/"

