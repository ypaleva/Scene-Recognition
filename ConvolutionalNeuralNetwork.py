import cv2  # working with, mainly resizing, images
import numpy as np  # dealing with arrays
import os  # dealing with directories
from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm  # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

import tensorflow as tf


TRAIN_DIR = "C:\\cygwin64\\home\\spas2\\Coursework1\\src\\main\\java\\ss1g16\\dataset\\training"
TEST_DIR = "C:\\cygwin64\\home\\spas2\\Coursework1\\src\\main\\java\\ss1g16\\dataset\\testing"
IMG_SIZE = 200
LR = 1e-3

MODEL_NAME = 'classifier-{}-{}.model'.format(LR,IMG_SIZE)

# the image labels are stored as a vectors with 15 numbers, each one referring to a single label
def label_img(folder):
    if folder == 'bedroom':
        return [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if folder == 'Coast':
        return [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if folder == 'Forest':
        return [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if folder == 'Highway':
        return [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if folder == 'industrial':
        return [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if folder == 'Insidecity':
        return [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    if folder == 'kitchen':
        return [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    if folder == 'livingroom':
        return [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    if folder == 'Mountain':
        return [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    if folder == 'Office':
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
    if folder == 'OpenCountry':
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
    if folder == 'store':
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
    if folder == 'Street':
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    if folder == 'Suburb':
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
    if folder == 'TallBuilding':
        return [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]

# read the training dataset
def create_train_data():
    training_data = []
    # iterate over each class of images
    for folder in os.listdir(TRAIN_DIR):
        if folder != ".DS_Store":

            # iterate over each image in a class
            FOLDER_DIR = os.path.join(TRAIN_DIR, folder)
            for img in os.listdir(FOLDER_DIR):
                if img != ".DS_Store":

                    #get the label of the image and the image and put them in the training list
                    label = label_img(folder)
                    path = os.path.join(FOLDER_DIR, img)
                    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    training_data.append([np.array(img), np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data


def process_test_data():
    testing_data = []
    # iterate over the testing set and put each image in the testing list
    for img in tqdm(os.listdir(TEST_DIR)):
        path = os.path.join(TEST_DIR, img)
        img_num = img.split('.')[0]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        testing_data.append([np.array(img), img_num])

    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data


train_data = create_train_data()
# train_data = np.load('train_data.npy')

#
tf.reset_default_graph()
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 128, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 15, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')

# split the training data in 1450 for training and 50 for testing
train_set = train_data[:-50]
test_set = train_data[-50:]

# separate the data and the labels
train_x = np.array([i[0] for i in train_set]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
train_y = [i[1] for i in train_set]

# separate the data and the labels
test_x = np.array([i[0] for i in test_set]).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
test_y = [i[1] for i in test_set]

model.fit({'input': train_x}, {'targets': train_y}, n_epoch=100, validation_set=({'input': test_x}, {'targets': test_y}), snapshot_step=500, show_metric=True, run_id=MODEL_NAME)

model.save(MODEL_NAME)

test_data = process_test_data()
# test_data = np.load('test_data.npy')

with open('run3.txt', 'a') as f:
    for data in tqdm(test_data):
        img_num = data[1]
        img_data = data[0]
        orig = img_data
        data = img_data.reshape(IMG_SIZE, IMG_SIZE, 1)
        model_out = model.predict([data])[0]

        if np.argmax(model_out) == 0:
            label = 'bedroom'
        elif np.argmax(model_out) == 1:
            label = 'Coast'
        elif np.argmax(model_out) == 2:
            label = 'Forest'
        elif np.argmax(model_out) == 3:
            label = 'Highway'
        elif np.argmax(model_out) == 4:
            label = 'industrial'
        elif np.argmax(model_out) == 5:
            label = 'Insidecity'
        elif np.argmax(model_out) == 6:
            label = 'kitchen'
        elif np.argmax(model_out) == 7:
            label = 'livingroom'
        elif np.argmax(model_out) == 8:
            label = 'Mountain'
        elif np.argmax(model_out) == 9:
            label = 'Office'
        elif np.argmax(model_out) == 10:
            label = 'OpenCountry'
        elif np.argmax(model_out) == 11:
            label = 'store'
        elif np.argmax(model_out) == 12:
            label = 'Street'
        elif np.argmax(model_out) == 13:
            label = 'Suburb'
        elif np.argmax(model_out) == 14:
            label = 'TallBuilding'
        else:
            label = 'WRONG'

        f.write('{}.jpg {}\n'.format(img_num, label))

