import pickle
import numpy as np
import random
import tensorflow as tf


# A class that utilises the cifar 10 dataset.
# It reads the path of the dataset and returns
# the data id a matrix or in batches.
#  Also forms a one hot vector for the train and test labels for easiest use with the tenesorflow models
class Cifar10Dataset:

    cifarTrainDataMatrix = []
    cifarTrainDataMatrixLabels = []
    cifarTestDataMatrix = []
    cifarTestDataMatrixLabels = []

    cifarOneHotTrain = []
    cifarOneHotTest = []

    cifarTrainData = []
    cifarTestData = []

    def __init__(self, datapath, flip, contrast, satur, hue):
        # self.data = data
        self.read_data(datapath)
        self.add_noise()
        # self.add_flip_batch(flip=flip)
        # self.add_contrast(contr=contrast)
        # self.add_saturation(saturation=satur)
        # self.add_hue(hue=hue)
        self.one_hot_train_labels()
        self.one_hot_test_labels()
        self.train_data_normalization()
        self.test_data_normalization()
        # self.train_data_batch()

    # Read the dataset as downloaded from its original source
    def read_data(self, datapath):
        X1 = self.unpickle(datapath + '/data_batch_1')
        X2 = self.unpickle(datapath + 'data_batch_2')
        X3 = self.unpickle(datapath + 'data_batch_3')
        X4 = self.unpickle(datapath + 'data_batch_4')
        X5 = self.unpickle(datapath + 'data_batch_5')
        X6 = self.unpickle(datapath + 'test_batch')
        self.cifarTrainDataMatrix = np.concatenate((X1[b'data'], X2[b'data'], X3[b'data'], X4[b'data'], X5[b'data']))
        self.cifarTrainDataMatrixLabels = np.concatenate((X1[b'labels'], X2[b'labels'], X3[b'labels'], X4[b'labels'], X5[b'labels']))
        self.cifarTestDataMatrix = X6[b'data']
        self.cifarTestDataMatrixLabels = X6[b'labels']

    # creates a one hot vector for the train labels
    def one_hot_train_labels(self):
        for i in range(50000):
            if self.cifarTrainDataMatrixLabels[i] == 1:
                self.cifarOneHotTrain.append([0., 1., 0., 0., 0., 0., 0., 0., 0., 0.])
            elif self.cifarTrainDataMatrixLabels[i] == 2:
                self.cifarOneHotTrain.append([0., 0., 1., 0., 0., 0., 0., 0., 0., 0.])
            elif self.cifarTrainDataMatrixLabels[i] == 3:
                self.cifarOneHotTrain.append([0., 0., 0., 1., 0., 0., 0., 0., 0., 0.])
            elif self.cifarTrainDataMatrixLabels[i] == 4:
                self.cifarOneHotTrain.append([0., 0., 0., 0., 1., 0., 0., 0., 0., 0.])
            elif self.cifarTrainDataMatrixLabels[i] == 5:
                self.cifarOneHotTrain.append([0., 0., 0., 0., 0., 1., 0., 0., 0., 0.])
            elif self.cifarTrainDataMatrixLabels[i] == 6:
                self.cifarOneHotTrain.append([0., 0., 0., 0., 0., 0., 1., 0., 0., 0.])
            elif self.cifarTrainDataMatrixLabels[i] == 7:
                self.cifarOneHotTrain.append([0., 0., 0., 0., 0., 0., 0., 1., 0., 0.])
            elif self.cifarTrainDataMatrixLabels[i] == 8:
                self.cifarOneHotTrain.append([0., 0., 0., 0., 0., 0., 0., 0., 1., 0.])
            elif self.cifarTrainDataMatrixLabels[i] == 9:
                self.cifarOneHotTrain.append([0., 0., 0., 0., 0., 0., 0., 0., 0., 1.])
            elif self.cifarTrainDataMatrixLabels[i] == 0:
                self.cifarOneHotTrain.append([1., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        return self.cifarOneHotTrain

    # creates a one hot vector for the test labels
    def one_hot_test_labels(self):
        for i in range(10000):
            if self.cifarTestDataMatrixLabels[i] == 1:
                self.cifarOneHotTest.append([0., 1., 0., 0., 0., 0., 0., 0., 0., 0.])
            elif self.cifarTestDataMatrixLabels[i] == 2:
                self.cifarOneHotTest.append([0., 0., 1., 0., 0., 0., 0., 0., 0., 0.])
            elif self.cifarTestDataMatrixLabels[i] == 3:
                self.cifarOneHotTest.append([0., 0., 0., 1., 0., 0., 0., 0., 0., 0.])
            elif self.cifarTestDataMatrixLabels[i] == 4:
                self.cifarOneHotTest.append([0., 0., 0., 0., 1., 0., 0., 0., 0., 0.])
            elif self.cifarTestDataMatrixLabels[i] == 5:
                self.cifarOneHotTest.append([0., 0., 0., 0., 0., 1., 0., 0., 0., 0.])
            elif self.cifarTestDataMatrixLabels[i] == 6:
                self.cifarOneHotTest.append([0., 0., 0., 0., 0., 0., 1., 0., 0., 0.])
            elif self.cifarTestDataMatrixLabels[i] == 7:
                self.cifarOneHotTest.append([0., 0., 0., 0., 0., 0., 0., 1., 0., 0.])
            elif self.cifarTestDataMatrixLabels[i] == 8:
                self.cifarOneHotTest.append([0., 0., 0., 0., 0., 0., 0., 0., 1., 0.])
            elif self.cifarTestDataMatrixLabels[i] == 9:
                self.cifarOneHotTest.append([0., 0., 0., 0., 0., 0., 0., 0., 0., 1.])
            elif self.cifarTestDataMatrixLabels[i] == 0:
                self.cifarOneHotTest.append([1., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        return self.cifarOneHotTest

    def train_data_normalization(self):
        self.cifarTrainData = ((self.cifarTrainDataMatrix / 255) - np.mean(self.cifarTrainDataMatrix / 255))
        # return self.cifarTrainData

    def test_data_normalization(self):
        self.cifarTestData = ((self.cifarTestDataMatrix / 255) - np.mean(self.cifarTestDataMatrix / 255))
        # return self.cifarTestData

    # Returns the matrix of the train data in hole
    def train_data(self):
        return self.cifarTrainData

    # Returns the labels of the train data in one hot
    def train_data_labels(self):
        return self.cifarOneHotTrain

    # Returns the matrix of the test data in hole
    def test_data(self):
        return self.cifarTestData

    # Returns the test data labels in one hot
    def test_data_labels(self):
        return self.cifarOneHotTest

    # Adds a noisy batch completely handled by the user
    def add_noise(self):
        noise = input("Whould you like to add noisy batch?  [y/n]:")
        if noise == "y":
            sess = tf.Session()
            saturation = input("Whould you like to add a saturated batch? [y/n]: ")
            if saturation == "y":
                saturation_size = input("Please enter the size of the Saturated batch: ")
                for i in range(int(saturation_size)):
                    l = random.randint(0, 49999)
                    image = tf.reshape(self.cifarTrainDataMatrix[l], [32, 32, 3])
                    noised = tf.image.random_saturation(image, 0.2, 1.4)
                    noised_tensor = tf.reshape(noised, [1, 3072])
                    self.cifarTrainDataMatrixLabels = np.append(self.cifarTrainDataMatrixLabels,
                                                                self.cifarTrainDataMatrixLabels[l])
                    self.cifarTrainDataMatrix = np.concatenate(
                        (self.cifarTrainDataMatrix, noised_tensor.eval(session=sess)))
            flip = input("Whould you like to add random flipped images batch? [y/n]:")
            if flip == "y":
                flip_size = input("Please enter the size of the Flipped batch: ")
                for i in range(int(flip_size)):
                    l = random.randint(0, 49999)
                    image = tf.reshape(self.cifarTrainDataMatrix[l], [32, 32, 3])
                    noised = tf.image.random_flip_left_right(image)
                    noised_tensor = tf.reshape(noised, [1, 3072])
                    # train_data_noised.append(noised_tensor.eval(session=sess))
                    self.cifarTrainDataMatrixLabels = np.append(self.cifarTrainDataMatrixLabels,
                                                                self.cifarTrainDataMatrixLabels[l])
                    self.cifarTrainDataMatrix = np.concatenate(
                        (self.cifarTrainDataMatrix, noised_tensor.eval(session=sess)))
            contrast = input("Whould you like to change the contrast of the pictures and add them to the data? [y/n]:")
            if contrast == "y":
                contr_size = input("Please enter the size of the Contrasted batch: ")
                for i in range(int(contr_size)):
                    l = random.randint(0, 49999)
                    image = tf.reshape(self.cifarTrainDataMatrix[l], [32, 32, 3])
                    noised = tf.image.random_contrast(image, 0.2, 1.4)
                    noised_tensor = tf.reshape(noised, [1, 3072])
                    self.cifarTrainDataMatrixLabels = np.append(self.cifarTrainDataMatrixLabels,
                                                                self.cifarTrainDataMatrixLabels[l])
                    self.cifarTrainDataMatrix = np.concatenate(
                        (self.cifarTrainDataMatrix, noised_tensor.eval(session=sess)))
            hue = input("Whould you like to add hue on the data?  [y/n]:")
            if hue == "y":
                hue_size = input("Please enter the size of the Hued batch: ")
                for i in range(int(hue_size)):
                    l = random.randint(0, 49999)
                    image = tf.reshape(self.cifarTrainDataMatrix[l], [32, 32, 3])
                    noised = tf.image.random_hue(image, 0.1)
                    noised_tensor = tf.reshape(noised, [1, 3072])
                    # train_data_noised.append(noised_tensor.eval(session=sess))
                    self.cifarTrainDataMatrixLabels = np.append(self.cifarTrainDataMatrixLabels,
                                                                self.cifarTrainDataMatrixLabels[l])
                    self.cifarTrainDataMatrix = np.concatenate((self.cifarTrainDataMatrix, noised_tensor.eval(session=sess)))

    # # Adds a randow hue batch
    # def add_hue(self, hue):
    #     if hue is True:
    #         sess = tf.Session()
    #         hue_size = input("Please enter the size of the Hued batch: ")
    #         for i in range(int(hue_size)):
    #             l = random.randint(0, 49999)
    #             image = tf.reshape(self.cifarTrainDataMatrix[l], [32, 32, 3])
    #             noised = tf.image.random_hue(image, 0.1)
    #             noised_tensor = tf.reshape(noised, [1, 3072])
    #             # train_data_noised.append(noised_tensor.eval(session=sess))
    #             self.cifarTrainDataMatrixLabels = np.append(self.cifarTrainDataMatrixLabels,
    #                                                         self.cifarTrainDataMatrixLabels[l])
    #             self.cifarTrainDataMatrix = np.concatenate(
    #                (self.cifarTrainDataMatrix, noised_tensor.eval(session=sess)))
    #
    # # Adds saturation batch
    # def add_saturation(self, saturation):
    #     if saturation is True:
    #         sess = tf.Session()
    #         saturation_size = input("Please enter the size of the Saturated batch: ")
    #         for i in range(int(saturation_size)):
    #             l = random.randint(0, 49999)
    #             image = tf.reshape(self.cifarTrainDataMatrix[l], [32, 32, 3])
    #             noised = tf.image.random_saturation(image, 0.2, 1.4)
    #             noised_tensor = tf.reshape(noised, [1, 3072])
    #             self.cifarTrainDataMatrixLabels = np.append(self.cifarTrainDataMatrixLabels,
    #                                                         self.cifarTrainDataMatrixLabels[l])
    #             self.cifarTrainDataMatrix = np.concatenate((self.cifarTrainDataMatrix, noised_tensor.eval(session=sess)))
    #
    # # Adds a flipped batch in the dataset
    # def add_flip_batch(self, flip):
    #     if flip is True:
    #         sess = tf.Session()
    #         flip_size = input("Please enter the size of the Flipped batch: ")
    #         for i in range(int(flip_size)):
    #             l = random.randint(0, 49999)
    #             image = tf.reshape(self.cifarTrainDataMatrix[l], [32, 32, 3])
    #             noised = tf.image.random_flip_left_right(image)
    #             noised_tensor = tf.reshape(noised, [1, 3072])
    #             # train_data_noised.append(noised_tensor.eval(session=sess))
    #             self.cifarTrainDataMatrixLabels = np.append(self.cifarTrainDataMatrixLabels, self.cifarTrainDataMatrixLabels[l])
    #             self.cifarTrainDataMatrix = np.concatenate((self.cifarTrainDataMatrix, noised_tensor.eval(session=sess)))
    #
    # # Adds Contrast batch
    # def add_contrast(self, contr):
    #     if contr is True:
    #         sess = tf.Session()
    #         contr_size = input("Please enter the size of the Contrasted batch: ")
    #         for i in range(int(contr_size)):
    #             l = random.randint(0, 49999)
    #             image = tf.reshape(self.cifarTrainDataMatrix[l], [32, 32, 3])
    #             noised = tf.image.random_contrast(image, 0.2, 1.4)
    #             noised_tensor = tf.reshape(noised, [1, 3072])
    #             self.cifarTrainDataMatrixLabels = np.append(self.cifarTrainDataMatrixLabels,
    #                                                         self.cifarTrainDataMatrixLabels[l])
    #             self.cifarTrainDataMatrix = np.concatenate((self.cifarTrainDataMatrix
    # , noised_tensor.eval(session=sess)))

    # Returns train data along with their labels in one hot matrix
    def train_data_batch(self, batch_size):
        train_data = []
        train_labels = []
        print("The generated dataset consists of ", len(self.cifarTrainDataMatrix), "images in array form and"
              ,  len(self.cifarTrainDataMatrixLabels), "labels in one hot vector form")
        for i in range(batch_size):
            l = random.randint(0, len(self.cifarTrainDataMatrix))
            train_data.append(self.cifarTrainData[l])
            train_labels.append(self.cifarOneHotTrain[l])
        return train_data, train_labels



    @staticmethod
    def unpickle(file):
        with open(file, 'rb') as fo:
            dictp = pickle.load(fo, encoding='bytes')
            return dictp

print(Cifar10Dataset("/Users/pantelispanka/Python/TensorflowModels/cifar10_models/cifar-10-batches-py/"
                     , True, True, True, True).train_data_batch(batch_size=10))
