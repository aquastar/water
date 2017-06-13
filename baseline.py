import csv
import random

import numpy as np
from keras.datasets import mnist
from keras.layers import Input, Dense
from keras.models import Model

# my_data = genfromtxt('data.csv', delimiter=',')

all_data = []
with open('data.csv', 'rb') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    for row in spamreader:
        tmp = row[1].split()[0].split('/')
        row[1] = (int(tmp[2]) - 1996) * 10000 + int(tmp[0]) * 100 + int(tmp[1])
        all_data.append([row[0], row[1], row[3], row[5], row[6], row[11], row[13], row[16]])

data_size = len(all_data)
fold_n = 10
fold_size = data_size / fold_n

all_data = np.array(all_data)

# clothes washer
clothes = all_data[:, (0, 1, 2)]
# dishwasher
dishwasher = all_data[:, (0, 1, 3)]
# faucet
faucet = all_data[:, (0, 1, 4)]
# shower
shower = all_data[:, (0, 1, 5)]
# toilet
toilet = all_data[:, (0, 1, 6)]
# whole-home
whole = all_data[:, (0, 1, 7)]

# homogeneous
# heterogeneous
training_index = random.sample(xrange(0, data_size), fold_size)

# this is the size of our encoded representations
encoding_dim = 32  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

# this is our input placeholder
input_img = Input(shape=(784,))
# "encoded" is the encoded representation of the input
encoded = Dense(encoding_dim, activation='relu')(input_img)
# "decoded" is the lossy reconstruction of the input
decoded = Dense(784, activation='sigmoid')(encoded)

# this model maps an input to its reconstruction
autoencoder = Model(input_img, decoded)

# this model maps an input to its encoded representation
encoder = Model(input_img, encoded)

# create a placeholder for an encoded (32-dimensional) input
encoded_input = Input(shape=(encoding_dim,))
# retrieve the last layer of the autoencoder model
decoder_layer = autoencoder.layers[-1]
# create the decoder model
decoder = Model(encoded_input, decoder_layer(encoded_input))

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

(x_train, _), (x_test, _) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print x_train.shape
print x_test.shape

autoencoder.fit(x_train, x_train,
                epochs=50,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, x_test))

encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)
