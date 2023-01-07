# Copyright 2022 Cristian Grosu
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

from utils.opf import opf
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # uncomment to disable GPU, run on CPU only 
import tensorflow as tf 
from keras import backend as K

import numpy as np
from utils.transformer import transform_mnist_to_tf_dataset
from utils.reader import import_mnist_dataset, load_mnist_preprocessed
from models.VGG16 import load_convolution_corpus, train
from models.tools import load_model, predict
from utils.tsneTools import load_tsne_embedding, plot_tsne_embedding
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

SAMPLES_LIMIT = 5000
DBM_RESOLUTION = 200

def DBM(X_train, Y_train, X_test, Y_test, resolution = DBM_RESOLUTION):
    # DBM - decision boundary mapping
    # Input: X_train, Y_train, X_test, Y_test - MNIST dataset
    # Output: A 2D image (resolution x resolution) where each pixel represents a class 
    
    # Steps: 
    # 1. Train a model on the training data say Logistic Regression
    # 2. Train an autoencoder on the training data (this will be used to reduce the dimensionality of the data) nD -> 2D
    # 3. For each pixel in the 2D image, predict the class of the pixel using the trained autoencoder, the decoder part
    # 4. Generate the 2D image
    # 5. Plot the 2D image
    # 6. Enjoy!
    
    # 1. Train a model on the training data say Logistic Regression
    num_classes = np.unique(Y_train).shape[0]
    
    """
    classifier = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)
    ])
    classifier.compile(optimizer='adam',
                       loss='sparse_categorical_crossentropy',
                       metrics=['accuracy'])
    
    classifier.build(input_shape=(None, 28, 28))
    #classifier.summary()
    
    # skiping training for now
    classifier.fit(X_train, Y_train, epochs=10)
    classifier.save("model/DBM/MNIST/classifier.h5")
    classifier = tf.keras.models.load_model("models/model/DBM/MNIST/classifier.h5")
    
    """
    
    # 2. Train an autoencoder on the training data (this will be used to reduce the dimensionality of the data) nD -> 2D
    data_shape = X_train.shape[1:]
    print(f"Data shape: {data_shape}")
    
    """
    classifier = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax)
    ])
    
    encoder = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu', bias_initializer=tf.keras.initializers.Constant(10e-4)),
        tf.keras.layers.Dense(128, activation='relu', bias_initializer=tf.keras.initializers.Constant(10e-4)),
        tf.keras.layers.Dense(32, activation='relu', bias_initializer=tf.keras.initializers.Constant(10e-4)),
        tf.keras.layers.Dense(2, activation='linear', activity_regularizer=tf.keras.regularizers.l2(1/2), bias_initializer=tf.keras.initializers.Constant(10e-4)),
    ], name="encoder")
    
    decoder = tf.keras.models.Sequential([
        tf.keras.layers.Dense(32, activation='relu', bias_initializer=tf.keras.initializers.Constant(10e-4)),
        tf.keras.layers.Dense(128, activation='relu', bias_initializer=tf.keras.initializers.Constant(10e-4)),
        tf.keras.layers.Dense(512, activation='relu', bias_initializer=tf.keras.initializers.Constant(10e-4)),
        tf.keras.layers.Dense(data_shape[0] * data_shape[1], activation='sigmoid'),
        tf.keras.layers.Reshape(data_shape)
    ], name="decoder")       
    
    autoencoder_classifier = tf.keras.models.Sequential([
        encoder,
        decoder,
        classifier,
    ], name="autoencoder_classifier")
   
    auto_encoder = tf.keras.models.Sequential([
        encoder,
        decoder,
    ], name="auto_encoder")
    
    input = tf.keras.Input(shape=data_shape)
    auto_encoder_classifier_output = autoencoder_classifier(input)
    auto_encoder_output = auto_encoder(input)
    
    autoencoder = tf.keras.models.Model(inputs=input, outputs=[auto_encoder_output, auto_encoder_classifier_output], name="autoencoder")
    autoencoder.compile(optimizer='adam', loss={"auto_encoder":"binary_crossentropy",
                                               "autoencoder_classifier": "sparse_categorical_crossentropy"}, 
                        metrics=['accuracy'])
    
    autoencoder.summary()

    autoencoder.fit(X_train, {"auto_encoder": X_train, 
                              "autoencoder_classifier":Y_train}, 
                    epochs=10, batch_size=128)
    
    auto_encoder.save("models/model/DBM/MNIST/auto_encoder.h5")
    autoencoder_classifier.save("models/model/DBM/MNIST/autoencoder_classifier.h5")
    autoencoder.save("models/model/DBM/MNIST/autoencoder.h5")
    decoder.save("models/model/DBM/MNIST/decoder.h5")
    encoder.save("models/model/DBM/MNIST/encoder.h5")
    classifier.save("models/model/DBM/MNIST/classifier.h5")
    """
    
    autoencoder_classifier = tf.keras.models.load_model("models/model/DBM/MNIST/autoencoder_classifier.h5")
    auto_encoder = tf.keras.models.load_model("models/model/DBM/MNIST/auto_encoder.h5")
    classifier = tf.keras.models.load_model("models/model/DBM/MNIST/classifier.h5")
    autoencoder = tf.keras.models.load_model("models/model/DBM/MNIST/autoencoder.h5")
    decoder = tf.keras.models.load_model("models/model/DBM/MNIST/decoder.h5")
    encoder = tf.keras.models.load_model("models/model/DBM/MNIST/encoder.h5")
    
    decoded = auto_encoder.predict(X_train)
    decoded_labels = autoencoder_classifier.predict(X_train)
    
    plt.figure(figsize=(20, 6))
    for i in range(1,20,2):
        plt.subplot(2, 10, i)
        plt.imshow(X_train[i], cmap='gray')
        plt.axis('off')
        plt.subplot(2, 10, i+1)
        plt.imshow(decoded[i], cmap='gray')
        plt.title(np.argmax(decoded_labels[i]))
        plt.axis('off')
    
    plt.show()
    
    encoded = encoder.predict(X_train)
    print("Encoded: ", encoded)
    plt.plot(encoded[:,0], encoded[:,1], 'ro')
    plt.show()
    
    min_i, max_i = np.min(encoded[:,0]), np.max(encoded[:,0])
    min_j, max_j = np.min(encoded[:,1]), np.max(encoded[:,1])
    
    # 3. For each pixel in the 2D image, predict the class of the pixel using the trained autoencoder, the decoder part
    # 4. Generate the 2D image
    x = np.array([(i / resolution * (max_i - min_i) + min_i, j / resolution * (max_j - min_j) + min_j) for i in range(resolution) for j in range(resolution)])
    xNd = decoder.predict(x)
    predictions = classifier.predict(xNd)
    #print("Predictions: ", predictions)
    predicted_labels = np.array([np.argmax(p) for p in predictions])
    #print("Predicted labels: ", predicted_labels)
    img = predicted_labels.reshape((resolution, resolution))
    print("2D boundary mapping: ", img)

    # 5. Plot the 2D image
    plt.title("2D boundary mapping")
    plt.axis('off')
    values = np.unique(img)
    im =  plt.imshow(img, cmap='rainbow')
    colors = [ im.cmap(im.norm(value)) for value in values]
    patches = [ mpatches.Patch(color=colors[i], label=f"Value region: {values[i]}") for i in range(len(values)) ]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )

    plt.show()
    
    # 6. Enjoy!
    return img
    
def main():
    # import MNIST dataset
    #(X_train, Y_train), (X_test, Y_test) = import_mnist_dataset()
    
    # load preprocessed dataset
    (X_train, Y_train), (X_test, Y_test) = load_mnist_preprocessed()
    
    # limiting to first 1000 samples for testing
    X_train = X_train[:int(0.7 * SAMPLES_LIMIT)]
    Y_train = Y_train[:int(0.7 * SAMPLES_LIMIT)]
    X_test = X_test[:int(0.3 * SAMPLES_LIMIT)]
    Y_test = Y_test[:int(0.3 * SAMPLES_LIMIT)]
    
    #(X_train, Y_train), (X_test, Y_test) = transform_mnist_to_tf_dataset(X_train, Y_train, X_test, Y_test)
    
    
    model = load_model("MNIST")
    # model, history = train(X_train, Y_train, X_test, Y_test)
    
    #predict(X_test, Y_test, model)
    
    # corpus = get_convolution_corpus(X_train, model)
    # corpus = load_convolution_corpus(model.name)
    labels = np.array([np.argmax(y) for y in Y_train])
    #plot_tsne_embedding(labels, model.name)
    X = load_tsne_embedding(model.name)
    Y = labels
    Y = opf(X, Y)
    print(Y.shape)
    Y_train = to_categorical(Y)
    model, history = train(X_train, Y_train, X_test, Y_test)
    

def main_dbm():
    # import MNIST dataset
    (X_train, Y_train), (X_test, Y_test) = import_mnist_dataset()
    
    # load preprocessed dataset
    #(X_train, Y_train), (X_test, Y_test) = load_mnist_preprocessed()
    
    # limiting to first 1000 samples for testing
    X_train = X_train[:int(0.7 * SAMPLES_LIMIT)]
    Y_train = Y_train[:int(0.7 * SAMPLES_LIMIT)]
    X_test = X_test[:int(0.3 * SAMPLES_LIMIT)]
    Y_test = Y_test[:int(0.3 * SAMPLES_LIMIT)]
    
    
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    
    DBM(X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test, resolution=DBM_RESOLUTION)
    
if __name__ == '__main__':
    main_dbm()
        