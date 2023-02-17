# Copyright 2023 Cristian Grosu
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

from src.decision_boundary_mapper.DBM.tools import get_inv_proj_error, get_proj_error
from src.decision_boundary_mapper.utils.dataReader import import_mnist_dataset
from src.decision_boundary_mapper.GUI.DBMPlotter import DBMPlotter
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from keras.datasets import mnist
import os
from math import sqrt
from sklearn.neighbors import KDTree

SAMPLES_LIMIT = 5000

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
X_train, Y_train = X_train[:int(0.7*SAMPLES_LIMIT)], Y_train[:int(0.7*SAMPLES_LIMIT)]
X_test, Y_test = X_test[:int(0.3*SAMPLES_LIMIT)], Y_test[:int(0.3*SAMPLES_LIMIT)]

X = np.concatenate((X_train, X_test))

def transform_tsne(X: np.ndarray):
    """ Transforms the given data to 2D using t-SNE.

    Args:
        X (np.ndarray): The data to be transformed
    """
    X_flat = X.reshape(X.shape[0], -1)
    projection = TSNE(n_components=2, random_state=0, learning_rate="auto", init="random")
    X2d = projection.fit_transform(X_flat)
    return X2d

"""
X_2d = transform_tsne(X)
X_train_2d, X_test_2d = X_2d[:int(0.7*SAMPLES_LIMIT)], X_2d[int(0.7*SAMPLES_LIMIT):]

with open("models/DBM/t-SNE_train_2d.npy", "wb") as f:
    np.save(f, X_train_2d)
with open("models/DBM/t-SNE_test_2d.npy", "wb") as f:
    np.save(f, X_test_2d)
"""

with open("models/DBM/t-SNE_train_2d.npy", "rb") as f:
    X_train_2d = np.load(f)
with open("models/DBM/t-SNE_test_2d.npy", "rb") as f:
    X_test_2d = np.load(f)


print(X_train_2d.shape)

x_min = min(np.min(X_train_2d[:,0]), np.min(X_test_2d[:,0]))
y_min = min(np.min(X_train_2d[:,1]), np.min(X_test_2d[:,1]))
x_max = max(np.max(X_train_2d[:,0]), np.max(X_test_2d[:,0]))
y_max = max(np.max(X_train_2d[:,1]), np.max(X_test_2d[:,1]))
X_train_2d = (X_train_2d - np.array([x_min, y_min])) / np.array([x_max - x_min, y_max - y_min])
X_test_2d = (X_test_2d - np.array([x_min, y_min])) / np.array([x_max - x_min, y_max - y_min])
        
output_shape = (28, 28)
stopping_callback = tf.keras.callbacks.EarlyStopping(verbose=1, min_delta=0.00001, mode='min', patience=20, restore_best_weights=True)

if os.path.exists("invNN.h5"):
    #decoder = tf.keras.models.load_model("decoder.h5")
    #classifier = tf.keras.models.load_model("classifier.h5")
    invNN = tf.keras.models.load_model("invNN.h5")
else:
    print("Building model...")
    decoder = tf.keras.Sequential([
                tf.keras.layers.Dense(32, activation='relu', kernel_initializer='he_uniform', kernel_regularizer=tf.keras.regularizers.l2(0.0002), input_shape=(2, )),
                tf.keras.layers.Dense(64, activation='relu', kernel_initializer='he_uniform', bias_initializer=tf.keras.initializers.Constant(0.01)),
                tf.keras.layers.Dense(128, activation='relu', kernel_initializer='he_uniform', bias_initializer=tf.keras.initializers.Constant(0.01)),
                tf.keras.layers.Dense(512, activation='relu', kernel_initializer='he_uniform', bias_initializer=tf.keras.initializers.Constant(0.01)),
                tf.keras.layers.Dense(output_shape[0] * output_shape[1], activation='sigmoid', kernel_initializer='he_uniform'),   
                tf.keras.layers.Reshape(output_shape)         
            ], name="decoder")

    optimizer = tf.keras.optimizers.Adam()

    classifier = tf.keras.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation=tf.nn.softmax)
        ], name="classifier")

    input_layer = tf.keras.layers.Input(shape=(2, ))
    decoder_output = decoder(input_layer)
    classifier_output = classifier(decoder_output)
    
    invNN = tf.keras.Model(
        inputs = input_layer,
        outputs = [decoder_output, classifier_output],
        name="invNN")
    
    invNN.compile(optimizer='adam',
                  loss={'decoder': 'binary_crossentropy', 'classifier': 'sparse_categorical_crossentropy'},
                  loss_weights={'decoder': 1.0, 'classifier': 0.125},
                  metrics={'decoder': 'accuracy', 'classifier': 'accuracy'},
    )

    print("Training model...")
    hist = invNN.fit(X_train_2d, [X_train, Y_train], 
                        epochs=300, 
                        batch_size=32, 
                        validation_data=(X_test_2d, [X_test, Y_test]),
                        callbacks=[stopping_callback]
                        )
    invNN.save("invNN.h5")

    plt.plot(hist.history['loss'], label='loss')
    plt.plot(hist.history['val_loss'], label='val_loss')
    plt.legend()
    plt.show()

    plt.plot(hist.history['decoder_accuracy'], label='accuracy')
    plt.plot(hist.history['val_decoder_accuracy'], label='val_accuracy')
    plt.legend()
    plt.show()

"""
if os.path.exists("classifier.h5"):
    classifier = tf.keras.models.load_model("classifier.h5")
else:
    classifier = tf.keras.Sequential([
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)
    ], name="classifier")

    classifier.compile(optimizer='adam',
                    loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])

    decoded_train = decoder.predict(X_train_2d)
    decoded_test = decoder.predict(X_test_2d)

    classifier.fit(decoded_train, Y_train, 
                epochs=300, batch_size=32,
                callbacks=[stopping_callback],
                validation_data=(decoded_test, Y_test))

    print("Classifier trained")
    classifier.save("classifier.h5")
"""

def show_predictions(invNN, dataNd: np.ndarray, data2d:np.ndarray, labels: np.ndarray, n=16):
        decoded, predicted = invNN.predict(data2d)
        predicted_labels = np.argmax(predicted, axis=1)
        
        plt.figure(figsize=(20, 10))
        m = int(sqrt(n))
        
        for i in range(1,2*n,2):
            plt.subplot(m, 2*m, i)
            plt.imshow(dataNd[i], cmap='gray')
            plt.title(f"Actual: {labels[i]}", color='green' if predicted_labels[i] == labels[i] else 'red', fontsize=12)
            plt.axis('off')
            plt.subplot(m, 2*m, i+1)
            plt.imshow(decoded[i], cmap='gray')
            plt.title(f"Predicted: {predicted_labels[i]}", color='green' if predicted_labels[i] == labels[i] else 'red', fontsize=12)
            plt.axis('off')
        
        plt.show()

show_predictions(invNN, X_test, X_test_2d, Y_test, n=49)

score = invNN.evaluate(X_test_2d, [X_test, Y_test])
print(f"Test loss: {score[0]}")
print(f"Decoder loss: {score[1]}")
print(f"Classifier loss: {score[2]}")
print(f"Decoder accuracy: {score[3]}")
print(f"Classifier accuracy: {score[4]}")


resolution = 256
img = np.zeros((resolution, resolution))
img_confidence = np.zeros((resolution, resolution))

space2d = np.array([(i / resolution, j / resolution) for i in range(resolution) for j in range(resolution)])
decoded, predictions = invNN.predict(space2d)
predicted_labels = np.argmax(predictions, axis=1)
predicted_confidence = np.max(predictions, axis=1)
spaceNd = decoded.reshape((resolution, resolution, 28, 28))
img = predicted_labels.reshape((resolution, resolution))
img_confidence = predicted_confidence.reshape((resolution, resolution))
        
for [x,y] in X_train_2d:
    i, j = int(x * (resolution - 1)) , int(y * (resolution - 1))
    img[i,j] = -1
    img_confidence[i,j] = 1
for [x,y] in X_test_2d:
    i, j = int(x * (resolution - 1)) , int(y * (resolution - 1))
    img[i,j] = -2
    img_confidence[i,j] = 1


errors = np.zeros((resolution, resolution))
for i in range(resolution):
    for j in range(resolution):
        errors[i,j] = get_inv_proj_error(i,j, spaceNd)
             
                
proj_errors = np.zeros((resolution, resolution))

K = 10
metric = "euclidean"

X2d = X_train_2d.reshape((X_train_2d.shape[0], -1))
Xnd = X_train.reshape((X_train.shape[0], -1))

"""
tree = KDTree(X2d, metric=metric)
print("Created KDTree for X2d")
indices_embedded = tree.query(X2d, k=K, return_distance=False)
# Drop the actual point itself
indices_embedded = indices_embedded[:, 1:]
print("Indices embedded computed")
tree = KDTree(Xnd, metric=metric)
print("Created KDTree for Xnd")
indices_source = tree.query(Xnd, k=K, return_distance=False)
# Drop the actual point itself
indices_source = indices_source[:, 1:]
print("Indices source computed")   

for k in range(X2d.shape[0]):
    i, j = X2d[k]
    i, j = int(i * (resolution - 1)) , int(j * (resolution - 1))
    proj_errors[i,j] = get_proj_error(indices_source[k], indices_embedded[k])
"""

X_train_2d *= resolution - 1
X_test_2d *= resolution - 1
X_train_2d = X_train_2d.astype(int)
X_test_2d = X_test_2d.astype(int)
space2d *= resolution - 1
space2d = space2d.astype(int)

dbm_plotter = DBMPlotter(img = img,
                         img_confidence = img_confidence,
                         img_projection_errors = proj_errors,
                         img_inverse_projection_errors = errors,
                         num_classes = 10, 
                         encoded_train = X_train_2d, 
                         encoded_test = X_test_2d,
                         X_train = X_train,
                         Y_train = Y_train,
                         X_test = X_test,
                         Y_test = Y_test,
                         spaceNd = decoded)

dbm_plotter.plot()