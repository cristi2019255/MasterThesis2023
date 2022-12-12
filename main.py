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
# Libraries for TensorFlow
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
import numpy as np
from utils.Logger import Logger

from utils.transformer import transform_mnist_to_tf_dataset
from utils.reader import import_mnist_dataset, load_mnist_preprocessed
from models.VGG16 import build_model, build_custom_model, compile_model
from models.tools import freeze_layers, load_model, plot, plot_history, predict, save_history, save_model

# Libraries to be deleted after refactoring
from keras.models import Model
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns


EPOCHS = 16
BATCH_SIZE = 128
MODELS_FOLDER = os.path.join("models", "model")
SAMPLES_LIMIT = 5000


def train(X_train, Y_train, X_test, Y_test):
    model = build_custom_model()
    model = freeze_layers(model, layers_to_freeze=2)
    model = compile_model(model)
    plot(model)
    
    filepath = os.path.join(MODELS_FOLDER, model.name, f"{model.name}.h5")
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', period=1)
    
    # Fit the Model
    history = model.fit(X_train,
                        Y_train,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        verbose=True,
                        validation_data=(X_test,Y_test)
                        )
    save_model(model)
    save_history(history, model.name)
    plot_history(model.name)
    
    return model, history

def get_convolution_corpus(X_train, model):
    console = Logger("Get convolution corpus")
    # remove last two layers
    model1 = Model(inputs=model.input, outputs=model.layers[-3].output)
    model1 = compile_model(model1, summary=False)
    # get convolution corpus
    console.log("Getting convolution corpus...")
    convolution_corpus = model1.predict(X_train)
    console.log(f"Convolution corpus shape: {convolution_corpus.shape}")    
    # returns a tensor of shape (batch_size, 1, 1, 512)
    # reshape to (batch_size, 512)
    convolution_corpus = convolution_corpus.reshape(convolution_corpus.shape[0], convolution_corpus.shape[3])
    console.log(f"Reshaping convolution corpus ... The new shape is: {convolution_corpus.shape}")
    
    # save convolution corpus
    if not os.path.exists(os.path.join("data", model.name)):
        os.makedirs(os.path.join("data", model.name))
    with open(os.path.join("data", model.name, "convolution_corpus.npy"), "wb") as f:
        np.save(f, convolution_corpus)
    
    return convolution_corpus

def load_convolution_corpus(model_name):
    with open(os.path.join("data", model_name, "convolution_corpus.npy"), "rb") as f:
        convolution_corpus = np.load(f)
    return convolution_corpus

def get_tsne_embedding(corpus, model_name = "MNIST"): 
    # We want to get TSNE embedding with 2 dimensions
    n_components = 2
    tsne = TSNE(n_components)
    
    tsne_result = tsne.fit_transform(corpus)
    print(tsne_result)
    # (corpus.length, 2)
    
    # save tsne embedding
    if not os.path.exists(os.path.join("data", model_name)):
        os.makedirs(os.path.join("data", model_name))
    with open(os.path.join("data", model_name, "tsne_embedding.npy"), "wb") as f:
        np.save(f, tsne_result)
    
    return tsne_result

def load_tsne_embedding(model_name):
    with open(os.path.join("data", model_name, "tsne_embedding.npy"), "rb") as f:
        tsne_embedding = np.load(f)
    return tsne_embedding

def plot_tsne_embedding(labels, model_name = "MNIST"):
    tsne_result = load_tsne_embedding(model_name)
    # Plot the result of our TSNE with the label color coded
    # A lot of the stuff here is about making the plot look pretty and not TSNE
    tsne_result_df = pd.DataFrame({'tsne_1': tsne_result[:,0], 'tsne_2': tsne_result[:,1], 'label': labels})
    fig, ax = plt.subplots(1)
    sns.scatterplot(x='tsne_1', y='tsne_2', hue='label', palette="deep", style='label', data=tsne_result_df, ax=ax,s=120)
    lim = (tsne_result.min()-5, tsne_result.max()+5)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)
    plt.show()

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
    #model, history = train(X_train, Y_train, X_test, Y_test)
    
    #predict(X_test, Y_test, model)
    
    # corpus = get_convolution_corpus(X_train, model)
    corpus = load_convolution_corpus(model.name)
    labels = np.array([np.argmax(y) for y in Y_train])
    plot_tsne_embedding(labels, model.name)
    
if __name__ == '__main__':
    main()