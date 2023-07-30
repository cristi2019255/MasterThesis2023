# Short Description

This is the package provides functionality for visualizing the classifiers decision boundaries.

It is based on the work of Cristian Grosu for the master thesis project for 2023 at Utrecht University.
If you use this package, please cite the following paper:
`[PLACEHOLDER FOR THE PAPER](http://google.com/)`

The package is available on PyPI and can be installed using pip: `pip install decision-boundary-mapper`

## Documentation

See more details at `https://decisionboundarymapper.000webhostapp.com/`

## Usage exmaples

1. This package comes with a simple GUI that allows you to visualize the decision boundaries of a classifier. The GUI is based on the `PySimpleGUI` package and can be started by running the following code:

```python
from decision_boundary_mapper import GUI

GUI().start()
```

2. The package comes with two examples of complete pipelines for visualizing the decision boundaries of a classifier.
Both examples use `MNIST` (handwritten digits) dataset.
The first example `DBM_usage_example` uses `t-SNE` to project the data from the `nD` space to the `2D` space, then neural network is trained to fit the inverse projection from `2D` to `nD` and the decision boundaries are visualized using the `2D` projection. The second example `SDBM_usage_example` uses a neural network with an autoencoder architecture to learn the projection and the inverse projection. After which a simple classifier is used to color each point of the `2D` projection.
The examples can be found in the `examples` folder.

```python
from decision_boundary_mapper import DBM_usage_example, SDBM_usage_example

DBM_usage_example() # run the first example
SDBM_usage_example() # run the second example
```

1. The package main functionality comes in two classes `DBM` (i.e. learns inverse projection when a 2D projection is given) and `SDBM` (i.e. learns both the projection and the inverse projection).
The classes can be used as follows:

```python
from decision_boundary_mapper import DBM, SDBM
from matplotlib import pyplot as plt

# load the data
...
X_train, X_test, y_train, y_test = load_data() 
...
# create a simple neural network
...
classifier = ... # for compatibility with the package the classifier should be constructed using tensorflow.keras
...

dbm = DBM(classifier) # create a DBM object
img, img_confidence, _, _ = dbm.generate_decision_boundary(X_train, y_train, X_test, y_test, resolution = 256) # generate the decision boundary

sdbm = SDBM(classifier) # create a SDBM object
img, img_confidence, _, _ = sdbm.generate_decision_boundary(X_train, y_train, X_test, y_test, resolution = 256) # generate the decision boundary
...
# visualize the decision boundaries
plt.imshow(img)
plt.show()

```

Created by Cristian Grosu for the master thesis project for 2023 at Utrecht University
