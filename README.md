# SW-transformation
The SW-transformation is a fast classifier for binary node classification in bipartite graphs ([Stankova et al., 2015](https://repository.uantwerpen.be/docman/irua/07acff/c5909d64.pdf)).
![title](https://github.com/SPraet/SW-transformation/blob/master/Bigraph.PNG)
It combines the weighted-vote Relational Neighbor (wvRN) classifier with an aggregation function that sums the weights of the top nodes. The transformation optimally considers for each test instance only the weights of the neighboring top nodes (where xik = 1 in the bigraph adjacency matrix) multiplied by the number of training instances in that column which have a positive label (the positive neighbors of the node). The SW-transformation yields very fast run times and allows easy scaling of the method to big data sets of millions of nodes. 

## Installation
To build and install on your local machine, download and unzip the repository and run:

```
python setup.py install
```

If you have pip, you can automatically download and install from the PyPI repository:

```
pip install sw-transformation
```
## Usage

### Parameters
* weight_function : 'tanh', 'inverse', 'simple' or 'own' (default='tanh'). 
  For 'own', the top node weights should be provided in weights
* weights : array-like, shape (1, n_top_nodes). Vector containing the top
  node weights, where n_top_nodes is the number of top nodes. When 
  weight_function is set at 'own', the top node weights should be 
  provided here. 

### Examples
```
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import SW
import numpy as np

matrix = load_adjacency_matrix_here
label = load_node_labels_here

#split your data in training and testing
X_train, X_test, y_train, y_test = train_test_split(matrix['matrix'], label['label'], test_size=1/3, shuffle=False)

sw = SW_transformation(weight_function='inverse')
sw.fit(X_train, y_train)
pred_scores = sw.predict_proba(X_test)

auc = roc_auc_score(y_test, pred_scores)
```
### Methods
* fit(X, y)  -  Fit the model according to the given training data.
* predict_proba(X)  -  Probability estimates.

## Authors

* Stiene Praet
* Marija Stankova

## License

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/SPraet/SW-transformation/blob/master/LICENSE) file for details
