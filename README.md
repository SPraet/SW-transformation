# SW-transformation
The SW-transformation is a fast classifier for binary node classification in bipartite graphs ([Stankova et al., 2015](
https://hdl.handle.net/10067/1274850151162165141)). Bipartite graphs (or bigraphs), are defined by having two types of nodes such that edges only exist between nodes of the different type (see Fig. 1). 

![title](https://github.com/SPraet/SW-transformation/blob/master/Bigraph.PNG)

**Fig. 1: Bigraph, top node projection and bottom node projection (left), adjacency matrix representation of the bigraph (right)**([Stankova et al., 2015, p. 8](
https://hdl.handle.net/10067/1274850151162165141)).

The SW-transformation combines the weighted-vote Relational Neighbor (wvRN) classifier with an aggregation function that sums the weights of the top nodes. The transformation optimally considers for each test instance only the weights of the neighboring top nodes multiplied by the number of training instances in that column which have a positive label (the positive neighbors of the node). The SW-transformation yields very fast run times and allows easy scaling of the method to big data sets of millions of nodes.

## Installation
To build and install on your local machine, download and unzip the repository and run from there:

```
python setup.py install
```

If you have pip, you can automatically download and install from the PyPI repository:

```
pip install SW-transformation
```
## Usage
The SW transformation can be used to calculate the probability of a node in a bipartite graph to belong to the positive class. Three top node weight functions are included: simple weight assignment, inverse degree and hyperbolic tangent. Users can also directly insert the top node weights, using their own weight function.

**SW-transformation**(weight_function='tanh', weights=None)

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
import numpy as np
import SW

matrix = load_adjacency_matrix_here #sparse matrix
label = load_node_labels_here

#split your data in training and testing
X_train, X_test, y_train, y_test = train_test_split(matrix['matrix'], label['label'], test_size=1/3, shuffle=False)

sw = SW.SW_transformation(weight_function='inverse')
sw.fit(X_train, y_train)
pred_scores = sw.predict_proba(X_test)

auc = roc_auc_score(y_test, pred_scores)
```
### Methods
* fit(X, y)  -  Fit the model according to the given training data.
* predict_proba(X)  -  Probability estimates.

X: sparse matrix, shape(n_bottom_nodes, n_top_nodes): the adjacency matrix (see Fig. 1, right)

y: array_like, shape(n_bottom_nodes, 1): the binary class labels (0 for negative and 1 for positive instance)

## Authors

Stiene Praet <stiene.praet@uantwerp.be>

## License

This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/SPraet/SW-transformation/blob/master/LICENSE) file for details

## Acknowledgements
Based on the work of Marija Stankova, David Martens and Foster Provost

## References
Stankova, M., Martens, D., & Provost, F. (2015). Classification over bipartite graphs through projection. (Research paper / University of Antwerp. Faculty of Applied Economics ; 2015-001 D/2015/1169/001). Full text (open access): https://repository.uantwerpen.be/docman/irua/07acff/c5909d64.pdf


