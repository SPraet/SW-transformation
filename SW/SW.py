# -*- coding: utf-8 -*-
"""
SW transformation
"""

# Authors: Stiene Praet <stiene.praet@uantwerp.be>
# Reference: Stankova, M., Martens, D., & Provost, F. (2015). Classification over bipartite graphs through projection. 
# (Research paper / University of Antwerp. Faculty of Applied Economics ; 2015-001 D/2015/1169/001). 
# Full text (open access): https://repository.uantwerpen.be/docman/irua/07acff/c5909d64.pdf

import numpy as np

def _check_adjacency(X):
    return (X.data!=1).sum()!=0

def _top_sum(X):
    return np.squeeze(np.asarray(X.sum(axis=0))) + EPS

def _tanh(X):
    X_sum_0 = _top_sum(X)
    return np.tanh(1.0/X_sum_0)

def _simple(X):
    return np.ones((X.shape[1],))

def _inverse(X):
    X_sum_0 = _top_sum(X)
    return 1.0/X_sum_0

WEIGHT_FUNCTIONS = {
    "tanh": _tanh,
    "simple": _simple,
    "inverse": _inverse
}

EPS = 1e-20

class SW_transformation:
    def __init__(self, weight_function='tanh'):
        """
        Parameters
        ----------
        weight_function : 'tanh','inverse' or 'simple' (default='tanh'). 
        If you want to add your own custom weights, you can provide a function,
        that takes as input argument the input matrix X and outputs the top node weights with
        dimensions (n_top_nodes=X.shape[1],)   
        """ 
        
        self.weight_function = weight_function
           
    def get_params(self, deep=True):
        return {"weight_function": self.weight_function}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def fit(self, X, y):
        """
        Fit the model according to the given training data.
        Parameters
        ----------
        X : sparse matrix, shape (n_bottom_nodes, n_top_nodes)
            Training vector, where n_bottom_nodes is the number of bottom nodes and
            n_top_nodes is the number of top nodes.
        y : array-like, shape (n_bottom nodes, 1)
            Target vector relative to X.
        """
        if _check_adjacency(X):
          raise ValueError("Input matrix X should only contain ones or zeros")

        if callable(self.weight_function):
          top_node_weights = self.weight_function(X)
        else:
          try:
            top_node_weights = WEIGHT_FUNCTIONS[self.weight_function](X)
          except KeyError:
            raise ValueError('please enter a valid weight function: "tanh", "simple" or "inverse"')
        
        if len(top_node_weights) != X.shape[1]:
          raise ValueError('please make sure your weight_function outputs top node weights with the correct dimensions (n_top_nodes=X.shape[1],)')
           
        nsk = np.squeeze(X.T*y)
        self.coef_= np.reshape(nsk*top_node_weights,(1,X.shape[1]))

        X_sum_0 = _top_sum(X)
        self.Z_=np.reshape(top_node_weights*X_sum_0,(1,X.shape[1]))
        
        return self
      
    def predict_proba(self, X):
        """
        Probability estimates.
     
        Parameters
        ----------
        X : sparse matrix, shape = [n_bottom_nodes, n_top_nodes]
        Returns
        -------
        pred_scores : array-like, shape = [n_bottom_nodes,2]
            Returns the probability of the bottom node for the negative and positive class.
        """
        if _check_adjacency(X):
          raise ValueError("Input matrix X should only contain ones or zeros")
        
        top_node_sum= X*self.coef_.T
        Z_sum=X*self.Z_.T
        scores = np.divide(top_node_sum, Z_sum+EPS)
        scores = np.hstack((1.0-scores,scores))
        return scores
