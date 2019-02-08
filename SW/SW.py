# -*- coding: utf-8 -*-
"""
SW transformation
"""

# Authors: Stiene Praet <stiene.praet@uantwerp.be>
# Reference: Stankova, M., Martens, D., & Provost, F. (2015). Classification over bipartite graphs through projection. 
# (Research paper / University of Antwerp. Faculty of Applied Economics ; 2015-001 D/2015/1169/001). 
# Full text (open access): https://repository.uantwerpen.be/docman/irua/07acff/c5909d64.pdf

import numpy as np

class SW_transformation:
    def __init__(self, weight_function='tanh', weights=None):
        """
        Parameters
        ----------
        weight_function : 'tanh','inverse','simple' or 'own' (default='tanh'). 
        For 'own', the top node weights should be provided in weights
           
        weights : array-like, shape (1, n_top_nodes). Vector containing the top
        node weights, where n_top_nodes is the number of top nodes. When 
        weight_function is set at 'own', the top node weights should be 
        provided here. 
        """
    
        self.weight_function = weight_function
        self.weights = weights
            
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
        if self.weight_function == 'tanh':
            top_node_weights = np.tanh(1/np.sum(X, axis=0)+1e-20)
        elif self.weight_function == 'simple':
            top_node_weights = np.ones((1,X.shape[1]))
        elif self.weight_function == 'inverse':
            top_node_weights = 1/(np.sum(X,axis=0)+1e-20)
        elif self.weight_function == 'own':
            top_node_weights = self.weights
          
        else:
            raise Exception('please enter a valid weight function: "tanh", "simple", "inverse" or "own" ')
            
        nsk = X.T*y
        try:
            self.top_node_scores= np.array(nsk.T)*np.array(top_node_weights)
        except:
            raise Exception('please enter top node weights with correct dimensions (1, n_top_nodes)')
        self.Z=np.array(top_node_weights)*np.array(np.sum(X,axis=0)+1e-20)
        return self
       
        
    def predict_proba(self, X):
        """
        Probability estimates.
     
        Parameters
        ----------
        X : sparse matrix, shape = [n_bottom_nodes, n_top_nodes]
        Returns
        -------
        pred_scores : array-like, shape = [n_bottom_nodes,1]
            Returns the probability of the bottom node for the positive class.
        """

        top_node_sum= X*self.top_node_scores.T
        Z_sum=X*self.Z.T
        pred_scores= np.divide(top_node_sum, Z_sum+1e-20)
        return pred_scores
