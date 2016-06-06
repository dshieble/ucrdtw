__author__ = "dan"

import numpy as np
import _ucrdtw

class DTWNeighborsClassifier():
    """
        This class provides an sklean-like interface to the ucr-dtw suite
    """
    def fit(self, X, y):
        """
            The method fits the model
            
            Args:
                Xarr (ndarray<float>): The data to train on
                y (ndarray<int>): The labels of the data
        """
        X = np.array(X)
        self.sequences = np.hstack((X, np.zeros(X.shape))).ravel()
        self.labels = y

    def predict(self, X):       
        """
            The method predicts over the input data
            
            Args:
                Xarr (ndarray<float>): The data to predict over

            Returns:
                (ndarray<int>) The predictions over the data
        """
        X = np.array(X)
        out = []
        for i in range(X.shape[0]):
            loc, dist = _ucrdtw.ucrdtw(self.sequences, X[i,:], 0.05, True)
            out.append(self.labels[np.floor(loc/(2.0*X.shape[1]))])
        return out
    
    
    
    