import numpy as np
from math import log
import collections

def probability(x):
    """
    Probability distribution of random variable x
    """
    coll = collections.Counter(x)
    n = len(x)
    for key, value in coll.most_common():
        coll[key] = value / n
    return coll

def joint_entropy(X):
    """
    Joint entropy of every two pairs of features
    H(X,Y) = - sum sum p(x,y) log p(x,y)
    """
    n, m = X.shape
    H = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            prob_ij = probability(list(zip(X[:, i], X[:, j])))
            for p in prob_ij.values():
                H[i, j] += -p * log(p)
    return H

def mutual_information(X, y):
    """
    Mutual information of every feature with labels
    I(X;Y) = sum sum p(x,y) log (p(x,y) / p(x) p(y))
    """
    n, m = X.shape
    I = np.zeros((m, ))
    for i in range(m):
        prob_iy = probability(list(zip(X[:, i], y)))
        prob_i = probability(X[:, i])
        prob_y = probability(y)
        for (a, b), p_iy in prob_iy.items():
            I[i] += p_iy * log(p_iy / (prob_i[a] * prob_y[b]))
    return I


def conditional_mutual_information(X, y):
    """
    Conditional mutual information of every feature and labels with every other feature
    I(X;Y|Z) = sum sum sum p(x,y,z) log(p(z) p(x,y,z) / p(x,z) p(y,z))
    """
    n, m = X.shape
    I = np.zeros((m, m))
    for i in range(m):
        for j in range(m):
            prob_iyj = probability(list(zip(X[:, i], y, X[:, j])))
            prob_j = probability(X[:, j])
            prob_ij = probability(list(zip(X[:, i], X[:, j])))
            prob_yj = probability(list(zip(y, X[:, j])))
            for (a, b, c), p_iyj  in prob_iyj.items():
                I[i, j] += p_iyj * log(p_iyj * prob_j[c] / (prob_yj[(b, c)] * prob_ij[(a, c)]))
    return I

class Massive():
    def DISR_matrix(self, X, y):
        """
        Double input symmetrical relevance matrix W
        Wij = I(Xi,j;Y) / H(Xi,j)
        I(Xi,j;Y) = I(Xi;Y) + I(Xj;Y|Xi)
        """
        H = joint_entropy(X)
        I_i = mutual_information(X, y)
        I = conditional_mutual_information(X, y)
        I = I + I_i
        W = I / H
        np.fill_diagonal(W, 0)
        return W

    def run(self, X, y, k, backward):
        """
        Filter for feature selection in microarray data characterized by a large number of
        input variables and a few samples. All variables must be discrete.
        Uses Double Input Symmetrical Relevance selection.

        Parameters
        ----------
        X : numpy array, shape (n_samples, n_features)
            The input samples.
        y : numpy array, shape (n_samples, )
            The classes for the samples.
        k : int
            Number of features to choose.
        backward : boolean
            0 - for forward selection
            1 - for backward elimination

        Returns
        -------
        List of selected features.
        For forward selection in order of selecting.

        See Also
        --------
        http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.318.6576&rep=rep1&type=pdf

        Examples
        --------
        X = np.random.randint(10, size=(100, 20))
        y = np.random.randint(10, size=100)
        print(Massive().run(X, y, 10, False))
        print(Massive().run(X, y, 10, True))

        """
        m = X.shape[1]
        W = self.DISR_matrix(X, y)
        features = []
        selected = np.empty((m,))
        selected.fill(backward)

        if backward:
            k = m - k
        else:
            k -= 1
            I = mutual_information(X, y)
            features.append(np.argmax(I))
            selected[np.argmax(I)] = 1
        
        for i in range(k):
            best_score = 0
            best_score_index = -1
            for j in range(m):
                if selected[j] == backward:
                    #j - now selected/unselected
                    score = W[j].dot(selected)
                    if best_score_index == -1 or\
                    (backward and score < best_score) or\
                    ((not backward) and score > best_score):
                        best_score = score
                        best_score_index = j
            selected[best_score_index] = 1 - backward
            features.append(best_score_index)

        if backward:
            features = []
            for i in range(m):
                if selected[i]:
                    features.append(i)
        return features
        