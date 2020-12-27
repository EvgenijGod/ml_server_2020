import numpy as np
from sklearn.tree import DecisionTreeRegressor
from scipy.optimize import minimize_scalar
import time


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


class RandomForestMSE:
    def __init__(self, n_estimators=20, max_depth=None, feature_subsample_size=None,
                 **trees_parameters):
        """
        n_estimators : int
            The number of trees in the forest.

        max_depth : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree. If None then use recommendations.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.trees_parameters = trees_parameters

    def params_dict(self):
        ans = {"n_estimators": self.n_estimators,
               "max_depth": self.max_depth,
               "feature_subsample_size": self.feature_subsample_size,
               #"trees_parameters": self.trees_parameters
               }
        return ans

    def fit(self, X, y, verbose=False, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects
        """
        if self.feature_subsample_size is None:
            self.feature_subsample_size = X.shape[1]

        inds = np.arange(X.shape[0])
        self.trees_ensemble = []

        x_res = []
        y_res = []

        for i in range(self.n_estimators):
            algo = DecisionTreeRegressor(max_features=self.feature_subsample_size,
                                         max_depth=self.max_depth,
                                         **self.trees_parameters)
            sample = np.random.choice(inds, len(inds) // 3)
            algo.fit(X[sample], y[sample])

            self.trees_ensemble.append(algo)

            if verbose:
                x_res.append(i)
                pred = self.predict(X_val)

                y_res.append(rmse(y_val, pred))

        if verbose:
            return x_res, y_res

    def predict(self, X):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        ans = np.zeros_like(X[:, 0])
        for algo in self.trees_ensemble:
            ans += algo.predict(X)

        ans /= len(self.trees_ensemble)

        return ans


class GradientBoostingMSE:
    def __init__(self, n_estimators, learning_rate=0.1, max_depth=5, feature_subsample_size=None,
                 **trees_parameters):
        """
        n_estimators : int
            The number of trees in the forest.

        learning_rate : float
            Use learning_rate * gamma instead of gamma
        max_depth : int
            The maximum depth of the tree. If None then there is no limits.

        feature_subsample_size : float
            The size of feature set for each tree. If None then use recommendations.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.feature_subsample_size = feature_subsample_size
        self.trees_parameters = trees_parameters
        self.history = None

    def params_dict(self):
        ans = {"n_estimators": self.n_estimators,
               "learning_rate": self.learning_rate,
               "max_depth": self.max_depth,
               "feature_subsample_size": self.feature_subsample_size,
               #"trees_parameters": self.trees_parameters
               }
        return ans

    def fit(self, X, y, verbose=False, X_val=None, y_val=None):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        y : numpy ndarray
            Array of size n_objects
        """

        if self.feature_subsample_size is None:
            self.feature_subsample_size = X.shape[1]

        inds = np.arange(X.shape[0])
        prev_res = np.zeros(X.shape[0])

        self.coefs = []
        self.trees_ensemble = []
        self.samples = []

        x_res = []
        y_res = []

        for i in range(self.n_estimators):
            algo = DecisionTreeRegressor(max_features=self.feature_subsample_size,
                                         max_depth=self.max_depth,
                                         **self.trees_parameters)
            sample = np.random.choice(inds, len(inds) // 3)
            self.samples.append(sample)

            target = y[sample] - prev_res[sample]
            algo.fit(X[sample], target)
            pred = algo.predict(X[sample])
            c = minimize_scalar(lambda coef: np.mean((coef * pred - target) ** 2)).x
            self.coefs.append(c * self.learning_rate)
            self.trees_ensemble.append(algo)
            prev_res[sample] += c * self.learning_rate * target

            if verbose:
                x_res.append(i)
                pred = self.predict(X_val)

                y_res.append(rmse(y_val, pred))

        if verbose:
            return x_res, y_res

    def predict(self, X, y=None, versbose=False):
        """
        X : numpy ndarray
            Array of size n_objects, n_features

        Returns
        -------
        y : numpy ndarray
            Array of size n_objects
        """
        if versbose:
            self.history = {'rmse': [],
                            'time': [],
                            'sample_sz': self.samples[0].__len__(),
                            'lr': self.learning_rate,
                            'n_estimators': self.n_estimators,
                            'depth': self.max_depth}
            if y is None:
                raise ValueError

        preds = []
        i = 0
        for algo in self.trees_ensemble:
            begin = time.time()
            cur_pred = algo.predict(X)
            preds.append(cur_pred)
            i += 1
            my_pred = np.sum(np.array(preds) * np.array(self.coefs)[:i, np.newaxis], axis=0)
            end = time.time()
            if versbose:
                self.history['rmse'].append(np.sqrt(np.mean((my_pred - y) ** 2)))
                self.history['time'].append(end - begin)

        return np.sum(np.array(preds) * np.array(self.coefs)[:, np.newaxis], axis=0)
