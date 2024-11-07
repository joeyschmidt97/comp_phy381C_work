#!/usr/bin/python
import numpy as np
import warnings


<<<<<<< HEAD
class BaseRegressor:
    """
    A base class for regression models.
    """
    def __init__(self):
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Fits the model to the data.
        """
        raise NotImplementedError("Subclasses must implement this method")

    def predict(self, X):
        return X @ self.weights + self.bias

    def score(self, X, y):
        """
        Returns the mean squared error of the model.
        """
        return np.mean((self.predict(X) - y)**2)




class LinearRegressor(BaseRegressor):
    """
    A linear regression model is a linear function of the form:
    y = w0 + w1 * x1 + w2 * x2 + ... + wn * xn

    The weights are the coefficients of the linear function.
    The bias is the constant term w0 of the linear function.

    Attributes:
        method: str, optional. The method to use for fitting the model.
        regularization: str, optional. The type of regularization to use.
    """
    
    def __init__(self, method="global", regularization="ridge", regstrength=0.1, **kwargs):
        super().__init__(**kwargs)
        self.method = method
        self.regularization = regularization
        self.regstrength = regstrength
=======

from sklearn.base import BaseEstimator, TransformerMixin

# We are going to use class inheritance to define our object. The two base classes from
# scikit-learn represent placeholder objects for working with datasets. They include 
# many generic methods, like fetching parameters, getting the data shape, etc.
# 
# By inheriting from these classes, we ensure that our object will have access to these
# functions, even though we don't have to define them ourselves
class PrincipalComponents(BaseEstimator, TransformerMixin):
    """
    A class for performing principal component analysis on a dataset.
    """

    def __init__(self, random_state=None):
        self.random_state = random_state
        np.random.seed(self.random_state)
        self.components_ = None
        self.singular_values_ = None
>>>>>>> 7bd07dc18191a01b64387109aee25776f51026d6
        print(
            "Running with Instructor Solutions. If you meant to run your own code, do not import from solutions", 
            flush=True
        )

<<<<<<< HEAD
    # functions that begin with underscores are private, by convention.
    # Technically we could access them from outside the class, but we should
    # not do that because they can be changed or removed at any time.
    def _fit_global(self, X, y):
        """
        Fits the model using the global least squares method.
        """
        ############################################################
        #
        #
        # YOUR CODE HERE
        #
        #
        ############################################################
        #raise NotImplementedError
        if self.regularization is None:
            self.weights = np.linalg.inv(X.T @ X) @ X.T @ y
        elif self.regularization == "ridge":
            self.weights = np.linalg.inv(X.T @ X + np.eye(X.shape[1]) * self.regstrength) @ X.T @ y
        else:
            warnings.warn("Unknown regularization method, defaulting to None")
            self.weights = np.linalg.inv(X.T @ X) @ X.T @ y
        self.bias = np.mean(y - X @ self.weights)
        return self.weights, self.bias

    def _fit_iterative(self, X, y, learning_rate=0.01):
        """
        Fit the model using gradient descent.
        """
        ############################################################
        #
        #
        # YOUR CODE HERE
        #
        #
        ############################################################
        #raise NotImplementedError
        self.weights = np.zeros((X.shape[1], X.shape[1]))
        self.bias = np.mean(y)
        for i in range(X.shape[0]):
            self.weights += learning_rate * (y[i] - X[i] @ self.weights - self.bias) * X[i] - self.regstrength * self.weights
        self.weights /= X.shape[0]
        return self.weights, self.bias

    def fit(self, X, y):
        """
        Fits the model to the data. The method used is determined by the
        `method` attribute.
        """
        ############################################################
        #
        #
        # YOUR CODE HERE. If you implement the _fit_iterative method, be sure to include
        # the logic to choose between the global and iterative methods.
        #
        #
        ############################################################
        #raise NotImplementedError
        if self.method == "global":
            out = self._fit_global(X, y)
        elif self.method == "iterative":
            out = self._fit_iterative(X, y)
        else:
            out = self._fit_global(X, y)
        return out

def featurize_flowfield(field):
    """
    Compute features of a 2D spatial field. These features are chosen based on the 
    intuition that the input field is a 2D spatial field with time translation 
    invariance.

    The output is an augmented feature along the last axis of the input field.

    Args:
        field (np.ndarray): A 3D array of shape (batch, nx, ny) containing the flow field

    Returns:
        field_features (np.ndarray): A 3D array of shape (batch, nx, ny, M) containing 
            the computed features stacked along the last axis
    """
    ############################################################################
    #
    #
    # YOUR CODE HERE
    # Hint: I used concatenate to combine the features together. You have some choice of
    # which features to include, but make sure that your features are computed 
    # separately for each batch element
    # My implementation is vectorized along the first (batch) axis
    #
    ############################################################################
    # raise NotImplementedError

    ## Compute the Fourier features
    field_fft = np.fft.fft2(field)
    field_fft = np.fft.fftshift(field_fft)
    field_fft_abs = np.log(np.abs(field_fft) + 1e-8)[..., None]
    field_fft_phase = np.angle(field_fft)[..., None]

    ## Compute the spatial gradients along x and y
    field_gradx = np.vstack([np.diff(field, axis=0), field[-1, None]])[..., None]
    field_grady = np.hstack([np.diff(field, axis=1), field[:, None, -1]])[..., None]
    # print(field_fft_abs.shape, field_gradx.shape, flush=True)
    # field_grad = np.gradient(field, axis=(-2, -1))
    # field_grad = np.stack(field_grad, axis=-1)

    ## Compute the spatial Laplacian
    # field_lap = np.stack(np.gradient(field_grad, axis=(-2, -1)), axis=-1)
    # field_lap = np.sum(field_lap, axis=-1)

    field = field[..., None]
    field_features = np.concatenate(
        [field_fft_phase, field_fft_abs, field_gradx, field_grady], 
        axis=-1
    )
    return field_features
=======
    def fit(self, X):
        """
        Fit the PCA model to the data X. Store the eigenvectors in the attribute
        self.components_ and the eigenvalues in the attribute self.singular_values_

        Args:
            X (np.ndarray): A 2D array of shape (n_samples, n_features) containing the
                data to be fit.
        
        Returns:
            self (PrincipalComponents): The fitted object.
        """

        ########## YOUR CODE HERE ##########
        #
        # # YOUR CODE HERE
        # # Hint: Keep track of whether you should be multiplying by a matrix or
        # # its transpose.
        #
        ########## YOUR CODE HERE ##########
        # raise NotImplementedError()
        
        Xc = X - np.mean(X, axis=0)

        cov = Xc.T.dot(Xc) / Xc.shape[0]
        # cov = np.cov(Xc, rowvar=False) # Alternatively, using the numpy built-in
        S, V = np.linalg.eigh(cov)
        V = V.T
        sort_inds = np.argsort(S)[::-1] # sort eigenvalues in descending order
        S, V = S[sort_inds], V[sort_inds]

        # Alternative, using singular value decomposition
        # U, S, V = np.linalg.svd(Xc, full_matrices=False)
        # S = S**2 / Xc.shape[0]

        self.components_ = V
        self.singular_values_ = S

        return self

    def transform(self, X):
        """
        Transform the data X into the new basis using the PCA components
        """
        # # YOUR CODE HERE
        # raise NotImplementedError()

        Xc = X - np.mean(X, axis=0)
        return Xc.dot(self.components_.T)

    def inverse_transform(self, X):
        """
        Transform from principal components space back to the original space
        """
        # # YOUR CODE HERE
        # raise NotImplementedError()
        return X.dot(self.components_) + np.mean(X, axis=0)






>>>>>>> 7bd07dc18191a01b64387109aee25776f51026d6
