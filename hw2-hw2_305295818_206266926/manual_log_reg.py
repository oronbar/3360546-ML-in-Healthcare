import numpy as np
import matplotlib.pyplot as plt


def sigmoid(z):
    """
    The sigmoid function mapping z from [-inf,inf] to [0,1].
    :param z: vector or a single number.
    :return: sigmoid(z).
    """
    # ------------------------ IMPLEMENT YOUR CODE HERE: ----------------------------------
    return 1 / (1 + np.exp(-z))
    # --------------------------------------------------------------------------------------


#todo: add columns of 1 before even splitting to train test in the last part


class ManualLogisticRegression:
    def __init__(self, random_state=336546):
        np.random.seed(random_state)
        self.w = np.random.randn(5)

    def fit(self, X, Y, eta=0.005, plot=False):
        """
        This function trains the model by applying the gradient descent for logistic regression loss which is the binary
        cross-entropy loss and updating the weights in every iteration.
        :param X: Feature matrix (column of 1 was already added in the notebook for the bias).
        :param Y: Adequate true labels (either 1 or 0).
        :param eta: Learning rate.
        :param plot: Boolean for creating a plot of the loss as a function of the iterations.
        :return:
        """
        if plot:
            loss_vec = np.zeros(len(X))
        for idx, (x, y) in enumerate(zip(X, Y)):  # x is a single example and y is its adequate label
            # ------------------------ IMPLEMENT YOUR CODE HERE: ----------------------------------
            hwx = self.predict_proba(x)
            self.w -= eta * (hwx - y) * x
            # --------------------------------------------------------------------------------------
            if plot:
                loss_vec[idx] = self.log_loss(X, Y)
        if plot:
            plt.plot(loss_vec)
            plt.xlabel('# of iterations')
            plt.ylabel('Loss')

    def log_loss(self, x, y):
        """
        This function computes the binary cross-entropy loss. For stability, epsilon should be added as written in the
        document.
        :param x: Feature matrix (could be also a single vector).
        :param y: Adequate true labels (either 1 or 0).
        :return: the mean cross-entropy loss.
        """
        # ------------------------ IMPLEMENT YOUR CODE HERE: ----------------------------------
        eps = 1e-10
        hwx = self.predict_proba(x)
        log_loss = -np.mean(y * np.log(hwx + eps) + (1 - y) * np.log((1 - hwx) + eps))
        # --------------------------------------------------------------------------------------
        return log_loss

    def predict_proba(self, x):
        """
        This function computes the probability of every example in x to belong to the class "1" using the trained model.
        :param x: Feature matrix (could be also a single vector).
        :return: vector at the length of examples in x where every element is the probability to belong to class "1" per example.
        """
        # ------------------------ IMPLEMENT YOUR CODE HERE: ----------------------------------
        z = x @ self.w
        y_pred_proba = sigmoid(z)
        # --------------------------------------------------------------------------------------
        return y_pred_proba

    def predict(self, x, thresh=0.5):
        """
        This function labels every example according to the calculated probability with the use of a threshold.
        :param x: Feature matrix (could be also a single vector).
        :param thresh: decision threshold.
        :return: vector at the length of examples in x where every element is the estimated label (0 or 1) per example.
        """
        # ------------------------ IMPLEMENT YOUR CODE HERE: ----------------------------------
        y_pred = self.predict_proba(x) > thresh
        # --------------------------------------------------------------------------------------
        return y_pred

    def score(self, x, y):
        """
        This function computes the accuracy of the trained model's estimations.
        :param x: Feature matrix (could be also a single vector).
        :param y: Adequate true labels (either 1 or 0).
        :return: Estimator's accuracy.
        """
        return np.sum(self.predict(x) == y)/len(y)

    def conf_matrix(self, x, y):
        """
        This function computes the confusion matrix for the prediction of the trained model. First value of the matrix
        was given as a hint.
        :param x: Feature matrix (could be also a single vector).
        :param y: Adequate true labels (either 1 or 0).
        :return: Confusion matrix.
        """
        conf_mat = np.zeros((2, 2))
        y_pred = self.predict(x)
        conf = (y_pred == y)
        conf_mat[0, 0] += np.sum(1 * (conf[y_pred == 0] == 1))
        # ------------------------ IMPLEMENT YOUR CODE HERE: ----------------------------------
        conf_mat[0, 0] = np.sum((y_pred == 0) & (y == 0))
        conf_mat[0, 1] = np.sum((y_pred == 1) & (y == 0))
        conf_mat[1, 0] = np.sum((y_pred == 0) & (y == 1))
        conf_mat[1, 1] = np.sum((y_pred == 1) & (y == 1))
        # --------------------------------------------------------------------------------------
        return conf_mat
