import numpy as np
import pandas as pd


def pearson_correlation( x, y):
    """
    Calculate the Pearson correlation coefficient for two given columns of data.

    Inputs:
    - x: An array containing a column of m numeric values.
    - y: An array containing a column of m numeric values. 

    Returns:
    - The Pearson correlation coefficient between the two columns.    
    """
    r = 0.0

    # Calculate the mean of x and y
    x_mean = np.mean(x)
    y_mean = np.mean(y)

    # Calculate the deviations from the mean
    x_deviation = x - x_mean
    y_deviation = y - y_mean

    # Calculate the products of deviations
    sum_of_products = np.sum(x_deviation * y_deviation)

    # Calculate the sum of squared deviations for x and y
    x_squared_deviation = np.sum(x_deviation ** 2)
    y_squared_deviation = np.sum(y_deviation ** 2)
    
    # Calculate the Pearson correlation
    r = sum_of_products / np.sqrt(x_squared_deviation * y_squared_deviation)

    return r

def feature_selection(X, y, n_features=5):
    """
    Select the best features using pearson correlation.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - best_features: list of best features (names - list of strings).  
    """
    best_features = []
    X["date"] = pd.to_numeric(pd.to_datetime(X["date"]))
    correlations = []
    for col_name in X.columns:
      # print(col_name)
      # print(X["date"].values)
      # print(y.values)
      # print(pearson_correlation(X["date"].values, y.values))
      
      correlation = pearson_correlation(X[col_name].values, y.values)
      correlations.append((col_name, correlation))
    
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)  # Sort by the highest absolute correlation
    
    best_features = [column for column, _ in correlations[:n_features]]
    
    return best_features

class LogisticRegressionGD(object):
    """
    Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    eps : float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random weight
      initialization.
    """

    def __init__(self, eta=0.00005, n_iter=10000, eps=0.000001, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        # model parameters
        self.theta = None

        # iterations history
        self.Js = []
        self.thetas = []

    def apply_bias_trick(self, X):
        """

        Applies the bias trick to the input data.

        Input:
        - X: Input data (m instances over n features).

        Returns:
        - X: Input data with an additional column of ones in the
            zeroth position (m instances over n+1 features).

        """
        # Create a vector of 1's and connecting it to be the first column (zero column)
        length_x = len(X)
        column0 = np.ones(length_x)
        X = np.c_[column0, X]
        return X

    def compute_cost(self, X, y ):
        """
        Computes the average squared difference between an observation's actual and
        predicted values for linear regression.

        Input:
        - X: Input data (m instances over n features).
        - y: True labels (m instances).
        - theta: the parameters (weights) of the model being learned.

        Returns:
        - J: the cost associated with the current set of parameters (single number).
        """

        J = 0  # We use J for the cost.
        m = len(y)
        y_pred = 1.0 / (1.0 + np.exp(-(np.dot(X, self.theta))))  # the Hypothesis function
        J = -(1/m) * np.sum(np.dot(y.T, np.log(y_pred)) + np.dot((1 - y).T , np.log(1 - y_pred)))  # Cost function
        return J
    
    def fit(self, X, y):
        """
        Fit training data (the learning phase).
        Update the theta vector in each iteration using gradient descent.
        Store the theta vector in self.thetas.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        The learned parameters must be saved in self.theta.
        This function has no return value.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        """
        # add a column of 1's to X
        X = self.apply_bias_trick(X)
        # set random seed
        np.random.seed(self.random_state)
        # Initialize theta with zeros
        self.theta = np.zeros(X.shape[1])

        for i in range(self.n_iter):
            y_pred = 1.0 / (1.0 + np.exp(-(np.dot(X, self.theta))))  # the Hypothesis function
            gradient = (1 / len(y)) * np.dot(X.T, (y_pred - y))
            self.theta -= self.eta * gradient # Update theta by learning rate and gradient

            # Compute cost and store it in the history
            cost = self.compute_cost(X, y)
            self.Js.append(cost)
            self.thetas.append(self.theta.copy())

            # Stop the function when the difference between the previous cost and the current is less than eps
            if len(self.Js) > 1 and abs(self.Js[-2] - self.Js[-1]) < self.eps:
                break


    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = None
        # Add column of 1's for theta0
        X = self.apply_bias_trick(X)
        y_pred = 1.0 / (1.0 + np.exp(-(np.dot(X, self.theta))))  # the Hypothesis function
        preds = [1 if i > 0.5 else 0 for i in y_pred]  # 𝑖𝑓 ℎ𝜃 𝑥 > 0.5 𝑡ℎ𝑒𝑛 1 𝑒𝑙𝑠𝑒 0
        return preds

def cross_validation(X, y, folds, algo, random_state):
    """
    This function performs cross validation as seen in class.

    1. shuffle the data and creates folds
    2. train the model on each fold
    3. calculate aggregated metrics

    Parameters
    ----------
    X : {array-like}, shape = [n_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y : array-like, shape = [n_examples]
      Target values.
    folds : number of folds (int)
    algo : an object of the classification algorithm
    random_state : int
      Random number generator seed for random weight
      initialization.

    Returns the cross validation accuracy.
    """

    cv_accuracy = None

    # set random seed
    np.random.seed(random_state)

    # Shuffle the data and creates folds
    index = np.arange(X.shape[0])
    np.random.shuffle(index)
    X = X[index]
    y = y[index]
    fold_size = int(X.shape[0] / folds)
    fold_accuracy = []

    for i in range(folds):
      # Create training and testing sets for each fold
        X_train = np.concatenate((X[:i * fold_size], X[(i + 1) * fold_size:]), axis=0)
        X_test = X[i * fold_size:(i + 1) * fold_size]
        y_train = np.concatenate((y[:i * fold_size], y[(i + 1) * fold_size:]), axis=0)
        y_test = y[i * fold_size:(i + 1) * fold_size]

        # Train the model on each fold
        algo.fit(X_train, y_train)

        # Calculate accuracy for the current fold
        n_correct = 0
        for sample, target in zip(X_test, y_test):
            prediction = algo.predict(sample.reshape(1, -1))[0]  # reshape sample for single prediction
            if prediction == target:
                n_correct += 1
        fold_accuracy.append(n_correct / len(y_test))

    # Calculate aggregated metrics (mean accuracy across all folds)
    cv_accuracy = np.mean(fold_accuracy)
    return cv_accuracy


def norm_pdf(data, mu, sigma):
    """
    Calculate normal desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mu: The mean value of the distribution.
    - sigma:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mu and sigma for the given x.    
    """
    p = (np.exp(np.square((data - mu)) / (-2 * np.square(sigma)))) / (np.sqrt(2 * np.pi * np.square(sigma)))

    return p

class EM(object):
    """
    Naive Bayes Classifier using Gauusian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    n_iter : int
      Passes over the training dataset in the EM proccess
    eps: float
      minimal change in the cost to declare convergence
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, n_iter=1000, eps=0.01, random_state=1991):
        self.k = k
        self.n_iter = n_iter
        self.eps = eps
        self.random_state = random_state

        np.random.seed(self.random_state)

        self.responsibilities = None
        self.weights = None
        self.mus = None
        self.sigmas = None
        self.costs = []

    # initial guesses for parameters
    def init_params(self, data):
        """
        Initialize distribution params
        """
        random_indices = np.random.permutation(data.shape[0])[:self.k]
        self.mus = data[random_indices].reshape(self.k)

        # Initialize standard deviations with random values
        self.sigmas = np.random.uniform(low=0.5, high=1.5, size=self.k)

        # Initialize weights equally
        self.weights = np.full(self.k, 1 / self.k)

    def expectation(self, data):
        """
        E step - This function should calculate and update the responsibilities
        """
        probabilities = np.multiply(self.weights, norm_pdf(data, self.mus, self.sigmas))
        normalization_factors = probabilities.sum(axis=1, keepdims=True)
        self.responsibilities = np.divide(probabilities, normalization_factors)

    def maximization(self, data):
        """
        M step - This function should calculate and update the distribution params
        """
        # Calculate the distribution params accoording to the formula.
        # Calculate the mean of the responsibilities across the data points for each Gaussian component
        self.weights = np.mean(self.responsibilities, axis=0)
        data_reshaped = data.reshape(-1, 1)
        # Calculate the weighted sum of the data points for each Gaussian component
        weighted_sum_data = self.responsibilities * data_reshaped
        # Sum the weighted data points for each component and normalize by the sum of responsibilities
        sum_weighted_data = np.sum(weighted_sum_data, axis=0)
        sum_responsibilities = np.sum(self.responsibilities, axis=0)
        self.mus = sum_weighted_data / sum_responsibilities
        # Calculate the difference between the data points and the means
        data_minus_mus = data_reshaped - self.mus
        # Calculate the weighted squared differences
        weighted_squared_diff = self.responsibilities * np.square(data_minus_mus)
        # Calculate the mean of these weighted squared differences for each Gaussian component
        variance = np.mean(weighted_squared_diff, axis=0)
        # Calculate the standard deviations (sigmas) by taking the square root of the variances divided by the weights
        self.sigmas = np.sqrt(variance / self.weights)

    def fit(self, data):
        """
        Fit training data (the learning phase).
        Use init_params and then expectation and maximization function in order to find params
        for the distribution.
        Store the params in attributes of the EM object.
        Stop the function when the difference between the previous cost and the current is less than eps
        or when you reach n_iter.
        """
        # Initialize parameters
        self.init_params(data)
        self.costs.append(self.cost(data))

        # Iterate to find the parameters for the distribution
        for iteration in range(self.n_iter):
            # Compute the current cost
            current_cost = self.cost(data)
            self.costs.append(current_cost)

            # E-step: update responsibilities
            self.expectation(data)

            # M-step: update weights, means, and variances
            self.maximization(data)

            # Check for convergence
            if len(self.costs) > 1:
                cost_diff = self.costs[-1] - self.costs[-2]
                if abs(cost_diff) < self.eps:
                    break

    def cost(self, data):
        """
        Calculate the cost (negative log-likelihood) of the data.
        """
        # Calculate the probability density function for each component
        prob_density = self.weights * norm_pdf(data, self.mus, self.sigmas)

        # Sum the probabilities for each data point
        sum_prob_density = np.sum(prob_density, axis=1)

        # Calculate the log-likelihood
        log_likelihood = np.sum(np.log(sum_prob_density))

        # Return the negative log-likelihood as the cost
        return -log_likelihood

    def get_dist_params(self):
        return self.weights, self.mus, self.sigmas

def gmm_pdf(data, weights, mus, sigmas):
    """
    Calculate gmm desnity function for a given data,
    mean and standrad deviation.
 
    Input:
    - data: A value we want to compute the distribution for.
    - weights: The weights for the GMM
    - mus: The mean values of the GMM.
    - sigmas:  The standard deviation of the GMM.
 
    Returns the GMM distribution pdf according to the given mus, sigmas and weights
    for the given data.    
    """
    pdf = None
    # Calculate density function by formula sum of weight* density func for each gaussian i
    pdf = np.sum(weights * norm_pdf(data.reshape(-1, 1), mus, sigmas), axis=1)
    return pdf

class NaiveBayesGaussian(object):
    """
    Naive Bayes Classifier using Gaussian Mixture Model (EM) for calculating the likelihood.

    Parameters
    ------------
    k : int
      Number of gaussians in each dimension
    random_state : int
      Random number generator seed for random params initialization.
    """

    def __init__(self, k=1, random_state=1991):
        self.k = k
        self.random_state = random_state
        self.prior = None

    def fit(self, X, y):
        """
        Fit training data.

        Parameters
        ----------
        X : array-like, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.
        """
        # Initialize priors and gmms dictionaries
        # Fitting the data.
        self.X = X
        self.y = y
        self.priors = {class_Label: len(y[y == class_Label]) / len(y) for class_Label in np.unique(y)}
        self.gmms = {class_Label: {feature: EM(self.k) for feature in range(X.shape[1])} for class_Label in
                     np.unique(y)}

        for label in self.gmms.keys():
            for feature in self.gmms[label].keys():
                self.gmms[label][feature].fit(X[y == label][:, feature].reshape(-1, 1))


    def predict(self, X):
        """
        Return the predicted class labels for a given instance.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        """
        preds = []
        for instance in X:
            posteriors = []
            for class_label in self.priors.keys():
                likelihood = 1
                for feature in range(len(instance)):
                    weights, mus, sigmas = self.gmms[class_label][feature].get_dist_params()
                    gmm = gmm_pdf(instance[feature], weights, mus, sigmas)
                    likelihood *= gmm
                posterior = self.priors[class_label] * likelihood
                posteriors.append((posterior, class_label))
            preds.append(max(posteriors, key=lambda t: t[0])[1])
        preds = np.array(preds).reshape(-1,1)
        return preds

def model_evaluation(x_train, y_train, x_test, y_test, k, best_eta, best_eps):
    ''' 
    Read the full description of this function in the notebook.

    You should use visualization for self debugging using the provided
    visualization functions in the notebook.
    Make sure you return the accuracies according to the return dict.

    Parameters
    ----------
    x_train : array-like, shape = [n_train_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_train : array-like, shape = [n_train_examples]
      Target values.
    x_test : array-like, shape = [n_test_examples, n_features]
      Training vectors, where n_examples is the number of examples and
      n_features is the number of features.
    y_test : array-like, shape = [n_test_examples]
      Target values.
    k : Number of gaussians in each dimension
    best_eta : best eta from cv
    best_eps : best eta from cv
    ''' 

    lor_train_acc = None
    lor_test_acc = None
    bayes_train_acc = None
    bayes_test_acc = None

    # Fit Logistic Regression model
    logistic_model = LogisticRegressionGD(eta=best_eta, eps=best_eps)
    logistic_model.fit(x_train, y_train)
    lor_train_preds = logistic_model.predict(x_train)
    lor_test_preds = logistic_model.predict(x_test)
    lor_train_acc = accuracy_score(y_train, lor_train_preds)
    lor_test_acc = accuracy_score(y_test, lor_test_preds)

    # Fit Naive Bayes model
    naive_bayes_model = NaiveBayesGaussian(k=k)
    naive_bayes_model.fit(x_train, y_train)
    bayes_train_preds = naive_bayes_model.predict(x_train)
    bayes_test_preds = naive_bayes_model.predict(x_test)
    bayes_train_acc = accuracy_score(y_train, bayes_train_preds)
    bayes_test_acc = accuracy_score(y_test, bayes_test_preds)

    return {'lor_train_acc': lor_train_acc,
            'lor_test_acc': lor_test_acc,
            'bayes_train_acc': bayes_train_acc,
            'bayes_test_acc': bayes_test_acc}

def accuracy_score(y_true, y_pred):
    """
    Calculate accuracy given true labels and predicted labels.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.

    Returns
    -------
    float
        Accuracy score.
    """
    correct = 0
    total = len(y_true)
    for true_label, pred_label in zip(y_true, y_pred):
        if true_label == pred_label:
            correct += 1
    return correct / total

def generate_datasets():
    from scipy.stats import multivariate_normal
    '''
    This function should have no input.
    It should generate the two dataset as described in the jupyter notebook,
    and return them according to the provided return dict.
    '''
    dataset_a_features = None
    dataset_a_labels = None
    dataset_b_features = None
    dataset_b_labels = None
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return{'dataset_a_features': dataset_a_features,
           'dataset_a_labels': dataset_a_labels,
           'dataset_b_features': dataset_b_features,
           'dataset_b_labels': dataset_b_labels
           }