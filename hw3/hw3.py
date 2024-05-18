import numpy as np

class conditional_independence():

    def __init__(self):

        # You need to fill the None value with *valid* probabilities
        self.X = {0: 0.3, 1: 0.7}  # P(X=x)
        self.Y = {0: 0.3, 1: 0.7}  # P(Y=y)
        self.C = {0: 0.5, 1: 0.5}  # P(C=c)

        self.X_Y = {
            (0, 0): 0.12,
            (0, 1): 0.16,
            (1, 0): 0.16,
            (1, 1): 0.54
        }  # P(X=x, Y=y)

        self.X_C = {
            (0, 0): 0.15,
            (0, 1): 0.15,
            (1, 0): 0.35,
            (1, 1): 0.35
        }  # P(X=x, C=y)

        self.Y_C = {
            (0, 0): 0.15,
            (0, 1): 0.15,
            (1, 0): 0.35,
            (1, 1): 0.35
        }  # P(Y=y, C=c)

        self.X_Y_C = {
            (0, 0, 0): 0.045,
            (0, 0, 1): 0.045,
            (0, 1, 0): 0.105,
            (0, 1, 1): 0.105,
            (1, 0, 0): 0.105,
            (1, 0, 1): 0.105,
            (1, 1, 0): 0.245,
            (1, 1, 1): 0.245,
        }  # P(X=x, Y=y, C=c)

    def is_X_Y_dependent(self):
        """
        return True iff X and Y are depndendent
        """
        X = self.X
        Y = self.Y
        X_Y = self.X_Y
        for x in X:
            for y in Y:
                if not np.isclose(X_Y[(x, y)], X[x] * Y[y]):
                    return True
        return False


    def is_X_Y_given_C_independent(self):
        """
        return True iff X_given_C and Y_given_C are indepndendent
        """
        X = self.X
        Y = self.Y
        C = self.C
        X_C = self.X_C
        Y_C = self.Y_C
        X_Y_C = self.X_Y_C
        for x in X:
            for y in Y:
                for c in C:
                    x_y_given_c = X_Y_C[(x,y,c)] / C[c]
                    x_given_c = X_C[(x,c)] / C[c]
                    y_given_c = Y_C[(y,c)] / C[c]
                    if not np.isclose(x_y_given_c, x_given_c * y_given_c):
                        return False
        return True

def poisson_log_pmf(k, rate):
    """
    k: A discrete instance
    rate: poisson rate parameter (lambda)

    return the log pmf value for instance k given the rate
    """
    ##log_p = (rate ** k * np.e ** -rate) / math.factorial(k)
     
    log_p = k * np.log(rate) - rate - np.log(np.math.factorial(k))  
    return log_p

def get_poisson_log_likelihoods(samples, rates):
    """
    samples: set of univariate discrete observations
    rates: an iterable of rates to calculate log-likelihood by.

    return: 1d numpy array, where each value represent that log-likelihood value of rates[i]
    """
    likelihoods = np.zeros(len(rates))
    for i, rate in enumerate(rates):
        i_prob = 0
        for sample in samples:
            i_prob += poisson_log_pmf(sample,rate)
        likelihoods[i] = i_prob
    return likelihoods

def possion_iterative_mle(samples, rates):
    """
    samples: set of univariate discrete observations
    rate: a rate to calculate log-likelihood by.

    return: the rate that maximizes the likelihood 
    """
    best_rate = 0.0
    best_likelihood = -99999999
    likelihoods = get_poisson_log_likelihoods(samples, rates) # might help
    for i, rate in enumerate(rates):
        if likelihoods[i] > best_likelihood:
            best_likelihood = likelihoods[i]
            best_rate = rate
    
    return best_rate


def possion_analytic_mle(samples):
    """
    samples: set of univariate discrete observations

    return: the rate that maximizes the likelihood
    """
    mean = None

    num_samples = len(samples)
    sum_samples = sum(samples)
    mean = sum_samples / num_samples

    return mean

def normal_pdf(x, mean, std):
    """
    Calculate normal desnity function for a given x, mean and standrad deviation.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean value of the distribution.
    - std:  The standard deviation of the distribution.
 
    Returns the normal distribution pdf according to the given mean and std for the given x.    
    """
    p = None
    mul = 1.0 / np.sqrt(2 * np.pi * (std ** 2))
    power = -((x - mean) ** 2)/ (std ** 2) * 2
    p = mul * np.e ** power
    return p

class NaiveNormalClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which encapsulates the relevant parameters(mean, std) for a class conditinoal normal distribution.
        The mean and std are computed from a given data set.
        
        Input
        - dataset: The dataset as a 2d numpy array, assuming the class label is the last column
        - class_value : The class to calculate the parameters for.
        """
        self.class_value = class_value
        self.dataset = dataset
        last_column = dataset[:, -1]
        self.class_dataset = dataset[last_column == class_value][:, :-1]
        self.mean = np.mean(self.class_dataset, axis=0)
        self.std = np.std(self.class_dataset, axis=0)
    
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        prior = None
        # The amount of samples in class divided by total sample amount
        prior = len(self.class_dataset) / len(self.dataset)

        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihhod porbability of the instance under the class according to the dataset distribution.
        """
        likelihood = 1
        # loop all features
        for i in range(len(x)):
            likelihood *= normal_pdf(x[i], self.mean[i], self.std[i])
        return likelihood
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        # calculate according to function (likelihood * prior)
        posterior = self.get_instance_likelihood(x) * self.get_prior()
        return posterior

class MAPClassifier():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions. 
        One for class 0 and one for class 1, and will predict an instance
        using the class that outputs the highest posterior probability 
        for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods 
                     for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods 
                     for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        pred = None
        if self.ccd0.get_instance_posterior(x) > self.ccd1.get_instance_posterior(x):
            pred = 0
        else:
            pred = 1;
        return pred

def compute_accuracy(test_set, map_classifier):
    """
    Compute the accuracy of a given a test_set using a MAP classifier object.
    
    Input
        - test_set: The test_set for which to compute the accuracy (Numpy array). where the class label is the last column
        - map_classifier : A MAPClassifier object capable of prediciting the class for each instance in the testset.
        
    Ouput
        - Accuracy = #Correctly Classified / test_set size
    """
    acc = None
    correct = 0
    # check for each instance in test set if the prediction is correct
    for instance in test_set:
        instance_class = instance[-1]
        prediction = map_classifier.predict(instance[:-1])
        if prediction == instance_class:
            correct += 1
    # num correct / num of instances checked
    acc = correct / len(test_set)
    return acc

def multi_normal_pdf(x, mean, cov):
    """
    Calculate multi variable normal desnity function for a given x, mean and covarince matrix.
 
    Input:
    - x: A value we want to compute the distribution for.
    - mean: The mean vector of the distribution.
    - cov:  The covariance matrix of the distribution.
 
    Returns the normal distribution pdf according to the given mean and var for the given x.    
    """
    pdf = None
    d = mean.shape[0]
    # Calculate the exponent term
    exponent = -0.5 * np.dot(np.dot((x - mean).T, np.linalg.inv(cov)), (x - mean))
    # Calculate the constant term
    constant = 1 / ((2 * np.pi) ** (d / 2) * np.sqrt(np.linalg.det(cov)))
    # Calculate the PDF
    pdf = constant * np.exp(exponent)
    
    return pdf

class MultiNormalClassDistribution():

    def __init__(self, dataset, class_value):
        """
        A class which encapsulate the relevant parameters(mean, cov matrix) for a class conditinoal multi normal distribution.
        The mean and cov matrix (You can use np.cov for this!) will be computed from a given data set.
        
        Input
        - dataset: The dataset as a numpy array
        - class_value : The class to calculate the parameters for.
        """
        self.class_value = class_value
        self.dataset = dataset
        last_column = dataset[:, -1]
        self.class_dataset = dataset[last_column == class_value][:, :-1]
        self.mean = np.mean(self.class_dataset, axis=0)
        self.cov = np.cov(self.class_dataset.T)
        
    def get_prior(self):
        """
        Returns the prior porbability of the class according to the dataset distribution.
        """
        prior = len(self.class_dataset)/ len(self.dataset)
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under the class according to the dataset distribution.
        """
        likelihood = multi_normal_pdf(x, self.mean, self.cov)
        return likelihood
        
    
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = self.get_instance_likelihood(x) * self.get_prior()
        return posterior

class MaxPrior():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum prior classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest prior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        # compare prior probability of class 0 and class 1
        if self.ccd0.get_prior() > self.ccd1.get_prior():
            return 0
        return 1

class MaxLikelihood():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum Likelihood classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predicit an instance
        by the class that outputs the highest likelihood probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1
    
    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        # compare likelihood probability of class 0 and class 1
        if self.ccd0.get_instance_likelihood(x) > self.ccd1.get_instance_likelihood(x):
            return 0
        return 1

EPSILLON = 1e-6 # if a certain value only occurs in the test set, the probability for that value will be EPSILLON.

class DiscreteNBClassDistribution():
    def __init__(self, dataset, class_value):
        """
        A class which computes and encapsulate the relevant probabilites for a discrete naive bayes 
        distribution for a specific class. The probabilites are computed with laplace smoothing.
        
        Input
        - dataset: The dataset as a numpy array.
        - class_value: Compute the relevant parameters only for instances from the given class.
        """
        self.dataset = dataset
        self.class_value = class_value
        last_column = dataset[:, -1]
        self.class_dataset = dataset[last_column == class_value][:, :-1]
    
    def get_prior(self):
        """
        Returns the prior porbability of the class 
        according to the dataset distribution.
        """
        prior = len(self.class_dataset) / len(self.dataset)
        return prior
    
    def get_instance_likelihood(self, x):
        """
        Returns the likelihood of the instance under 
        the class according to the dataset distribution.
        """
        likelihood = 1.0
        for i, value in enumerate(x):
            # calc all parameters needed
            n_i = len(self.class_dataset)
            V_j = len(set(self.class_dataset[:, i]))
            n_ij = np.count_nonzero(self.class_dataset[:, i] == value)
            if (np.count_nonzero(self.dataset[:, i] == value) == 0):
                calc_likelihood = EPSILLON
            else:
                calc_likelihood = (n_ij + 1) / (n_i + V_j)
                
            likelihood *= calc_likelihood
        return likelihood
        
    def get_instance_posterior(self, x):
        """
        Returns the posterior porbability of the instance 
        under the class according to the dataset distribution.
        * Ignoring p(x)
        """
        posterior = self.get_instance_likelihood(x) * self.get_prior()
        return posterior


class MAPClassifier_DNB():
    def __init__(self, ccd0 , ccd1):
        """
        A Maximum a posteriori classifier. 
        This class will hold 2 class distributions, one for class 0 and one for class 1, and will predict an instance
        by the class that outputs the highest posterior probability for the given instance.
    
        Input
            - ccd0 : An object contating the relevant parameters and methods for the distribution of class 0.
            - ccd1 : An object contating the relevant parameters and methods for the distribution of class 1.
        """
        self.ccd0 = ccd0
        self.ccd1 = ccd1

    def predict(self, x):
        """
        Predicts the instance class using the 2 distribution objects given in the object constructor.
    
        Input
            - An instance to predict.
        Output
            - 0 if the posterior probability of class 0 is higher and 1 otherwise.
        """
        if self.ccd0.get_instance_posterior(x) > self.ccd1.get_instance_posterior(x):
            return 0
        return 1

    def compute_accuracy(self, test_set):
        """
        Compute the accuracy of a given a testset using a MAP classifier object.

        Input
            - test_set: The test_set for which to compute the accuracy (Numpy array).
        Ouput
            - Accuracy = #Correctly Classified / #test_set size
        """
        acc = None
        correct = 0
        # check for each instance in test set if the prediction is correct
        for instance in test_set:
            instance_class = instance[-1]
            prediction = self.predict(instance[:-1])
            if prediction == instance_class:
                correct += 1
        # num correct / num of instances checked
        acc = correct / len(test_set)
        return acc


