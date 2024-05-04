import numpy as np
import matplotlib.pyplot as plt

### Chi square table values ###
# The first key is the degree of freedom 
# The second key is the p-value cut-off
# The values are the chi-statistic that you need to use in the pruning

chi_table = {1: {0.5 : 0.45,
             0.25 : 1.32,
             0.1 : 2.71,
             0.05 : 3.84,
             0.0001 : 100000},
         2: {0.5 : 1.39,
             0.25 : 2.77,
             0.1 : 4.60,
             0.05 : 5.99,
             0.0001 : 100000},
         3: {0.5 : 2.37,
             0.25 : 4.11,
             0.1 : 6.25,
             0.05 : 7.82,
             0.0001 : 100000},
         4: {0.5 : 3.36,
             0.25 : 5.38,
             0.1 : 7.78,
             0.05 : 9.49,
             0.0001 : 100000},
         5: {0.5 : 4.35,
             0.25 : 6.63,
             0.1 : 9.24,
             0.05 : 11.07,
             0.0001 : 100000},
         6: {0.5 : 5.35,
             0.25 : 7.84,
             0.1 : 10.64,
             0.05 : 12.59,
             0.0001 : 100000},
         7: {0.5 : 6.35,
             0.25 : 9.04,
             0.1 : 12.01,
             0.05 : 14.07,
             0.0001 : 100000},
         8: {0.5 : 7.34,
             0.25 : 10.22,
             0.1 : 13.36,
             0.05 : 15.51,
             0.0001 : 100000},
         9: {0.5 : 8.34,
             0.25 : 11.39,
             0.1 : 14.68,
             0.05 : 16.92,
             0.0001 : 100000},
         10: {0.5 : 9.34,
              0.25 : 12.55,
              0.1 : 15.99,
              0.05 : 18.31,
              0.0001 : 100000},
         11: {0.5 : 10.34,
              0.25 : 13.7,
              0.1 : 17.27,
              0.05 : 19.68,
              0.0001 : 100000}}

def calc_gini(data):
    """
    Calculate gini impurity measure of a dataset.
 
    Input:
    - data: any dataset where the last column holds the labels.
 
    Returns:
    - gini: The gini impurity value.
    """
    gini = 1.0
    # cut last column, as it contains the labels
    labels = data[:, -1]
    # calculate how many samples each label has
    label, count = np.unique(labels, return_counts=True)
    # calculate based on gini function
    for i in count:
        p_i = i / len(data)
        gini -= p_i**2
    return gini



def calc_entropy(data):
    """
    Calculate the entropy of a dataset.

    Input:
    - data: any dataset where the last column holds the labels.

    Returns:
    - entropy: The entropy value.
    """
    entropy = 0.0
    # cut last column, as it contains the labels
    labels = data[:, -1]
    # calculate how many samples each label has
    label, count = np.unique(labels, return_counts=True)
    # calculate based on entropy function
    for i in count:
        p_i = i / len(data)
        entropy += (-1) * p_i * np.log2(p_i)
    return entropy

class DecisionNode:

    
    def __init__(self, data, impurity_func, feature=-1,depth=0, chi=1, max_depth=1000, gain_ratio=False):
        
        self.data = data # the relevant data for the node
        self.feature = feature # column index of criteria being tested
        self.pred = self.calc_node_pred() # the prediction of the node
        self.depth = depth # the current depth of the node
        self.children = [] # array that holds this nodes children
        self.children_values = []
        self.terminal = False # determines if the node is a leaf
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.impurity_func = impurity_func
        self.gain_ratio = gain_ratio
        self.feature_importance = 0
    
    def calc_node_pred(self):
        """
        Calculate the node prediction.

        Returns:
        - pred: the prediction of the node
        """
        pred = None
        label, counts = np.unique(self.data[:, -1], return_counts=True)
        biggest_index = np.argmax(counts)
        pred = label[biggest_index]
        return pred

    def add_child(self, node, val):
        """
        Adds a child node to self.children and updates self.children_values

        This function has no return value
        """
        self.children.append(node)
        self.children_values.append(val)
        
    def calc_feature_importance(self, n_total_sample):
        """
        Calculate the selected feature importance.
        
        Input:
        - n_total_sample: the number of samples in the dataset.

        This function has no return value - it stores the feature importance in 
        self.feature_importance
        """
        prob = len(self.data) / n_total_sample
        goodness = self.goodness_of_split(self.feature)
        self.feature_importance = prob * goodness
    
    def goodness_of_split(self, feature):
        """
        Calculate the goodness of split of a dataset given a feature and impurity function.

        Input:
        - feature: the feature index the split is being evaluated according to.

        Returns:
        - goodness: the goodness of split
        - groups: a dictionary holding the data after splitting 
                  according to the feature values.
        """
        if self.gain_ratio == True:
            self.impurity_func = calc_entropy;
            
        goodness = 0
        groups = {} # groups[feature_value] = data_subset
        sum = 0
        split_info = 0;

        # Calculate the base
        base = self.impurity_func(self.data)
       
        # calculate how many instances for each feature value
        feature_column = self.data[:, feature]
        feature_values, values_occurences = np.unique(feature_column, return_counts=True)

        # calculate the sum part of the goodness formula
        for feature_value, feature_count in zip (feature_values, values_occurences):
            data_subset = self.data[self.data[:, feature] == feature_value]
            groups[feature_value] = data_subset
            sum += feature_count/len(self.data) * self.impurity_func(data_subset)
            split_info -= feature_count/len(self.data)*np.log2(feature_count/len(self.data))
        gain = base - sum

        if self.gain_ratio == True:
            goodness = gain/split_info
        else:
            goodness = gain

        return goodness, groups

    def labels_counts(self, labels):
            return {label: np.sum(labels == label) for label in np.unique(labels)}


    def compute_chi(self, data, sub_data):
        chi = 0
        data_label_counts = self.labels_counts(data[:, -1])
        data_size = len(data)

        for child in sub_data.values():
            child_label_counts = self.labels_counts(child[:, -1])
            child_size = len(child)

            for label, data_count in data_label_counts.items():
                expected = child_size * (data_count / data_size)
                res = child_label_counts.get(label, 0)
                if expected > 0:
                    chi += ((res - expected) ** 2) / expected

        return chi

    def split(self):
        """
        Splits the current node according to the self.impurity_func. This function finds
        the best feature to split according to and create the corresponding children.
        This function should support pruning according to self.chi and self.max_depth.

        This function has no return value
        """
        # ensure doesnt increase if at max length
        if self.depth >= self.max_depth or self.terminal:
            self.terminal = True
            return

        best_goodness = 0
        best_feature_index = None
        best_feature_groups = None

        # find the index of the best feature to split by
        for feature in range(self.data.shape[1] - 1):
            curr_goodness, groups = self.goodness_of_split(feature)
            if curr_goodness > best_goodness:
                best_goodness = curr_goodness
                best_feature_index = feature
                best_feature_groups = groups

        # If no improvement was found, mark the node as terminal and return
        if best_goodness <= 0.0:
            self.terminal = True
            return

        # Apply the chi-squared test for pruning if chi value is not 1
        chi_val = self.compute_chi(self.data, best_feature_groups)
        if self.chi != 1 and chi_val < chi_table[len(best_feature_groups) - 1][self.chi]:
            self.terminal = True
            return

        self.feature = best_feature_index

        # create and add all children for the best feature
        for group_value, subset in best_feature_groups.items():
            child_node = DecisionNode(subset, self.impurity_func, depth=self.depth + 1,
                                    chi=self.chi, max_depth=self.max_depth, gain_ratio=self.gain_ratio)
            self.add_child(child_node, group_value)

        

class DecisionTree:
    def __init__(self, data, impurity_func, feature=-1, chi=1, max_depth=1000, gain_ratio=False):
        self.data = data # the relevant data for the tree
        self.impurity_func = impurity_func # the impurity function to be used in the tree
        self.chi = chi 
        self.max_depth = max_depth # the maximum allowed depth of the tree
        self.gain_ratio = gain_ratio #
        self.root = None # the root node of the tree
        
    def build_tree(self):
        """
        Build a tree using the given impurity measure and training dataset. 
        You are required to fully grow the tree until all leaves are pure 
        or the goodness of split is 0.

        This function has no return value
        """
        self.root = DecisionNode(self.data, self.impurity_func, chi=self.chi, max_depth=self.max_depth, gain_ratio=self.gain_ratio)
        incomplete_nodes = [self.root]
        while(len(incomplete_nodes) > 0):
            node = incomplete_nodes.pop(0)
            if node.terminal:
                continue
            node.split()
            for child in node.children:
                incomplete_nodes.append(child)
  

    def predict(self, instance):
        """
        Predict a given instance
     
        Input:
        - instance: an row vector from the dataset. Note that the last element of this vector is the label of the instance.
     
        Output: the prediction of the instance.
        """
        node = self.root
        found = True
        while (not node.terminal and found):
            found = False
            for child,value in zip(node.children, node.children_values):
                if (instance[node.feature] == value):
                    node = child
                    found = True
                    break
        return node.pred

    def calc_accuracy(self, dataset):
        """
        Predict a given dataset 
     
        Input:
        - dataset: the dataset on which the accuracy is evaluated
     
        Output: the accuracy of the decision tree on the given dataset (%).
        """
        accuracy = 0
        pred_right = 0
        for i in range(dataset.shape[0]): # for each row
        # select row of current instance to be checked
            instance = dataset[i]
            label = self.predict(instance)
            if label == instance[-1]:
                pred_right += 1
        if dataset.shape[0] != 0:
            accuracy = pred_right/len(dataset)
        return accuracy
        
    def depth(self):
        return self.root.depth()

def depth_pruning(X_train, X_validation):
    """
    Calculate the training and validation accuracies for different depths
    using the best impurity function and the gain_ratio flag you got
    previously. On a single plot, draw the training and testing accuracy 
    as a function of the max_depth. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output: the training and validation accuracies per max depth
    """
    training = []
    validation  = []
    root = None
    for max_depth in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        tree_entropy_gain_ratio = DecisionTree(data=X_train, impurity_func=calc_entropy, max_depth=max_depth, gain_ratio=True) # entropy and gain ratio
        tree_entropy_gain_ratio.build_tree()
        training.append(tree_entropy_gain_ratio.calc_accuracy(X_train))
        validation.append(tree_entropy_gain_ratio.calc_accuracy(X_validation))
    return training, validation


def chi_pruning(X_train, X_test):

    """
    Calculate the training and validation accuracies for different chi values
    using the best impurity function and the gain_ratio flag you got
    previously. 

    Input:
    - X_train: the training data where the last column holds the labels
    - X_validation: the validation data where the last column holds the labels
 
    Output:
    - chi_training_acc: the training accuracy per chi value
    - chi_validation_acc: the validation accuracy per chi value
    - depth: the tree depth for each chi value
    """
    chi_training_acc = []
    chi_validation_acc  = []
    depth = []

    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
        
    return chi_training_acc, chi_testing_acc, depth


def count_nodes(node):
    """
    Count the number of node in a given tree
 
    Input:
    - node: a node in the decision tree.
 
    Output: the number of node in the tree.
    """
    ###########################################################################
    # TODO: Implement the function.                                           #
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return n_nodes






