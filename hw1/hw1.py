###### Your ID ######
# ID1: 123456789
# ID2: 987654321
#####################

# imports 
import numpy as np
import pandas as pd

def preprocess(X,y):
    """
    Perform mean normalization on the features and true labels.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - X: The mean normalized inputs.
    - y: The mean normalized labels.
    """
    # Calculate the mean (average) of the X,y arrays
    mean_X = np.mean(X)
    mean_y = np.mean(y)

    # Calculate the maximum of the X,y arrays
    max_X = np.max(X)
    max_y = np.max(y)

    # Calculate the minimum of the X,y arrays
    min_X = np.min(X)
    min_y = np.min(y)

    # Normalize X and y
    X = (X - mean_X) / (max_X-min_X)
    y = (y - mean_y) / (max_y-min_y)

    return X, y

def apply_bias_trick(X):
    """
    Applies the bias trick to the input data.

    Input:
    - X: Input data (m instances over n features).

    Returns:
    - X: Input data with an additional column of ones in the
        zeroth position (m instances over n+1 features).
    """    
    # Create an array of ones of length of X
    column0 = np.ones(len(X))

    # concatenates the array column0 with the original array X - adds a column of ones to the beginning of the array X.
    X = np.c_[column0, X]

    return X


    

def compute_cost(X, y, theta):
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
    m = X.shape[0]  # Number of instances
    # Calculate J with the cost function formula
    for i in range(1, m+1): # For every instance
        prediction = np.dot(theta, X[i-1]) # theta dot X(i)
        actual_value = y[i-1] # y(i)
        error = prediction - actual_value
        J += error ** 2
    J /= 2*m # Average all m instances to get the cost

    return J

def gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of the model using gradient descent using 
    the training set. Gradient descent is an optimization algorithm 
    used to minimize some (loss) function by iteratively moving in 
    the direction of steepest descent as defined by the negative of 
    the gradient. We use gradient descent to update the parameters
    (weights) of our model.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy()  # optional: theta outside the function will not change
    J_history = []  # Use a python list to save the loss value in every iteration
    m = X.shape[0]  # Number of instances
    temp_theta_values = [0] * len(theta) # Use a python list to do simultaneous thetas values updated

    for iteration in range(num_iters):
        for j in range(len(theta)):
            sigma = 0

            for i in range(1, m+1): # For every instance
                prediction = np.dot(theta, X[i-1]) # theta dot X(i)
                actual_value = y[i-1] # y(i)
                error = prediction - actual_value
                sigma += error * X[i-1][j] # Relevant feature value
            sigma /= m
            sigma *= alpha  # alpha times the partial derivative of the error function with respect theta j
            temp_theta_values[j] = theta[j] - sigma
        theta = temp_theta_values # Set the new theta values
        J_history.append(compute_cost(X,y,theta))

    return theta, J_history

def compute_pinv(X, y):
    """
    Compute the optimal values of the parameters using the pseudoinverse
    approach as you saw in class using the training set.

    #########################################
    #### Note: DO NOT USE np.linalg.pinv ####
    #########################################

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).

    Returns:
    - pinv_theta: The optimal parameters of your model.
    """
    
    pinv_theta = []
    X_transposed = np.transpose(X)
    X_transposed_dot_X = np.linalg.inv(np.dot(X_transposed, X))
    pinv_X = np.dot(X_transposed_dot_X, X_transposed)
    pinv_theta = np.dot(pinv_X, y)
    return pinv_theta

def efficient_gradient_descent(X, y, theta, alpha, num_iters):
    """
    Learn the parameters of your model using the training set, but stop 
    the learning process once the improvement of the loss value is smaller 
    than 1e-8. This function is very similar to the gradient descent 
    function you already implemented.

    Input:
    - X: Input data (m instances over n features).
    - y: True labels (m instances).
    - theta: The parameters (weights) of the model being learned.
    - alpha: The learning rate of your model.
    - num_iters: The number of updates performed.

    Returns:
    - theta: The learned parameters of your model.
    - J_history: the loss value for every iteration.
    """
    
    theta = theta.copy() # optional: theta outside the function will not change
    J_history = [] # Use a python list to save the cost value in every iteration
    m = X.shape[0]  # Number of instances
    temp_theta_values = [0] * len(theta) # Use a python list to do simultaneous thetas values updated

    for iteration in range(num_iters):
        for j in range(len(theta)):
            sigma = 0

            for i in range(1, m+1): # For every instance
                prediction = np.dot(theta, X[i-1]) # theta dot X(i)
                actual_value = y[i-1] # y(i)
                error = prediction - actual_value
                sigma += error * X[i-1][j] # Relevant feature value
            sigma /= m
            sigma *= alpha  # alpha times the partial derivative of the error function with respect theta j
            temp_theta_values[j] = theta[j] - sigma
        theta = temp_theta_values # Set the new theta values
        J_history.append(compute_cost(X,y,theta))
        # If the improvement of the loss value is smaller than 1e-8, stop the learning process
        if iteration > 0 and J_history[iteration - 1] - J_history[iteration] < 1e-8:
            break
    return theta, J_history

def find_best_alpha(X_train, y_train, X_val, y_val, iterations):
    """
    Iterate over the provided values of alpha and train a model using 
    the training dataset. maintain a python dictionary with alpha as the 
    key and the loss on the validation set as the value.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the training and validation data
    - iterations: maximum number of iterations

    Returns:
    - alpha_dict: A python dictionary - {alpha_value : validation_loss}
    """
    
    alphas = [0.00001, 0.00003, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3]
    alpha_dict = {} # {alpha_value: validation_loss}
    random_theta=np.random.random(size=X_train.shape[1])
    for alpha in alphas:
        theta, J_history = efficient_gradient_descent(X_train, y_train, random_theta, alpha, iterations)
        validation_loss = compute_cost(X_val, y_val, theta)
        alpha_dict[alpha] = validation_loss

    return alpha_dict

def forward_feature_selection(X_train, y_train, X_val, y_val, best_alpha, iterations):
    """
    Forward feature selection is a greedy, iterative algorithm used to 
    select the most relevant features for a predictive model. The objective 
    of this algorithm is to improve the model's performance by identifying 
    and using only the most relevant features, potentially reducing overfitting, 
    improving accuracy, and reducing computational cost.

    You should use the efficient version of gradient descent for this part. 

    Input:
    - X_train, y_train, X_val, y_val: the input data without bias trick
    - best_alpha: the best learning rate previously obtained
    - iterations: maximum number of iterations for gradient descent

    Returns:
    - selected_features: A list of selected top 5 feature indices
    """
    
    selected_features = []
    for i in range(5):
        np.random.seed(42)
        # create a dict for feature (the column) and its cost
        best_cost = {}
        for feature in range(X_train.shape[1]): #  select a feature (column in data set)
            if feature not in selected_features:
                selected_features.append(feature)
                # slicing to select only the columns that are in selected_features
                theta2 = np.random.random(size=i + 2)
                idx_train, idx_val = apply_bias_trick(X_train[:, selected_features]), apply_bias_trick(X_val[:, selected_features])
                theta, cost_hist = efficient_gradient_descent(idx_train,y_train,theta2, best_alpha, iterations)
                cost = compute_cost(idx_val, y_val, theta)
                best_cost[feature] = cost
                selected_features.pop()
        best_feature = min(best_cost, key=best_cost.get)
        selected_features.append(best_feature)

    return selected_features

def create_square_features(df):
    """
    Create square features for the input data.

    Input:
    - df: Input data (m instances over n features) as a dataframe.

    Returns:
    - df_poly: The input data with polynomial features added as a dataframe
               with appropriate feature names
    """

     # List to store new columns
    new_columns = []
    # Iterate through each column in the original DataFrame
    for col1 in df.columns:
        # Compute square feature
        new_columns.append(df[col1] ** 2)

        # Compute cross-product with other columns
        for col2 in df.columns:
            if col1 != col2: # Don't compute cross-product with itself
                new_columns.append(df[col1] * df[col2])

    # Concatenate all new columns to the original DataFrame
    df_poly = pd.concat(new_columns, axis=1)
    # Rename columns with appropriate names
    column_names = []
    for col1 in df.columns:
        column_names.append(col1+'^2')
        for col2 in df.columns:
            if col1 != col2:
                column_names.append(col1 +'*'+col2)
    df_poly.columns = column_names
    return df_poly

