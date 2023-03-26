#  run this pipreqs ./ to get requirements fle

import streamlit as st 
import numpy as np
from sklearn.datasets import load_breast_cancer #using the breast cancer dataset 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_diabetes

###### -----------classification tree functions----------- #####
def entropy(y):
    unique, counts = np.unique(y, return_counts=True) #count the unique value occurence in y
    p = counts / len(y) #where p is the prob of the unique value in y
    return -np.sum(p * np.log2(p)) #the entropy euqation is this 

def get_gain(X, y, feature_idex, threshold):
    left_idexs = X[:, feature_idex] < threshold #if value is smaller than threshold, assign to the left
    right_idexs = X[:, feature_idex] >= threshold #otherwise, assign to the right

    number_left, number_right = len(y[left_idexs]), len(y[right_idexs]) #check the number of exaples on left and right split 
    n_total = number_left + number_right

    if number_left == 0 or number_right == 0:
        return 0 #if either side does not have any example, it is not good for learning, then the info gain is 0

    gain = entropy(y) - (number_left/n_total)*entropy(y[left_idexs]) - (number_right/n_total)*entropy(y[right_idexs])
    #otherwise, we calculate the info gain by taking the entropy of parent node - weighted entropy left - weighted entropy right

    return gain

def get_best_feature(X, y, feature_idexs):
    best_gain = -1 #initialise best gain
    split_idex, split_threshold = None, None

    for feature_idex in feature_idexs:
        thresholds = np.unique(X[:, feature_idex]) #go through each feature in X
        for threshold in thresholds:
            gain = get_gain(X, y, feature_idex, threshold) #within each feature, set each unique value to threshold and calculate the info gain
            if gain > best_gain:
                best_gain = gain 
                split_idex = feature_idex
                split_threshold = threshold

    return split_idex, split_threshold #eventually, we will get the best index to split and the best threshold for that feature 

def build_tree(X, y, depth=0, max_depth=5, min_samples_split=10):
    n_samples = X.shape[0] #the number of samples I have in the dataset 
    n_labels = len(np.unique(y)) # the unique labels in the dataset 

    #stop conditions
    #there are three scenarios: 
    #when current depth is exceeding max depth (default is 10)
    #when all labels in the split is the same (meaning unique value of labe in the node = 1)
    #when current split has fewer samples than min_samples_split
    if depth >= max_depth or n_labels == 1 or n_samples < min_samples_split:
        unique, counts = np.unique(y, return_counts=True)
        most_common_label = unique[np.argmax(counts)]
        return {'leaf_value': most_common_label, 'n_sample_count': n_samples} #then return a leaf node with a predicted label

    # Find the best feature to split on:
    feature_idexs = np.random.choice(X.shape[1], int(np.sqrt(X.shape[1])), replace=False) 
    #size of randomly selected features is the square root of the number of features rounded down to the nearest integer
    best_feature, best_threshold = get_best_feature(X, y, feature_idexs)
    #use get_best_feature function to get the best feature and the threshold

    #split data to left and right based on best feature and threshold
    left_idexs = X[:, best_feature] < best_threshold
    right_idexs = X[:, best_feature] >= best_threshold

    #build tree from the left and right index
    left = build_tree(X[left_idexs], y[left_idexs], depth+1, max_depth, min_samples_split)
    right = build_tree(X[right_idexs], y[right_idexs], depth+1, max_depth, min_samples_split)

    #create a dictionary to represent the current node, that is best feature, best threshold, the left and right sub tree
    return {'feature_idx': best_feature, 'threshold': best_threshold,
            'left': left, 'right': right, 'n_sample_count': n_samples}

def predict(X, tree):
    def traverse_tree(x, node):
        if 'leaf_value' in node:
            return node['leaf_value'] #if this is a leaf node then tree should return a predicted label 

        if x[node['feature_idx']] < node['threshold']:
            return traverse_tree(x, node['left'])
        else:
            return traverse_tree(x, node['right'])
        #either going to the left or to the right depending on the value and the threshold

    return np.array([traverse_tree(x, tree) for x in X])

def build_tree_string(node, feature_names, target_names, depth=0):
    indent = '  ' * depth
    if 'leaf_value' in node:
        target_name = target_names[node['leaf_value']]
        n_samples = node['n_sample_count']
        return f"{indent}*[{target_name}]----------({n_samples} samples)\n"
    else:
        feature_name = feature_names[node['feature_idx']]
        left_subtree = build_tree_string(node['left'], feature_names, target_names, depth+1)
        right_subtree = build_tree_string(node['right'], feature_names, target_names, depth+1)
        return f"{indent}[{feature_name} < {node['threshold']:.3f}]\n{left_subtree}{right_subtree}"
    
# def print_tree(node, feature_names, target_names, depth=0):
#     indent = '  ' * depth
#     if 'leaf_value' in node:
#         target_name = target_names[node['leaf_value']]
#         n_samples = node['n_sample_count']
#         st.write(f"{indent}*[{target_name}]({n_samples} samples)")
#     else:
#         feature_name = feature_names[node['feature_idx']]
#         st.write(f"{indent}[{feature_name} < {node['threshold']:.3f}]")
#         print_tree(node['left'], feature_names, target_names, depth+1)
#         print_tree(node['right'], feature_names, target_names, depth+1)


###### -----------regression tree functions----------- #####
# Define the mean squared error (MSE) function
def mse(y):
    return np.mean((y - np.mean(y))**2)

# Define a function to split the data into left and right subsets based on a threshold
def split(X, y, feature_idx, threshold):
    left_indices = np.where(X[:, feature_idx] <= threshold)[0]
    right_indices = np.where(X[:, feature_idx] > threshold)[0]
    if len(left_indices) == 0 or len(right_indices) == 0:
        return None, None, None, None
    else:
        return X[left_indices], y[left_indices], X[right_indices], y[right_indices]

# Define a function to find the best split based on minimizing the MSE
def best_split(X, y, min_samples_split_set=2):
    best_feature_idx, best_threshold, best_mse = None, None, np.inf
    # Loop through all the features in X
    for feature_idx in range(X.shape[1]):
        # Find all the unique values of the feature
        thresholds = np.unique(X[:, feature_idx])
        # Loop through all the unique values of the feature
        for threshold in thresholds:
            # Split the data into left and right subsets based on the threshold
            X_left, y_left, X_right, y_right = split(X, y, feature_idx, threshold)
            # Check that the split is valid (i.e. neither the left nor right subset is empty)
            if y_left is not None and y_right is not None:
                # Check that the number of samples in each subset is greater than or equal to min_samples_split_set
                if len(y_left) < min_samples_split_set or len(y_right) < min_samples_split_set:
                    continue
                # Calculate the total MSE of the left and right subsets
                total_mse = mse(y_left) + mse(y_right)
                # Update the best split if the total MSE is lower than the current best MSE
                if total_mse < best_mse:
                    best_feature_idx, best_threshold, best_mse = feature_idx, threshold, total_mse
    return best_feature_idx, best_threshold
# Define a function to build the decision tree recursively
def build_tree_regression(X, y, depth, max_depth, min_samples_split):
    # Check if the maximum depth has been reached or if there is no further reduction in MSE
    if depth == max_depth or mse(y) == 0 or len(X) < min_samples_split:
        # Return the mean target value
        return np.mean(y)
    # Find the best split based on minimizing the MSE
    feature_idx, threshold = best_split(X, y)
    # Check if there is no valid split
    if feature_idx is None:
        # Return the mean target value
        return np.mean(y)
    # Split the data into left and right subsets based on the best split
    else:
        X_left, y_left, X_right, y_right = split(X, y, feature_idx, threshold)
        # Recursively build the left and right subtrees
        left_tree = build_tree_regression(X_left, y_left, depth+1, max_depth, min_samples_split)
        right_tree = build_tree_regression(X_right, y_right, depth+1, max_depth, min_samples_split)
        # Return the decision node with the best split and the left and right subtrees
        return (feature_idx, threshold, left_tree, right_tree)
# Define a function to make predictions for a single input using the decision tree


def predict_one(x, tree):
    # Check if the current node is a leaf node (i.e. a float value)
    if isinstance(tree, float):
        # Return the mean target value of the leaf node
        return tree
    
    # Check which subtree to go down based on the feature value of the current data point
    else:
        feature_idx, threshold, left_tree, right_tree = tree
        if x[feature_idx] <= threshold:
            # Recursively go down the left subtree
            return predict_one(x, left_tree)
        else:
            # Recursively go down the right subtree
            return predict_one(x, right_tree)


# A function to predict the target values of multiple data points using a decision tree
def predict_regression(X, tree):
    # Predict the target value of each data point using the predict_one function
    return np.array([predict_one(x, tree) for x in X])


# A function to build a decision tree for regression
def decision_tree_regression(X_train, y_train, max_depth, min_samples_split_set):
    # Build the decision tree using the build_tree function
    tree = build_tree_regression(X_train, y_train, 0, max_depth, min_samples_split_set)
    # Predict the target values of the training set using the predict function
    y_pred = predict_regression(X_train, tree)
    # Return the decision tree and the predicted target values of the training set
    return tree, y_pred

def print_tree_regression(node, feature_names, target_names, depth=0):
    indent = '  ' * depth
    if isinstance(node, float):
        print(f"{indent}*[{node:.3f}]")
    else:
        feature_name = feature_names[node[0]]
        print(f"{indent}[{feature_name} < {node[1]:.3f}]")
        print_tree_regression(node[2], feature_names, target_names, depth+1)
        print_tree_regression(node[3], feature_names, target_names, depth+1)

def build_regression_tree_string(node, feature_names, target_names, depth=0):
    indent = '  ' * depth
    if isinstance(node, float):
        return f"{indent}*[{node:.3f}]\n"
    else:
        feature_name = feature_names[node[0]]
        left_subtree = build_regression_tree_string(node[2], feature_names, target_names, depth+1)
        right_subtree = build_regression_tree_string(node[3], feature_names, target_names, depth+1)
        return f"{indent}[{feature_name} < {node[1]:.3f}]\n{left_subtree}{right_subtree}"




###### -----------start code----------- #####

header_classification = st.container()
dashboard_classification = st.container()
header_regression = st.container()
dashboard_regression = st.container()


@st.cache_data
def get_data():
    X_bc, y_bc = load_breast_cancer(return_X_y=True)
    X_train_bc, X_test_bc, y_train_bc, y_test_bc = train_test_split(X_bc, y_bc, test_size=0.2, random_state=42)
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    return X_train_bc, X_test_bc, y_train_bc, y_test_bc, X, y

X_train_bc, X_test_bc, y_train_bc, y_test_bc, X, y = get_data()

with header_classification: 
    st.title('Implementation of a decision tree')
    st.header('Classification Tree')
    # st.text('time 1500') #time stamp to check if streamlit web app is updated
    st.text(' ')

with dashboard_classification: 
    st.write('Parameters for the classification tree are: max_depth and min_samples_split')
    max_depth_set = st.slider('The maximum depth of the decision tree is ', min_value = 1, max_value = 10, value = 5, step = 1)
    min_samples_split_set = st.slider('The minimum sample of a leaf in a split is ', min_value = 1, max_value = 20, value = 10, step = 1)
    st.write('max_depth_set is ', max_depth_set)
    st.write('min_samples_split_set is ', min_samples_split_set)
    st.write(' ')
    
    tree_bc = build_tree(X_train_bc, y_train_bc, max_depth=max_depth_set, min_samples_split=min_samples_split_set)
    y_pred_bc = predict(X_test_bc, tree_bc)

    accuracy_bc = accuracy_score(y_test_bc, y_pred_bc)
    # st.write('The accuracy score of the tree is ', accuracy_bc)

    st.write('This is the trained tree result ')
    tree_str = build_tree_string(tree_bc, load_breast_cancer().feature_names, load_breast_cancer().target_names)
    st.text(tree_str)

with header_regression: 
    st.header('Regression Tree')
    st.text(' ')

with dashboard_regression: 
    st.write('Parameters for the regression tree are: max_depth and min_samples_split')
    max_depth_set_regression = st.slider('The maximum depth of the decision tree is ', min_value = 1, max_value = 10, value = 2, step = 1, key='slider_key_regression_1')
    min_samples_split_set_regression = st.slider('The minimum sample of a leaf in a split is ', min_value = 1, max_value = 20, value = 10, step = 1, key='slider_key_regression_2')
    st.write('max_depth_set is ', max_depth_set_regression)
    st.write('min_samples_split_set is ', min_samples_split_set_regression)
    st.write(' ')

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    max_depth = max_depth_set_regression
    min_samples_split_set = min_samples_split_set_regression
    tree_regression, y_pred_train = decision_tree_regression(X_train, y_train, max_depth, min_samples_split_set)

    # Make predictions on the testing set
    y_pred_test = predict_regression(X_test, tree_regression)

    tree_final_regression = build_tree_regression(X, y, 0, max_depth=max_depth_set_regression,min_samples_split=min_samples_split_set_regression)
    target_names = ['target']
    feature_names = load_diabetes().feature_names
    tree_str = build_regression_tree_string(tree_final_regression, feature_names, target_names)
    st.text(tree_str)
