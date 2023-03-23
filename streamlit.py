#  run this pipreqs ./ to get requirements fle

import streamlit as st 
import numpy as np
from sklearn.datasets import load_breast_cancer #using the breast cancer dataset 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score

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


header = st.container()
dashboard = st.container()

@st.cache_data
def get_data():
    X_bc, y_bc = load_breast_cancer(return_X_y=True)
    X_train_bc, X_test_bc, y_train_bc, y_test_bc = train_test_split(X_bc, y_bc, test_size=0.2, random_state=42)
    return X_train_bc, X_test_bc, y_train_bc, y_test_bc

X_train_bc, X_test_bc, y_train_bc, y_test_bc = get_data()

with header: 
    st.title('Implementation of a decision tree')
    st.header('Classification Tree')
    # st.text('time 1500') #time stamp to check if streamlit web app is updated
    st.text(' ')

with dashboard: 
    st.write('Parameters for the tree are: max_depth and min_samples_split')
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