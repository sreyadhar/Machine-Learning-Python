import numpy as np
from scipy.optimize import minimize
from math import sqrt
import pickle
# %%
'''
You need to modify the functions except for initializeWeights() 
'''

def initializeWeights(n_in, n_out):
    '''
    initializeWeights return the random weights for Neural Network given the
    number of node in the input layer and output layer

    Input:
    n_in: number of nodes of the input layer
    n_out: number of nodes of the output layer

    Output:
    W: matrix of random initial weights with size (n_out x (n_in + 1))
    '''
    np.random.rand(10**3) #(Report 1.1)
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W

# %%
def sigmoid(z):
    '''
    Notice that z can be a scalar, a vector or a matrix
    return the sigmoid of input z (same dimensions as z)
    '''
    # remove the next line and replace it with your code
    sig = 1.0 / (1.0 + np.exp(-1.0 * z))
    return sig 

# %%
def nnObjFunction(params, *args):
    '''
    % nnObjFunction computes the value of objective function (cross-entropy
    % with regularization) given the weights and the training data and lambda
    % - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices W1 (weights of connections from
    %     input layer to hidden layer) and W2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of nodes in input layer (not including the bias node)
    % n_hidden: number of nodes in hidden layer (not including the bias node)
    % n_class: number of nodes in output layer (number of classes in
    %     classification problem
    % train_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of the corresponding instance 
    % train_label: the vector of true labels of training instances. Each entry
    %     in the vector represents the true label of its corresponding training instance.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.

    % Output:
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector (not a matrix) of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.
    '''
    # do not remove the next 5 lines
    n_input, n_hidden, n_class, train_data, train_label, lambdaval = args
    # First reshape 'params' vector into 2 matrices of weights W1 and W2

    W1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    W2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

    # remove the next two lines and replace them with your code 
    # obj_val = 0
    # obj_grad = params 
    
    # one hot encoding of train_level
    y_l= np.zeros((n_class, 1), dtype='int64')  ## 10 * 1
    y_l = (np.arange(n_class) == train_label[:,None]).astype(np.int64) ## 25000 * 10
    train_label = y_l.astype(np.int64) ## 25000 * 10

    
    # Forward Feed Propagation
    bias_hidden = np.ones(train_data.shape[0]) # Added a column for bias with value 1
    
    train_data = np.column_stack([train_data, bias_hidden]) # 25000*785
    
    sum_hidden = np.dot(train_data, W1.T)   # sum_z at hidden layer 25000*50
    output_hidden = sigmoid(sum_hidden)# output at hidden layer 25000*50
    out_bias = np.ones(output_hidden.shape[0]) # Added a column for bias with value 1
    output_hidden_new = np.column_stack([output_hidden, out_bias]) # 25000*51 
    
    sum_out = np.dot(output_hidden_new, W2.T) # sum_l at output layer 25000 * 10
    o_l = sigmoid(sum_out)  # 25000 * 10

    
    # Backward Feed Propagation
    # gradient of W2
    delta_o = (o_l - train_label) # 25000*10
    grad_W2 = np.dot(output_hidden_new.T, delta_o) # 51 * 10 
    
    # gradient of W1   
    W1_1 = np.multiply(np.subtract(1, output_hidden_new), output_hidden_new) # 25000*51
    W1_2 = np.dot(delta_o , W2) # 25000 * 51
    W1_tot = np.multiply(W1_1 , W1_2) # 25000 * 51
    grad_W1 = np.dot(train_data.T, W1_tot) # 785 * 51
    grad_W1 = grad_W1[:,0:n_hidden] # 785 * 50

    # Loss function calculation 
    N = train_data.shape[0] #25000
    loss_1 = np.multiply(y_l, np.log(o_l)) ## 25000 * 10
    loss_2 = np.multiply(np.subtract(1, y_l), np.log(np.subtract(1, o_l))) ## 25000 * 10
    loss_function = - np.mean(np.sum(np.add(loss_1, loss_2), axis= 1)) ## eq. 7 scalar
       
    # Regularization Calculation   
    reg_weights = np.sum(np.sum(np.square(W1)) + np.sum(np.square(W2))) ## scalar
    L2_reg = np.multiply(np.divide(lambdaval, 2*N), reg_weights) ## scalar
    obj_val = np.add(loss_function , L2_reg) ## eq. 15 scalar

    #Update the gradient functions   
    grad_W1_new = np.divide(np.add(grad_W1.T, np.multiply(lambdaval, W1)), N) ## 50 * 785
    grad_W2_new = np.divide(np.add(grad_W2.T, np.multiply(lambdaval, W2)), N) ## 10 * 51
    obj_grad = np.concatenate((grad_W1_new.ravel(), grad_W2_new.ravel()),0) ## (39760, )   
    return (obj_val,obj_grad)

# %%
def nnPredict(W1, W2, data):
    '''
    % nnPredict predicts the label of data given the parameter W1, W2 of Neural
    % Network.

    % Input:
    % W1: matrix of weights for hidden layer units
    % W2: matrix of weights for output layer units
    % data: matrix of data. Each row of this matrix represents the feature vector for the corresponding data instance

    % Output:
    % label: a column vector of predicted labels
    '''
    # remove the next line and replace it with your code
    labels = np.zeros((data.shape[0],1))  
    new_bias_H = np.ones(data.shape[0]) # Added a column for bias with value 1
    data = np.hstack([data, new_bias_H[:, None]])
    sum_hidden = np.dot(data ,W1.T)  # sum_z at hidden layer 25000*784   
    output_hidden = sigmoid(sum_hidden)# output at hidden layer 25000*784  
    new_bias_O = np.ones(output_hidden.shape[0]) # Added a column for bias with value 1
    out_hidden = np.hstack([output_hidden, new_bias_O[:, None]])
    o_l = sigmoid(np.dot(out_hidden,W2.T))  # sig(sum_l) at output layer
    max_val = np.argmax(o_l, axis=1)
    labels = max_val.T
    return labels

# %%