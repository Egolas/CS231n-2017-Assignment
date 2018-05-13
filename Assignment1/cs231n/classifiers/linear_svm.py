import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  gradient = np.zeros(W.shape)
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    for j in xrange(num_classes):

      if j == y[i]:
        continue
       # note delta = 1
      margin = scores[j] - correct_class_score + 1
      if margin > 0:
        loss += margin
        gradient[:, y[i]] -= X[i]
        gradient[:, j] += X[i]


  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  gradient /= num_train

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)
  gradient += 2 * reg * W

  dW = gradient
  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################


  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  num_train = X.shape[0]
  M = np.dot(X, W)
  true_class_idx_line = np.arange(X.shape[0])
  true_class_idx_column = y
  true_class_score = M[true_class_idx_line, true_class_idx_column]
  margins = M - true_class_score.reshape(-1, 1) + 1
  margins[true_class_idx_line, true_class_idx_column] = 0

  loss = reg * np.sum(W * W) + np.sum(np.maximum(0, margins)) / num_train

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  # pass
  # sign = np.max(0, margins / np.abs(margins))
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  margins[margins>0] = 1
  margins[margins<=0] = 0
  row_sum = np.sum(margins, axis=1)
  margins[true_class_idx_line, true_class_idx_column] = -row_sum
  dW += np.dot(X.T, margins) / num_train
  dW += 2 * reg * W

  return loss, dW
