import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

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
  # Initialize the loss and gradient to zero.
  num_train = X.shape[0]
  num_class = W.shape[1]
  loss = 0.0
  dW = np.zeros_like(W)


  for i in range(num_train):
    scores = X[i].dot(W)
    scores -= np.max(scores)
    exp_correct_class_score = np.exp(scores[y[i]])
    score_sum = 0.0
    tmp_g = np.zeros(W.shape)
    for j in range(num_class):
      score_sum += np.exp(scores[j])
      if j != y[i]:
        inter = X[i] * np.exp(scores[j])
        tmp_g[:, j] += inter
        tmp_g[:, y[i]] += -inter

    tmp_g /= score_sum
    dW += tmp_g
    loss += -np.log(exp_correct_class_score / score_sum)

  dW /= num_train
  dW += 2 * reg * W
  loss /= num_train
  loss += reg * np.sum(W * W)








  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  num_train = X.shape[0]
  num_class = W.shape[1]
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  scores = X.dot(W)
  scores -= np.max(scores, axis=1).reshape(-1, 1)
  scores = np.exp(scores)
  true_class_idx_line = np.arange(X.shape[0])
  true_class_idx_column = y
  true_class_scores = scores[true_class_idx_line, true_class_idx_column]
  loss += np.sum(-np.log(true_class_scores / np.sum(scores, axis=1)))


  scores_sum = np.sum(scores, axis=1).reshape(-1, 1)
  scores[true_class_idx_line, true_class_idx_column] = 0
  scores[true_class_idx_line, true_class_idx_column] = -np.sum(scores, axis=1)
  scores /= scores_sum

  dW = np.dot(X.T, scores)

  dW /= num_train
  loss /= num_train

  dW += 2 * reg * W
  loss += reg * np.sum(W * W)




  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

