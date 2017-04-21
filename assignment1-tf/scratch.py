"""
Some scratch functions I used to figure out how to get a few things to work in TF
"""

import numpy as np
import tensorflow as tf


def demo_select_indexes():
    """
    Given a 2d vector of values and a 1-d vector of indices
    we wish to select from each row of the 2d vector.
    """

    ps = tf.constant([[0.2, 0.8],
                      [0.4, 0.6],
                      [0.25, 0.75]])

    idxs = tf.constant([1, 0, 1])

    y = tf.gather_nd(
        ps,
        tf.transpose(tf.stack([tf.range(idxs.shape[0]), idxs]))) # [[0, 1], [1, 0], [2, 1]]

    with tf.Session('') as sess:
        print(sess.run(y))


def demo_select_indexes_dynamic_dimensions():
    """
    Given a 2d vector of values and a 1-d vector of indices
    we wish to select from each row of the 2d vector.
    """

    ps = tf.placeholder(tf.float32, [None, 2])
    idxs = tf.placeholder(tf.int32, [None])

    y = tf.gather_nd(
        ps,
        tf.transpose(tf.stack([tf.range(tf.shape(idxs)[0]), idxs])))

    with tf.Session('') as sess:
        print(sess.run(y, feed_dict={
            ps: [[0.2, 0.8],
                 [0.4, 0.6],
                 [0.25, 0.75]],
            idxs: [1, 0, 1]
        }))
        print(sess.run(y, feed_dict={
            ps: [[0.2, 0.8],
                 [0.4, 0.6],
                 [0.4, 0.6],
                 [0.4, 0.6],
                 [0.25, 0.75]],
            idxs: [1, 0, 0, 1, 1]
        }))


def one_hot(a, num_classes=10):
    one_hot_a = np.zeros((a.shape[0], num_classes), dtype=np.int64)
    one_hot_a[range(a.shape[0]), a] = 1
    return one_hot_a

def demo_64_range():
    ys = tf.constant([1, 1, 2, 3, 4], dtype=tf.int64)
    r = tf.range(tf.cast(tf.shape(ys)[0], tf.int64), dtype=tf.int64)
    with tf.Session('') as sess:
        print(sess.run(r))

def demo_select_index_from_one_hot():

    ys_onehot = tf.placeholder(tf.int64, [None, 10])
    print(ys_onehot)
    ys = tf.argmax(ys_onehot, 1)

    ps = tf.placeholder(tf.float32, [None, 2])

    y = tf.gather_nd(
        ps,
        tf.transpose(tf.stack(
            [tf.range(tf.cast(tf.shape(ys)[0], tf.int64),
                      dtype=tf.int64),
             ys])))

    mean_y = tf.reduce_mean(
        tf.reduce_sum(y)
    )

    with tf.Session('') as sess:
        print(sess.run(mean_y, feed_dict={
            ps: [[0.2, 0.8],
                 [0.4, 0.6],
                 [0.25, 0.75]],
            ys_onehot: one_hot(np.array([1, 0, 1]))
        }))
        print(sess.run(mean_y, feed_dict={
            ps: [[0.2, 0.8],
                 [0.4, 0.6],
                 [0.4, 0.6],
                 [0.4, 0.6],
                 [0.25, 0.75]],
            ys_onehot: one_hot(np.array([1, 0, 0, 1, 1]))
        }))

def np_softmax_loss(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """

    num_train = X.shape[0]

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    scores = X.dot(W)
    scores_shifted = scores - scores.max(axis=1).reshape(scores.shape[0], -1)
    exp_scores = np.exp(scores_shifted)
    exp_scores_sum = np.sum(exp_scores, axis=1).reshape(exp_scores.shape[0], -1)
    p = exp_scores / exp_scores_sum
    # p = exp_scores / np.sum(exp_scores)
    p_correct = np.choose(y, p.T)
    loss = np.sum(-np.log(p_correct))

    loss /= num_train

    loss += reg * np.sum(W * W)

    return p_correct, loss


def tf_softmax_loss(W, X, y, reg):
    tf_W = tf.placeholder(tf.float32, [None, None], name='W')
    tf_X = tf.placeholder(tf.float32, [None, None], name='X')
    tf_y = tf.placeholder(tf.int32, [None], name='y')
    tf_reg = tf.placeholder(tf.float32)

    scores = tf.matmul(tf_X, tf_W, name='scores')
    # normalize for numerical stability http://cs231n.github.io/linear-classify/#softmax
    max_scores = tf.reduce_max(scores, reduction_indices=[1], keep_dims=True, name='max_scores')
    scores_stable = tf.subtract(scores, max_scores, name='scores_stable')
    exp_scores = tf.exp(scores_stable)
    exp_scores_sum = tf.reduce_sum(exp_scores, axis=1, keep_dims=True)

    p = exp_scores / exp_scores_sum


    p_correct = tf.gather_nd(
        p,
        tf.transpose(tf.stack([tf.range(tf.shape(tf_y)[0]),
                               tf_y])))
    loss = tf.reduce_mean(-tf.log(p_correct))

    regularized_loss = loss + reg * tf.reduce_sum(tf.square(tf_W))

    with tf.Session('') as sess:
        return sess.run(
            [p_correct, regularized_loss],
            feed_dict={
                tf_W: W,
                tf_X: X,
                tf_y: y,
                tf_reg: reg
            })


def sample_images(num_pixels, num_images):
    return np.clip(
        (np.ones((num_images, num_pixels)) * 100 + np.random.randn(num_images, num_pixels) * 50).astype(np.int32),
        0, 255)

def demo_np_vs_tf_softmax():
    """
    let's say we are predicting 4 classes for 3x3 images for 10 images
    X is a 10x9
    y is 10x1 (each element an index 0 to 3 saying which class)
    W is 9x4 (pixel weights for each class)

    """
    X = sample_images(9, 10)
    W = np.random.randn(9, 4) * 0.0001
    y = np.array([0, 1, 1, 1, 3, 2, 0, 2, 0, 3])

    print("np softmax:\n{}".format("\n-----\n".join(['{}'.format(el) for el in np_softmax_loss(W, X, y, 0.00001)])))
    print("\n\ntf softmax:\n{}".format("\n-----\n".join(['{}'.format(el) for el in tf_softmax_loss(W, X, y, 0.00001)])))

# demo_select_indexes()
# demo_select_indexes_dynamic_dimensions()
# demo_64_range()
# demo_select_index_from_one_hot()
demo_np_vs_tf_softmax()
