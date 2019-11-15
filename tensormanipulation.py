import tensorflow as tf
import numpy as np



# #batch_size = 3
# #seq_len = 5
# #dim = 2

# # [batch_size x seq_len x dim]  -- hidden states
# Y = tf.constant(np.random.randn(batch_size, seq_len, dim), tf.float32)
# # [batch_size x dim]            -- h_N
# h = tf.constant(np.random.randn(batch_size, dim), tf.float32)

# initializer = tf.random_uniform_initializer()
# W = tf.get_variable("weights_Y", [dim, dim], initializer=initializer)
# w = tf.get_variable("weights_w", [dim], initializer=initializer)

# # [batch_size x seq_len x dim]  -- tanh(W^{Y}Y)
# M = tf.tanh(tf.einsum("aij,jk->aik", Y, W))
# # [batch_size x seq_len]        -- softmax(Y w^T)
# a = tf.nn.softmax(tf.einsum("aij,j->ai", M, w))
# # [batch_size x dim]            -- Ya^T
# r = tf.einsum("aij,ai->aj", Y, a)

# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     a_val, r_val = sess.run([a, r])
#     print("a:", a_val, "\nr:", r_val)



def cos_sim(v1, v2):
    norm1 = tf.sqrt(tf.reduce_sum(tf.square(v1), axis=1))
    norm2 = tf.sqrt(tf.reduce_sum(tf.square(v2), axis=1))
    dot_products = tf.reduce_sum(v1 * v2, axis=1, name="cos_sim")

    return dot_products / (norm1 * norm2)

def euclidean_score(v1, v2):
    euclidean = tf.sqrt(tf.reduce_sum(tf.square(v1 - v2), axis=1))
    return 1 / (1 + euclidean)

def make_attention_mat(x1, x2):
    # x1, x2 = [batch, height, width, 1] = [batch, d, s, 1]
    # x2 => [batch, height, 1, width]
    # [batch, width, wdith] = [batch, s, s]
    euclidean = tf.sqrt(tf.reduce_sum(tf.square(x1 - tf.matrix_transpose(x2)), axis=1))
    return 1 / (1 + euclidean)