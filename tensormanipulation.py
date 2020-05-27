import tensorflow as tf
import numpy as np

#STUFF

def normalization(p):
    """
    :param p: list of values
    :return: return min-max normalization
    """
    r = []
    min_v = min(p)
    max_v = max(p)
    for i in p:
        r.append((i-min_v)/(max_v - min_v))
    return r

def ecludian_distance(p, q):
    n = len(p)
    assert n == len(q)
    return tf.sqrt(sum([pow(p[i] - q[i], 2) for i in range(n)]))

#VECTORS

def cosine_similarity(p, q):
    """
    :param p: document vector, eg = [1,2, 1, 0.6]
    :param q: query vector, eg = [0.4, 1, 0]
    :return:
    """
    n = len(p)
    assert n == len(q)
    dot = sum([a*b for a, b in zip(p, q)])
    x = tf.sqrt(sum([pow(i, 2) for i in p]))
    y = tf.sqrt(sum([pow(i, 2) for i in q]))

    return dot / (x*y)

def cos_sim(v1, v2):
    norm1 = tf.sqrt(tf.reduce_sum(tf.square(v1), axis=1))
    norm2 = tf.sqrt(tf.reduce_sum(tf.square(v2), axis=1))
    dot_products = tf.reduce_sum(v1 * v2, axis=1, name="cos_sim")

    return dot_products / (norm1 * norm2)

def euclidean_score(v1, v2):
    euclidean = tf.sqrt(tf.reduce_sum(tf.square(v1 - v2), axis=1))
    return 1 / (1 + euclidean)

#MATRIX OPERATIONS

def add(matrix_a, matrix_b):
    if _check_not_integer(matrix_a) and _check_not_integer(matrix_b):
        rows, cols = _verify_matrix_sizes(matrix_a, matrix_b)
        matrix_c = []
        for i in range(rows[0]):
            list_1 = []
            for j in range(cols[0]):
                val = matrix_a[i][j] + matrix_b[i][j]
                list_1.append(val)
            matrix_c.append(list_1)
        return matrix_c

def subtract(matrix_a, matrix_b):
    if _check_not_integer(matrix_a) and _check_not_integer(matrix_b):
        rows, cols = _verify_matrix_sizes(matrix_a, matrix_b)
        matrix_c = []
        for i in range(rows[0]):
            list_1 = []
            for j in range(cols[0]):
                val = matrix_a[i][j] - matrix_b[i][j]
                list_1.append(val)
            matrix_c.append(list_1)
        return matrix_c

def scalar_multiply(matrix, n):
    return [[x * n for x in row] for row in matrix]

def multiply(matrix_a, matrix_b):
    if _check_not_integer(matrix_a) and _check_not_integer(matrix_b):
        matrix_c = []
        rows, cols = _verify_matrix_sizes(matrix_a, matrix_b)

        if cols[0] != rows[1]:
            raise ValueError(
                f"Cannot multiply matrix of dimensions ({rows[0]},{cols[0]}) "
                f"and ({rows[1]},{cols[1]})"
            )
        for i in range(rows[0]):
            list_1 = []
            for j in range(cols[1]):
                val = 0
                for k in range(cols[1]):
                    val = val + matrix_a[i][k] * matrix_b[k][j]
                list_1.append(val)
            matrix_c.append(list_1)
        return matrix_c

def identity(n):
    """
    :param n: dimension for nxn matrix
    :type n: int
    :return: Identity matrix of shape [n, n]
    """
    n = int(n)
    return [[int(row == column) for column in range(n)] for row in range(n)]

def transpose(matrix, return_map=True):
    if _check_not_integer(matrix):
        if return_map:
            return map(list, zip(*matrix))
        else:
            # mt = []
            # for i in range(len(matrix[0])):
            #     mt.append([row[i] for row in matrix])
            # return mt
            return [[row[i] for row in matrix] for i in range(len(matrix[0]))]

def minor(matrix, row, column):
    minor = matrix[:row] + matrix[row + 1 :]
    minor = [row[:column] + row[column + 1 :] for row in minor]
    return minor

def determinant(matrix):
    if len(matrix) == 1:
        return matrix[0][0]

    res = 0
    for x in range(len(matrix)):
        res += matrix[0][x] * determinant(minor(matrix, 0, x)) * (-1) ** x
    return res

def inverse(matrix):
    det = determinant(matrix)
    if det == 0:
        return None

    matrix_minor = [[] for _ in range(len(matrix))]
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            matrix_minor[i].append(determinant(minor(matrix, i, j)))

    cofactors = [[x * (-1) ** (row + col) for col, x in enumerate(matrix_minor[row])] for row in range(len(matrix))]
    adjugate = transpose(cofactors)
    return scalar_multiply(adjugate, 1 / det)

#MATRIX UTILS

def _check_not_integer(matrix):
    try:
        rows = len(matrix)
        cols = len(matrix[0])
        return True
    except TypeError:
        raise TypeError("Cannot input an integer value, it must be a matrix")

def _shape(matrix):
    return list((len(matrix), len(matrix[0])))

def _verify_matrix_sizes(matrix_a, matrix_b):
    shape = _shape(matrix_a)
    shape += _shape(matrix_b)
    if shape[0] != shape[2] or shape[1] != shape[3]:
        raise ValueError(
            f"operands could not be broadcast together with shape "
            f"({shape[0], shape[1]}), ({shape[2], shape[3]})"
        )
    return [shape[0], shape[2]], [shape[1], shape[3]]

#ADVANCED OPERATIONS

def make_attention_mat(x1, x2):
    # x1, x2 = [batch, height, width, 1] = [batch, d, s, 1]
    # x2 => [batch, height, 1, width]
    # [batch, width, wdith] = [batch, s, s]
    euclidean = tf.sqrt(tf.reduce_sum(tf.square(x1 - tf.matrix_transpose(x2)), axis=1))
    return 1 / (1 + euclidean)
