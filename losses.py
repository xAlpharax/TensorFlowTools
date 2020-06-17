import tensorflow.keras.backend as K
import numpy as np

#######################################################################################

def categorical_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0):
    y_pred = K.constant(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)

    if label_smoothing is not 0:
        smoothing = K.cast_to_floatx(label_smoothing)

        def _smooth_labels():
            num_classes = K.cast(K.shape(y_true)[1], y_pred.dtype)
            return y_true * (1.0 - smoothing) + (smoothing / num_classes)

        y_true = K.switch(K.greater(smoothing, 0), _smooth_labels, lambda: y_true)
    return K.categorical_crossentropy(y_true, y_pred, from_logits=from_logits)


def binary_crossentropy(y_true, y_pred, from_logits=False, label_smoothing=0):
    y_pred = K.constant(y_pred)
    y_true = K.cast(y_true, y_pred.dtype)
    if label_smoothing is not 0:
        smoothing = K.cast_to_floatx(label_smoothing)
        y_true = K.switch(K.greater(smoothing, 0),
                          lambda: y_true * (1.0 - smoothing) + 0.5 * smoothing,
                          lambda: y_true)
    return K.mean(K.binary_crossentropy(y_true, y_pred, from_logits=from_logits), axis=-1)

def sparse_categorical_crossentropy(y_true, y_pred, from_logits=False, axis=-1):
    return K.sparse_categorical_crossentropy(y_true, y_pred, from_logits=from_logits, axis=axis)

#######################################################################################

if __name__ == '__main__':

    y = np.array([[0,1]])
    pred = np.array([[0.1, 0.7]])
    #pred = np.array([[0,1]])

    #print(categorical_crossentropy(y,pred))
    print(binary_crossentropy(y,pred))

#######################################################################################