from tensorflow.keras.layers import Layer, UpSampling2D, Conv2D, LeakyReLU, Dropout, Concatenate, InputSpec
import tensorflow.keras.backend as K
from tensorflow.keras import initializers, regularizers, constraints
import numpy as np

K.set_floatx('float64')

#######################################################################################

class InstanceNormalization(Layer):

    def __init__(self,axis=None,epsilon=1e-3,center=True,scale=True,beta_initializer='zeros',gamma_initializer='ones',
                 beta_regularizer=None,gamma_regularizer=None,beta_constraint=None,gamma_constraint=None, **kwargs):

        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)
        super(InstanceNormalization, self).__init__(**kwargs)

    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        self.input_spec = InputSpec(ndim=ndim)

        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]

        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev

        broadcast_shape = [1] * len(input_shape)

        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]

        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            normed = normed * broadcast_gamma

        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            normed = normed + broadcast_beta

        return normed

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

#######################################################################################

import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras.initializers import Constant

class BatchAttNorm(layers.BatchNormalization):
    def __init__(self, momentum=0.99, epsilon=0.001, axis=-1, **kwargs):
        super(BatchAttNorm, self).__init__(momentum=momentum, epsilon=epsilon, axis=axis, center=False, scale=False, **kwargs)
        
        if self.axis == -1:
            self.data_format = 'channels_last'
        else:
            self.data_format = 'channel_first'
        
    def build(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError('expected 4D input (got {}D input)'.format(input_shape))
                
        super(BatchAttNorm, self).build(input_shape)
        
        dim = input_shape[self.axis]
        shape = (dim, )
        
        self.GlobalAvgPooling = layers.GlobalAveragePooling2D(self.data_format)
        self.GlobalAvgPooling.build(input_shape)
    
        self.weight = self.add_weight(name='weight',
                                      shape=shape,
                                      initializer=Constant(1),
                                      trainable=True)

        self.bias = self.add_weight(name='bias',
                                    shape=shape,
                                    initializer=Constant(0),
                                    trainable=True)

        self.weight_readjust = self.add_weight(name='weight_readjust',
                                               shape=shape,
                                               initializer=Constant(0),
                                               trainable=True)
        
        self.bias_readjust = self.add_weight(name='bias_readjust',
                                             shape=shape,
                                             initializer=Constant(-1),
                                             trainable=True)
        

    def call(self, input):
        
        avg = self.GlobalAvgPooling(input)
        attention = K.sigmoid(avg * self.weight_readjust + self.bias_readjust)

        bn_weights = self.weight * attention
        
        out_bn = super(BatchAttNorm, self).call(input)
        
        if K.int_shape(input)[0] is None or K.int_shape(input)[0] > 1:
            bn_weights = bn_weights[:, None, None, :]
            self.bias  = self.bias[None, None, None, :]

        return out_bn * bn_weights + self.bias

#######################################################################################

def conv2d(layer_input, filters, f_size=5, dropout_rate=0):
	"""Layers used during downsampling"""
	d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
	d = LeakyReLU(alpha=0.3)(d)
	if dropout_rate:
		d = Dropout(dropout_rate)(d)
	d = InstanceNormalization()(d)
	return d

def deconv2d(layer_input, filters, skip_input=None, f_size=5, dropout_rate=0): #U-NET CONCAT
	"""Layers used during upsampling"""
	u = UpSampling2D(size=2)(layer_input)
	u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
	if dropout_rate:
		u = Dropout(dropout_rate)(u)
	u = InstanceNormalization()(u)
	if skip_input.any(): #only if u-net
		u = Concatenate()([u, skip_input])
	return u

def disconv2d(layer_input, filters, f_size=5, normalization=True):
    """Discriminator layer"""
    d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
    d = LeakyReLU(alpha=0.3)(d)
    if normalization:
        d = InstanceNormalization()(d)
    return d

#######################################################################################

if __name__ == '__main__':
	testarray = np.ones((1,32,32,3))
	# testprevconcat = np.ones((1,64,64,16))

	# testarray = deconv2d(testarray, 16, skip_input=testprevconcat)
    # testarray = disconv2d(testarray, 16)

	testarray = conv2d(testarray, 16)
	print(testarray.shape)

#######################################################################################