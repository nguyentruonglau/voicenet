from keras.layers import Dropout, Input, Add, Dense, Activation, ZeroPadding2D
from keras.layers import BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
from keras.initializers import glorot_uniform
import scipy.misc
import tensorflow as tf
from keras.layers import ReLU, DepthwiseConv2D, Add
from keras.layers import GlobalAveragePooling2D, Reshape, multiply, Softmax
from keras.utils.generic_utils import get_custom_objects
import numpy as np
from sklearn import metrics
import keras.backend as K

"""I. MODEL"""

def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 3
    
    Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        f -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network
    
    Returns:
        X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """
    
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    
    # Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X


def convolutional_block(X, f, filters, stage, block, s):
    """
    Implementation of the convolutional block as defined in Figure 4
    
    Arguments:
        X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
        f -- integer, specifying the shape of the middle CONV's window for the main path
        filters -- python list of integers, defining the number of filters in the CONV layers of the main path
        stage -- integer, used to name the layers, depending on their position in the network
        block -- string/character, used to name the layers, depending on their position in the network
        s -- Integer, specifying the stride to be used
        
    Returns:
        X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path 
    X = Conv2D(F1, (1, 1), strides = (s,s), kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)

    # Second component of main path (≈3 lines)
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3)(X)
    X = Activation('relu')(X)


    # Third component of main path (≈2 lines)
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3)(X)


    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(filters = F3, kernel_size = (1, 1), strides = (s,s), padding = 'valid', kernel_initializer = glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3)(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
  
    return X


def __conv2d_block(_inputs, filters, kernel, strides, is_use_bias=False, padding='same', activation='RE', name=None):
    x = Conv2D(filters, kernel, strides= strides, padding=padding, use_bias=is_use_bias)(_inputs)
    x = BatchNormalization()(x)
    if activation == 'RE':
        x = ReLU(name=name)(x)
    elif activation == 'HS':
        x = Activation('relu', name=name)(x)
    else:
        raise NotImplementedError
    return x



def __depthwise_block(_inputs, kernel=(3, 3), strides=(1, 1), activation='RE', is_use_se=True, num_layers=0):
    x = DepthwiseConv2D(kernel_size=kernel, strides=strides, depth_multiplier=1, padding='same')(_inputs)
    x = BatchNormalization()(x)
    if is_use_se:
        x = __se_block(x)
    if activation == 'RE':
        x = ReLU()(x)
    elif activation == 'HS':
        x = Activation('relu')(x)
    else:
        raise NotImplementedError
    return x

def __se_block(_inputs, ratio=4, pooling_type='avg'):
    filters = _inputs.shape[-1]
    se_shape = (1, 1, filters)
    if pooling_type == 'avg':
        se = GlobalAveragePooling2D()(_inputs)
    elif pooling_type == 'depthwise':
        se = __global_depthwise_block(_inputs)
    else:
        raise NotImplementedError
    se = Reshape(se_shape)(se)
    se = Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = Dense(filters, activation='hard_sigmoid', kernel_initializer='he_normal', use_bias=False)(se)
    return multiply([_inputs, se])


def __bottleneck_block(_inputs, out_dim, kernel, strides, expansion_dim, is_use_bias=False, shortcut=True, is_use_se=True, activation='RE', num_layers=0, *args):
    with tf.name_scope('bottleneck_block'):
        # ** to high dim 
        bottleneck_dim = expansion_dim

        # ** pointwise conv 
        x = __conv2d_block(_inputs, bottleneck_dim, kernel=(1, 1), strides=(1, 1), is_use_bias=is_use_bias, activation=activation)

        # ** depthwise conv
        x = __depthwise_block(x, kernel=kernel, strides=strides, is_use_se=is_use_se, activation=activation, num_layers=num_layers)

        # ** pointwise conv
        x = Conv2D(out_dim, (1, 1), strides=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)

        if shortcut and strides == (1, 1):
            in_dim = K.int_shape(_inputs)[-1]
            if in_dim != out_dim:
                ins = Conv2D(out_dim, (1, 1), strides=(1, 1), padding='same')(_inputs)
                x = Add()([x, ins])
            else:
                x = Add()([x, _inputs])
    return x


def VoiceNet(input_shape = (40,80,1), classes=40):
    """
    Implementation of Genetic VoiceNet

    Arguments:
        input_shape -- shape of the images of the dataset
        classes -- integer, number of classes

    Returns:
        model -- a Model() instance in Keras
    """

    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 0
    X = Conv2D(64, (7, 7), strides=(2, 2), kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)

    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 1: code -> 0-10
    Y_in = convolutional_block(X, f=3, filters=[64, 64, 256], stage=1, block='a', s=1)
    Y_I = identity_block(Y_in, 3, [64, 64, 256], stage=1, block='1_I')
    Y_III = identity_block(Y_I, 3, [64, 64, 256], stage=1, block='1_III')
    Y_out = convolutional_block(Y_III, f=3, filters=[64, 64, 256], stage=1, block='a', s=1)

    # Stage 2: code -> 1-01-001
    X = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(Y_out)

    Z_in = X
    Z_I = __bottleneck_block(Z_in, out_dim=256, kernel=(3,3), strides=(1,1), expansion_dim=256, is_use_bias=False, shortcut=True, is_use_se=True, activation='HS', num_layers=1)
    Z_II = __bottleneck_block(Z_I, out_dim=256, kernel=(3,3), strides=(1,1), expansion_dim=256, is_use_bias=False, shortcut=True, is_use_se=True, activation='HS', num_layers=2)
    Z_III = __bottleneck_block(Z_II, out_dim=256, kernel=(3,3), strides=(1,1), expansion_dim=256, is_use_bias=False, shortcut=True, is_use_se=True, activation='HS', num_layers=3)
    Z_IV = __bottleneck_block(Z_III, out_dim=256, kernel=(3,3), strides=(1,1), expansion_dim=256, is_use_bias=False, shortcut=True, is_use_se=True, activation='HS', num_layers=4)
    Z_out = Z_IV

    X = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(Z_out)

    # Stage 3: code -> 0-00
    M_in = convolutional_block(X, f = 3, filters = [256, 256, 512], stage = 3, block='c', s = 1)
    M_out = convolutional_block(M_in, f = 3, filters = [256, 256, 512], stage = 3, block='c', s = 1)

    # AVGPOOL
    X = AveragePooling2D((2,2), name="avg_pool")(M_out)

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    # Create model
    model = Model(inputs = X_input, outputs = X)

    return model

"""II. MAIN"""

def main():
    #load model
    model = VoiceNet(input_shape = (40,80,1), classes=40)
    #summary
    model.summary()

if __name__=='__main__':
    main()