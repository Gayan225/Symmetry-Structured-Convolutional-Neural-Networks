import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from conv0 import Conv_kenel_init1
from conv import Conv_kenel_init2
from tensorflow.python.keras.optimizers import Adam

# CUSTOM KERAS LAYERS

#Initialize kernel when k=1
def my_initkernel0(shape, dtype=None, partition_info=None):
	s = list(shape)
	W1h = Conv_kenel_init1(s[0], s[2], s[3])
	Wfull = W1h.form_full_kernel1()
	
	return Wfull

#Initialize kernel when k>1
def my_initkernel1(shape, dtype=None, partition_info=None):
	s = list(shape)
	W1h = Conv_kenel_init2(s[0], s[2], s[3])
	Wfull = W1h.form_full_kernel2()
	
	return Wfull

#Self-cartesian 
def SelfCartesian(x):
    newshape = tf.stack([1, 1, tf.keras.backend.shape(x)[1], 1])
    
    x_expanded = tf.expand_dims(x, axis = -2)
    x_tiled = tf.tile(x_expanded, newshape)
    x_transposed = tf.keras.backend.permute_dimensions(x_tiled, (0,2,1,3))
    x_concat = tf.keras.backend.concatenate([x_tiled, x_transposed], axis=-1)
    return x_concat

def SelfCartesianShape(input_shape):
    shape = list(input_shape)
    return [shape[0], shape[1], shape[1], shape[2]*2]


def weighted_binary_cross_entropy(labels, logits,weight):
    weight_f= tf.keras.backend.constant(weight, dtype=tf.float32)
    class_weights = labels*weight_f + (1 - labels)
    unweighted_losses = tf.keras.backend.binary_crossentropy(target=labels, output=logits)
    weighted_losses = unweighted_losses * class_weights
    loss = tf.keras.backend.mean(tf.matrix_band_part(tf.keras.backend.squeeze(weighted_losses, -1), 0, -1))
    return loss


# ARCHITECTURE

def makemodel(LSTMlayers, BN, weight,reg, lr,inputs):
    
    l2reg = tf.keras.regularizers.l2(reg)
    weight_f= tf.keras.backend.constant(weight, dtype=tf.float32)
    #weight = K.constant(weight) use_bias=True,bias_initializer=tf.keras.initializers.Constant(0.0)
    
    def weighted_binary_cross_entropy(labels, logits,weight):
    	
    	weight_f= tf.keras.backend.constant(weight, dtype=tf.float32)
    	class_weights = labels*weight_f + (1 - labels)
    	unweighted_losses = tf.keras.backend.binary_crossentropy(target=labels, output=logits)
    	weighted_losses = unweighted_losses * class_weights
    	loss = tf.keras.backend.mean(tf.matrix_band_part(tf.keras.backend.squeeze(weighted_losses, -1), 0, -1))
    	return loss
   
    if LSTMlayers:
        h1_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(75, return_sequences = True))(inputs)
        if LSTMlayers > 1:
            h1_lstm = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(50, return_sequences = True))(h1_lstm)
        h1_lstmout = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(15))(h1_lstm)
        h1 = tf.keras.layers.Concatenate(axis=-1)([inputs, h1_lstmout])
        h1square = tf.keras.layers.Lambda(SelfCartesian, output_shape = SelfCartesianShape)(h1)
    else:
        h1square = tf.keras.layers.Lambda(SelfCartesian, output_shape = SelfCartesianShape)(inputs)
    h2square_I = tf.keras.layers.Conv2D(filters=40, kernel_size=1, use_bias=True, bias_initializer=tf.keras.initializers.Constant(0.0) , kernel_regularizer = l2reg, kernel_initializer=my_initkernel0, padding='same',name = 'conv2d_0')(h1square)
    if BN:
        h2square_I1 = tf.keras.layers.BatchNormalization(axis=-1)(h2square_I)
        h2square = tf.keras.layers.Activation('relu')(h2square_I1)
    else:
        h2square = tf.keras.layers.Activation('relu')(h2square_I)
            
    h2square_1 = tf.keras.layers.Conv2D(filters=20, kernel_size=13, use_bias=True, bias_initializer=tf.keras.initializers.Constant(0.0) , kernel_regularizer = l2reg, kernel_initializer=my_initkernel1, padding='same',name = 'conv2d_1')(h2square)
    h2square_2 = tf.keras.layers.Conv2D(filters=20, kernel_size=9, use_bias=True,bias_initializer=tf.keras.initializers.Constant(0.0), kernel_regularizer = l2reg, kernel_initializer=my_initkernel1, padding='same',name = 'conv2d_2')(h2square)
    h2square_3 = tf.keras.layers.Conv2D(filters=10, kernel_size=5, use_bias=True,bias_initializer=tf.keras.initializers.Constant(0.0), kernel_regularizer = l2reg, kernel_initializer=my_initkernel1, padding='same',name = 'conv2d_3')(h2square)
    h2square_a = tf.keras.layers.Concatenate(axis=-1)([h2square_1, h2square_2, h2square_3])
    if BN:
        h2square_b = tf.keras.layers.BatchNormalization(axis=-1)(h2square_a)
        h2squarea = tf.keras.layers.Activation('relu')(h2square_b)
    else:
        h2squarea = tf.keras.layers.Activation('relu')(h2square_a)

    h3square_1 = tf.keras.layers.Conv2D(filters=20, kernel_size=9,kernel_initializer=my_initkernel1, use_bias=True, bias_initializer=tf.keras.initializers.Constant(0.0), kernel_regularizer = l2reg,  padding='same',name = 'conv2d_4')(h2squarea)
    h3square_2 = tf.keras.layers.Conv2D(filters=20, kernel_size=5,kernel_initializer=my_initkernel1, use_bias=True, bias_initializer=tf.keras.initializers.Constant(0.0), kernel_regularizer = l2reg, padding='same',name = 'conv2d_5')(h2squarea)
    h3square_a = tf.keras.layers.Concatenate(axis=-1)([h3square_1, h3square_2])
    if BN:
        h3square_b = tf.keras.layers.BatchNormalization(axis=-1)(h3square_a)
        h3square = tf.keras.layers.Activation('relu')(h3square_b)
    else:
        h3square = tf.keras.layers.Activation('relu')(h3square_a)

    h4square_1 = tf.keras.layers.Conv2D(filters=20, kernel_size=5, use_bias=True,bias_initializer=tf.keras.initializers.Constant(0.0), kernel_regularizer = l2reg, kernel_initializer=my_initkernel1,  padding='same',name = 'conv2d_6')(h3square)
    if BN:
        h4square_b = tf.keras.layers.BatchNormalization(axis=-1)(h4square_1)
        h4square = tf.keras.layers.Activation('relu')(h4square_b)
    else:
        h4square = tf.keras.layers.Activation('relu')(h4square_1)    
    sequencesquare_a = tf.keras.layers.Lambda(SelfCartesian, output_shape = SelfCartesianShape)(inputs)
    sequencesquare_1 = tf.keras.layers.Conv2D(filters=20, kernel_size=1, use_bias=True,bias_initializer=tf.keras.initializers.Constant(0.0), kernel_regularizer = l2reg, kernel_initializer=my_initkernel0, padding='same',name = 'conv2d_7')(sequencesquare_a)
    if BN:
        sequencesquare_b = tf.keras.layers.BatchNormalization(axis=-1)(sequencesquare_1)
        sequencesquare = tf.keras.layers.Activation('relu')(sequencesquare_b)
    else:
        sequencesquare = tf.keras.layers.Activation('relu')(sequencesquare_1)
    h5square = tf.keras.layers.Concatenate(axis=-1)([h4square, sequencesquare])

    output_1 = tf.keras.layers.Conv2D(filters=1, kernel_size=3,use_bias=True,bias_initializer=tf.keras.initializers.Constant(0.0), kernel_regularizer = l2reg, kernel_initializer=my_initkernel1, padding='same',name = 'conv2d_8')(h5square)
    if BN:
        output_b = tf.keras.layers.BatchNormalization(axis=-1)(output_1)
        output = tf.keras.layers.Activation('sigmoid')(output_b)
    else:
        output = tf.keras.layers.Activation('sigmoid')(output_1) 

    model = tf.keras.models.Model(inputs = inputs, outputs = output)
    
    return model, output

