from keras.layers import Layer
from keras import backend as K
from keras.constraints import NonNeg,UnitNorm
from keras.initializers import Constant
import numpy as np


class PrototypeLayer(Layer):
    """
    Prototype_distances keras layer. Calculates the distance of the input to each
    prototype. The prototypes are stores as a weight of the layer.
    """
    def __init__(self, n_prototypes, distance= 'euclidean', use_weighted_sum=True, **kwargs):
        self.n_prototypes = n_prototypes
        self.distance = distance        
        self.use_weighted_sum = use_weighted_sum
        super(PrototypeLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.times = input_shape[1]
        self.kernel = self.add_weight(name='prototype_feature_vectors', 
                                      shape=(self.n_prototypes, input_shape[1],input_shape[2],input_shape[3]),
                                      initializer='uniform',
                                      trainable=True)
        super(PrototypeLayer, self).build(input_shape)

    def call(self, x):     
        x = K.expand_dims(x,axis=1)
        if not self.use_weighted_sum:
            axis = (2,3,4)
        else: 
            axis = (2,4)
        if self.distance == 'euclidean':
            distance = K.sum(K.pow((x - self.kernel),2), axis=axis)
        if self.distance == 'euclidean_patches':
            distance = K.sum(K.pow((x - self.kernel),2), axis=-1)

        if self.distance == 'cosine':
            x = K.l2_normalize(x, axis=axis)
            kernel = K.expand_dims(self.kernel,axis=0)
            kernel = K.l2_normalize(kernel, axis=axis)
            distance = - K.sum(x * kernel, axis=axis)    
          
        return distance
    
    def get_config(self):
        config = {'n_prototypes': self.n_prototypes,
                  'distance': self.distance,
                  'use_weighted_sum': self.use_weighted_sum}
        base_config = super(PrototypeLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))    
    
    def compute_output_shape(self, input_shape):
        
        if self.distance == 'euclidean_patches':
            return (input_shape[0], self.n_prototypes,input_shape[1],input_shape[2])

        if not self.use_weighted_sum:
            return (input_shape[0], self.n_prototypes)
        else: 
            return (input_shape[0], self.n_prototypes,input_shape[2])



class WeightedSum(Layer):
    """
    Mean keras layer. Calculates the weighted mean of the input.
    The weights of the mean are store as a keras weight.
    """
    def __init__(self, D3=False, **kwargs):
        self.D3 = D3
        super(WeightedSum, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        #print(input_shape)
        if self.D3:
            shape = (input_shape[1],input_shape[2],input_shape[3])
        else:
            shape = (input_shape[1],input_shape[2])

        self.kernel = self.add_weight(name='prototype_feature_vectors', 
                                      shape=shape,
                                      initializer=Constant(value=1/float(input_shape[2])),constraint=UnitNorm(axis=0),#,
                                      trainable=True)#,regularizer=l2(0.01))#NonNeg()

        super(WeightedSum, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        kernel = K.expand_dims(self.kernel,axis=0)
        return K.sum(x*self.kernel,axis=-1)
        
    def get_config(self):
        config = {'D3': self.D3}
        base_config = super(WeightedSum, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))        
    
    def compute_output_shape(self, input_shape):
        if self.D3:
            shape = (input_shape[0],input_shape[1],input_shape[2])
        else:
            shape = (input_shape[0],input_shape[1])
        return shape