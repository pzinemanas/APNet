from keras import backend as K
from tensorflow.nn import top_k


def prototype_loss(y_true, y_pred):
    print('loss',y_pred.shape)
    if len(y_pred.shape) == 3:
        y_pred = K.mean(y_pred,axis=-1)
    if len(y_pred.shape) == 4:
        y_pred = K.mean(y_pred,axis=(2,3))
    print('loss',y_pred.shape)

    error_1 = K.mean(K.min(y_pred, axis = 0))
    error_2 = K.mean(K.min(y_pred, axis = 1))     

    return 1.0*error_1 + 1.0*error_2