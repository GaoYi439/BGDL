from keras import backend as K
def drop_loss(gamma=2, margin=0.8):
    def drop_loss_fixed(y_true, y_pred):
        eps = 1e-12
        theta = lambda t: (K.sign(t) + 1.) / 2.  # the theta function
        y_pred = K.clip(y_pred, eps, 1.-eps)  # improve the stability of the loss and see issues 1 for more information
        y_pred = K.max(y_true*y_pred, axis=-1)
        return -K.sum(theta(margin - y_pred)*K.pow(1-y_pred, gamma)*K.log(y_pred))
    return drop_loss_fixed