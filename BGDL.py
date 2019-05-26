from __future__ import print_function
from __future__ import absolute_import
from keras.optimizers import RMSprop
from keras import backend
from keras.layers import Dense, Input, Lambda, TimeDistributed, GRU
from keras.layers.merge import concatenate
from keras.layers.embeddings import Embedding
from keras.models import Model
import numpy as np
import pickle
from keras.preprocessing import sequence
from keras.utils import np_utils
from loss import drop_loss

np.random.seed(1337)

#set parmeters
maxlen = 300
embedding_dims = 300  # the length of word vector
batch_size = 32
epochs = 10
hidden_dim_1 = 50
hidden_dim_2 = 100

def train(x_train, y_train, x_test, y_test, batch_size, epochs, hidden_dim_1, hidden_dim_2):
    print('Build model...')
    document = Input(shape=(None,), dtype="int32")
    left_context = Input(shape=(None,), dtype="int32")
    right_context = Input(shape=(None,), dtype="int32")
    embedder = Embedding(output_dim=300, input_dim=n_symbols, weights=[embedding_weights],
                         input_length=maxlen)

    doc_embedding = embedder(document)
    l_embedding_f = embedder(left_context)
    r_embedding_f = embedder(right_context)

    l_embedding_b = Lambda(lambda x: backend.reverse(x, axes=1))(l_embedding_f)
    r_embedding_b = Lambda(lambda x: backend.reverse(x, axes=1))(r_embedding_f)


    # the left context vector, see equation (1) and (2).
    forward_l = GRU(hidden_dim_1, return_sequences=True)(l_embedding_f)
    backward_l = GRU(hidden_dim_1, return_sequences=True)(l_embedding_b)

    # the right context vector, see equation (4) and (5).
    forward_r = GRU(hidden_dim_1, return_sequences=True)(r_embedding_f)
    backward_r = GRU(hidden_dim_1, return_sequences=True)(r_embedding_b)

    together = concatenate([forward_l, backward_l, doc_embedding, forward_r, backward_r], axis=2)  # See equation (7).
    semantic = TimeDistributed(Dense(hidden_dim_2, activation="relu"))(together)  # See equation (8).
    pool_rnn = Lambda(lambda x: backend.max(x, axis=1), output_shape=(hidden_dim_2,))(semantic)  # See equation (9).

    output = Dense(10, input_dim=hidden_dim_2, activation="softmax")(pool_rnn)  # See equations (10) and (11).
    model = Model(inputs=[document, left_context, right_context], outputs=output)
    rmsprop = RMSprop(lr=0.001, decay=1e-6, rho=0.9)

    model.compile(optimizer=rmsprop, loss=drop_loss(1, 0.9), metrics=['accuracy'])

    #Build left and right data
    doc_x_train = np.array(x_train)
    left_x_train = np.array(x_train)
    right_x_train = np.array(x_train)

    doc_x_test = np.array(x_test)
    left_x_test = np.array(x_test)
    right_x_test = np.array(x_test)

    print('Train...')
    model.fit([doc_x_train, left_x_train, right_x_train], y_train, batch_size=batch_size,
              epochs=epochs, validation_data=[[doc_x_test, left_x_test, right_x_test], y_test], verbose=2)

