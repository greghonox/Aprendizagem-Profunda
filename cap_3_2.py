from cap_3_1 import historico
from keras.datasets import reuters
from keras.utils import to_categorical

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
print(len(train_data), len(test_data))
print(train_data[10], train_data.shape)

word_index = reuters.get_word_index()
reverse_word_index_data = dict((valor, data) for valor, data in word_index.items())
print(reverse_word_index_data)

import numpy as np

def vetorizar_dados(sequencias, dimensao=10000):
    res = np.zeros((len(sequencias), dimensao))
    for i, sequencia in enumerate(sequencias):
        res[i, sequencia] = 1.
    return res

x_train = vetorizar_dados(train_data)
y_train = vetorizar_dados(test_data)

def um_quente(rotulos, dimensao=46):
    res = np.zeros((len(rotulos), dimensao))
    for i, rotulo in enumerate(rotulos):
        res[i, rotulo] = 1.
    return res

#um_quente_train_labels = um_quente(test_lebals)
#um_quente_test_labels = um_quente(test_data)

um_quente_train_labels = to_categorical(train_labels) # É O MESMO QUE CHAMAR A FUNCAO VETORIZAR(um_quente)
um_quente_test_labels = to_categorical(test_labels) # É O MESMO QUE CHAMAR A FUNCAO VETORIZAR(um_quente)

from keras import models
from keras import layers

modelos = models.Sequential()
modelos.add(layers.Dense(64, activation='relu', input_shape=(10000, )))
modelos.add(layers.Dense(64, activation='relu'))
modelos.add(layers.Dense(46, activation='softmax'))

modelos.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

x_val = x_train[:10000]
partial_x_train = y_train[10000:]

y_val = y_train[:100000]
partial_y_train = y_train[10000:]

historico = modelos.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_train, y_train))
results = modelos.evaluate(x_test, um_quente_test_labels)