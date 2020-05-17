from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics 
from keras.datasets import imdb

import numpy as np

#CRIANDO OS MODELOS
modelos = models.Sequential()
modelos.add(layers.Dense(16, activation='relu', input_shape=(10000, )))
modelos.add(layers.Dense(16, activation='relu'))
modelos.add(layers.Dense(1, activation='sigmoid'))
modelos.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
modelos.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# PEGANDO OS DADOS, LABEL, TESTES E LABEL DOS TESTES
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

# VETORIZANDO OS DADOS
def vetorizar_sequencia(sequencias, dimensoes=10000):
    result = np.zeros((len(sequencias), dimensoes))
    for i, sequencia in enumerate(sequencias):
        result[i, sequencia] = 1.
    return result
x_train = vetorizar_sequencia(train_data)
y_train = vetorizar_sequencia(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

#REFIRANDO UMA AMOSTRA PARA TESTE
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

#TRAINAMENTO COM OS DADOS E AS 10000 AMOSTRAS
modelos.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

historico = modelos.fit(partial_x_train, partial_y_train, epochs=20, batch_size=512, validation_data=(x_val, y_val))
historico_dict = historico.history
print(historico_dict.keys())
from matplotlib import pyplot as plt
acc = historico.history['acc']
val_acc = historico.history['val_acc']
loss = historico.history['loss']
val_loss = historico.history['val_loss']
epochs = range(1, len(acc) + 1)
# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

#PRECISÃO DO TREIANMENTO
modelos.evaluate(x_train, y_train)

import ipdb; ipdb.set_trace()
