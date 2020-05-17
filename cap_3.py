from keras import layers
from keras import models
from keras import optimizers


#camadas = layers.Dense(32, input_shape= (748,)) # CRIACAÇÃO DE UMA CADA COM 32 SAIDAS 2D

modelos = models.Sequential()
modelos.add(layers.Dense(32, input_shape=(748, ))) # ADICIONANDO NOVAMENTE A MESMA CAMADA ANTERIORMENTE NO MODELO
modelos.add(layers.Dense(32)) # CRIANDO OUTRA CAMADA, AMBAS SEM ENTRADAS

print(modelos)

# DEFININDO MODELO PARA A CAMADA    
modelos.add(layers.Dense(32, activation='relu', input_shape=(748, )))
modelos.add(layers.Dense(10, activation='softmax'))

# ABAIXO DEFINIMOS OS MESMOS MODELOS PELA API   
input_tensor = layers.Input(shape=(784, ))
x = layers.Dense(32, activation='relu') (input_tensor)
out_tensor = layers.Dense(10, activation='softmax')(x)
outros_modelos = models.Model(inputs=input_tensor, outputs=out_tensor)

#ETAPA DE TREINAMENTO
modelos.compile(optimizer=optimizers.RMSprop(lr=0.001), loss='mse', metrics=['accuracy'])
#modelos.fit(input_tensor, out_tensor, batch_size=128, epochs=10) # NÃO FUNCIONOU!

from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
print(train_labels[55])
print(max([max(sequencia) for sequencia in train_data]))

palavras = imdb.get_word_index()
reverse_palavra = dict([(valor, chave) for (valor, chave) in palavras.items()])
print(reverse_palavra)
decode = ' '.join([reverse_palavra.get(i-3, '?') for i in train_data[0]])
print(decode)

import numpy as np
def vetorizar_sequencia(sequencias, dimensoes=10000):
    result = np.zeros((len(sequencias), dimensoes))
    for i, sequencia in enumerate(sequencias):
        result[i, sequencia] = 1.
    return result

x_train = vetorizar_sequencia(train_data)
x_test = vetorizar_sequencia(test_data)
print(x_train, x_test)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

