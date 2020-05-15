import numpy as np

def naive_relu(x):
    assert len(x.shape) == 2
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i,j] = max(x[i,j], 0)
    return x

def naive_add(x, y):
    assert len(x.shape) == 2
    assert x.shape == y.shape
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[i, j]
    return x

def naive_vector_dot(x, y):
    assert len(x.shape) == 1
    assert len(y.shape) == 1
    assert x.shape[0] == y.shape[0]

    z = 0.
    for i in range(x.shape[0]):
        z += x[i] * y[i]
    return z

def naive_matrix_vector_dot(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 1
    assert x.shape[1] == y.shape[0]

    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            z[i] += x[i, j] * y[j]
    return z
	
#print(naive_relu(np.array([range(1)]*2)))
x = np.array(range(10))
#print(np.maximum(x, 0.))
x = np.random.random((64, 3, 32, 10))
y = np.random.random((32,10))

    
print(np.maximum(x, y))

print('RANDONS:')
r1 = np.random.random([10])
r2 = np.random.random([10])
print(r1, r2)
print(np.dot(r1, r2))
print(naive_vector_dot(r1, r2))
print(naive_vector_dot(np.array(range(10)), np.array(range(10))))

#FAZER ALTERAÇÕES NA MATRIZ CARREGADA
remodelar = np.array([[0., 1.], [2., 3.], [4., 5.]])
print(remodelar, remodelar.shape)
print(remodelar.reshape((6, 1)))
print(remodelar.reshape((1, 6)))

#Matriz Transpostas
print("matris transpostas")
matriz_transposta = np.zeros((300, 20))
print(matriz_transposta, matriz_transposta.shape)
print(np.transpose(matriz_transposta), np.transpose(matriz_transposta).shape)