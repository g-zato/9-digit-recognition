import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np





#train_images = gzip.open('train-images-idx3-ubyte.gz','r')
#train_labels = gzip.open('train-labels-idx1-ubyte.gz','r')
#test_images = gzip.open('t10k-images-idx3-ubyte.gz','r')
#test_labels = gzip.open('t10k-labels-idx1-ubyte.gz','r')

from mlxtend.data import loadlocal_mnist

X_train, y_train = loadlocal_mnist(images_path = 'train-images-idx3-ubyte', labels_path = 'train-labels-idx1-ubyte')
X_test, y_test = loadlocal_mnist(images_path = 't10k-images-idx3-ubyte', labels_path = 't10k-labels-idx1-ubyte')


X_train = tf.keras.utils.normalize(X_train, axis = 1)
X_test = tf.keras.utils.normalize(X_test, axis = 1)

print(y_train)

#sample = np.reshape(X_train[-1], (28, 28))  #transforma a array unidimensional (784 linhas, 1 coluna) em uma array 28 por 28
#plt.imshow(sample, cmap=plt.cm.binary)
#plt.show()

#Montando o modelo

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Dense(1024, activation = tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation = tf.nn.softmax))

model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

#model.fit(X_train, y_train, epochs=10)


#val_loss, val_acc = model.evaluate(X_test, y_test)
#print(f'Validated Accuracy: {val_acc*100}%')
#print(f'Validated Loss: {val_loss*100}%')

#model.save('MNIST_digit_recognized_vanilla_NN')
#sample = np.reshape(X_train[0], (28, 28))  #transforma a array unidimensional (784 linhas, 1 coluna) em uma array 28 por 28
#plt.imshow(sample, cmap=plt.cm.binary)
#plt.show()
new_model = model.load('MNIST_digit_recognized_vanilla_NN')
new_model.predict(np.argmax(X_test[0]))

