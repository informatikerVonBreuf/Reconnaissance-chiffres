import tensorflow as tf
import os

import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Utilisation des données MNIST directement depuis TensorFlow
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalisation des données entre 0 et 1

# Création du modèle
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compilation du modèle
model.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Entraînement du modèle
history = model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Évaluation du modèle sur l'ensemble de test
test_loss, test_acc = model.evaluate(x_test, y_test)
print('Précision Test:', test_acc)

# Affichage de l'évolution de la précision et de la perte au cours de l'entraînement
plt.plot(history.history['accuracy'], label='Précision Entraînement')
plt.plot(history.history['val_accuracy'], label='Précision Validation')
plt.xlabel('Époque')
plt.ylabel('Précision')
plt.legend()
plt.show()

print('Précision Test:', sess.run(accuracy,feed_dict=feed_dict(False)))
