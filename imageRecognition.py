# Importa le librerie necessarie
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# Carica e prepara il dataset CIFAR-10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Normalizza le immagini e aggiungi una dimensione per il canale (scala tra 0 e 1)
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# Converte le etichette in forma one-hot encoding
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Costruisci il modello CNN
model = models.Sequential()

# Aggiungi uno strato convoluzionale con attivazione ReLU e dimensione del filtro 3x3
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))

# Aggiungi uno strato di pooling per ridurre le dimensioni dell'immagine
model.add(layers.MaxPooling2D((2, 2)))

# Aggiungi un altro strato convoluzionale e di pooling
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Aggiungi un altro strato convoluzionale
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# Appiattisci i dati e aggiungi uno strato densamente connesso con attivazione ReLU
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))

# Strato di output con attivazione softmax per la classificazione
model.add(layers.Dense(10, activation='softmax'))

# Compila il modello specificando ottimizzatore, funzione di perdita e metriche
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Addestra il modello con il set di addestramento
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.2)

# Valuta il modello sul set di test
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f'Test accuracy: {test_acc}')
