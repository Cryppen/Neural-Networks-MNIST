import tensorflow as tf
from tensorflow import keras

class mycallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('acc') > 0.90:
            print("\n90% accuracy reached")
            self.model.stop_training = True


callback = mycallback()
fashion_mnist = keras.datasets.fashion_mnist

(train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()

train_data, test_data = train_data.reshape(60000,28,28,1), test_data.reshape(10000,28,28,1)
train_data, test_data = train_data / 255.0, test_data / 255.0

model = keras.models.Sequential([keras.layers.Conv2D(16,(3,3), activation='relu',input_shape=(28,28,1)),
                                 keras.layers.MaxPool2D(2,2),
                                 keras.layers.Conv2D(64,(3,3),activation='relu'),
                                 keras.layers.MaxPool2D(2,2),
                                 keras.layers.Flatten(),
                                 keras.layers.Dense(512,activation=tf.nn.relu),
                                 keras.layers.Dense(10,activation=tf.nn.softmax)])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(train_data,train_labels,epochs=5,callbacks=[callback])

print(model.summary())