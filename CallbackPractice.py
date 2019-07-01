import tensorflow as tf
from tensorflow import keras

#its a good idea to check your accuracy on the epoch end instead of at
# the beginning as it can vary a lot during executing.

class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('acc')>0.6):
      print("\nReached 60% accuracy so cancelling training!")
      self.model.stop_training = True


callback = myCallback()

dataset = keras.datasets.fashion_mnist
(training_data, training_labels), (test_data, test_labels) = dataset.load_data()

training_data, test_images = training_data / 255.0, test_data / 255.0

model = keras.Sequential([keras.layers.Flatten(input_shape=(28, 28)),
                          keras.layers.Dense(512, activation=tf.nn.relu),
                          keras.layers.Dense(10, activation=tf.nn.softmax)
                          ])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_data, training_labels, epochs=5, callbacks=[callback])

test_loss, test_acc = model.evaluate(test_data, test_labels)
print('Test accuracy:', test_acc)
