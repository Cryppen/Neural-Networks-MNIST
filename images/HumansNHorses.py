import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
import PIL

image_generated_data = ImageDataGenerator(rescale=1. / 255)
test_generated_data = ImageDataGenerator(rescale=1. / 255)

train_generator = image_generated_data.flow_from_directory(
                        r"images/train_dir",
                        target_size=(300,300),
                        batch_size=128,
                        class_mode='binary'
)



validation_generator = test_generated_data.flow_from_directory(
                        r"images/validation_dir",
                        target_size=(300,300),
                        batch_size=128,
                        class_mode='binary'
)

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('acc') > 0.9:
            print("\nReached 90% accuracy cancelling training")
            self.model.stop_training = True

callback = myCallback()

model = keras.models.Sequential([keras.layers.Conv2D(16,(3,3),activation='relu', input_shape=(300,300,3)),
                                 keras.layers.MaxPool2D(2,2),
                                 keras.layers.Conv2D(32,(3,3),activation='relu'),
                                 keras.layers.MaxPool2D(2,2),
                                 keras.layers.Conv2D(64,(3,3),activation='relu'),
                                 keras.layers.MaxPool2D(2, 2),
                                 keras.layers.Conv2D(64, (3, 3), activation='relu'),
                                 keras.layers.MaxPool2D(2, 2),
                                 keras.layers.Conv2D(64, (3, 3), activation='relu'),
                                 keras.layers.MaxPool2D(2,2),
                                 keras.layers.Flatten(),
                                 keras.layers.Dense(512,activation='relu'),
                                 keras.layers.Dense(1,activation='sigmoid')
                                 ])

model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['accuracy'])

fit_model = model.fit(train_generator,steps_per_epoch=8,epochs=20,verbose=1)

print(model.summary())
