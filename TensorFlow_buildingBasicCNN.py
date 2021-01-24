import tensorflow as tf
print(tf.__version__)

class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end (self, epoch, logs={}):
        if (logs.get('acc') > 0.998):
            print("\nReached 99.8% accracy, cancelling training.")
            self.model.stop_training = True

def train_mnist_conv():

  mnist = tf.keras.datasets.fashion_mnist
  (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
  training_images=training_images.reshape(60000, 28, 28, 1)
  training_images=training_images / 255.0
  test_images = test_images.reshape(10000, 28, 28, 1)
  test_images=test_images/255.0

  callbacks = MyCallback()

  model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
  ])

  model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
  summary = model.summary()
  history = model.fit(training_images, training_labels, epochs=10, callbacks=[callbacks])
  
  return  summary, history.epoch, history.history['acc'][-1]

  test_loss = model.evaluate(test_images, test_labels)
