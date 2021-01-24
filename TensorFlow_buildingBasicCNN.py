import tensorflow as tf
print(tf.__version__)

# intitializing callback class that cancelling trainig when reaching 99.8% accuracy at the end of each epoch
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end (self, epoch, logs={}):
        if (logs.get('acc') > 0.998):
            print("\nReached 99.8% accracy, cancelling training.")
            self.model.stop_training = True

# builindg CNN method
def train_mnist_conv():

    # loading fashion_mnist data(60,000 + 10,000 / 28 x 28 images with monochrom color)
    mnist = tf.keras.datasets.fashion_mnist
    (training_images, training_labels), (test_images, test_labels) = mnist.load_data()
    # reshaping (n, 28, 28) into (n, 28, 28, 1) for later CNN input
    training_images=training_images.reshape(60000, 28, 28, 1)
    test_images = test_images.reshape(10000, 28, 28, 1)
    # normalizing pixel value between 0.0 ~ 1.0
    training_images=training_images / 255.0
    test_images=test_images/255.0

    # callback variance from MyCallback class
    callbacks = MyCallback()

    # building a sequential cnn model 
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
    # max epochs 20 iterations + callback options when accuracy reaches 99.8%
    history = model.fit(training_images, training_labels, epochs=20, callbacks=[callbacks])
  
    return  summary, history.epoch, history.history['acc'][-1]

_, _, _ = train_mnist_conv()
