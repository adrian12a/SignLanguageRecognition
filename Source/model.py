import tensorflow as tf
import numpy as np


class Model:
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        'dataset\\train',
        validation_split=0.2,
        subset="training",
        seed=1234,
        image_size=(256, 256)
    )

    valid_dataset = tf.keras.utils.image_dataset_from_directory(
        'dataset\\train',
        validation_split=0.2,
        subset="validation",
        seed=1234,
        image_size=(256, 256)
    )

    test_dataset = tf.keras.utils.image_dataset_from_directory(
        'dataset\\test2',
        image_size=(256, 256)
    )

    class_names = train_dataset.class_names
    epochs = 10
    history = None

    # model sieci neuronowej
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(16, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(),

        tf.keras.layers.Flatten(input_shape=(256, 256, 3)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(len(class_names))
    ])

    def __init__(self):
        # normalizacja wartosci pikseli z zakresu 0..255 do 0..1
        norm = tf.keras.layers.Rescaling(1. / 255)
        self.train_dataset = self.train_dataset.map(lambda x, y: (norm(x), y))
        self.valid_dataset = self.valid_dataset.map(lambda x, y: (norm(x), y))
        self.test_dataset = self.test_dataset.map(lambda x, y: (norm(x), y))

    # uczenie sieci
    def training(self):
        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

        self.history = self.model.fit(self.train_dataset, validation_data=self.valid_dataset, epochs=self.epochs)

    # zapis wyuczonej sieci do pliku
    def save_model(self, model_filename, history_filename):
        self.model.save(model_filename)
        np.save(history_filename, self.history.history)

    # testowanie sieci
    def testing(self):
        test_loss, test_acc = self.model.evaluate(self.test_dataset, verbose=2)
        print('\nTest accuracy:', test_acc)

    # odczyt wyuczonej sieci z pliku
    def load_model(self, filename):
        try:
            self.model = tf.keras.models.load_model(filename)
            return self.model
        except:
            print("Error loading model from file")

    # odczyt statystyk z uczenia sieci z pliku
    def load_history(self, filename):
        try:
            self.history = np.load(filename, allow_pickle='TRUE').item()
            return self.history
        except:
            print("Error loading history from file")
