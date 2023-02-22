import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# Wykres dokładności
def accuracy(history):
    acc = [round(i*100, 1) for i in history['accuracy']]
    val_acc = [round(i*100, 1) for i in history['val_accuracy']]
    epochs = []
    for i in range(len(acc)):
        epochs.append(i + 1)

    if len(epochs) > 1:
        plt.plot(epochs, acc, label='Training Accuracy')
        plt.plot(epochs, val_acc, label='Validation Accuracy')
    else:
        plt.scatter(epochs, acc, label='Training Accuracy')
        plt.scatter(epochs, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    plt.locator_params(axis="x", integer=True, tight=True)
    plt.xlabel("Number of epochs")
    plt.ylabel("Accuracy percentage")
    plt.show()


# Wykres strat
def loss(history):
    loss = [round(i*100, 1) for i in history['loss']]
    val_loss = [round(i*100, 1) for i in history['val_loss']]
    epochs = []
    for i in range(len(loss)):
        epochs.append(i + 1)

    if len(epochs) > 1:
        plt.plot(epochs, loss, label='Training Loss')
        plt.plot(epochs, val_loss, label='Validation Loss')
    else:
        plt.scatter(epochs, loss, label='Training Loss')
        plt.scatter(epochs, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.locator_params(axis="x", integer=True, tight=True)
    plt.xlabel("Number of epochs")
    plt.ylabel("Loss percentage")
    plt.show()


# Odpowiedz sieci na podany obraz
def predict(model, image_path, class_names, correct_answer):
    image = tf.keras.utils.load_img(image_path, target_size=(256, 256))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    for i in range(0, len(class_names)):
        class_names[i] = class_names[i][:3]

    plt.figure(figsize=(21, 3))
    plt.subplots_adjust(wspace=0)
    plt.subplot(1, 2, 1)
    if class_names[np.argmax(score)] == correct_answer:
        plt.title('Answer: {} with a {:.2f}% confidence.'.format(class_names[np.argmax(score)], 100 * np.max(score)), color='g')
    else:
        plt.title('Answer: {} with a {:.2f}% confidence.'.format(class_names[np.argmax(score)], 100 * np.max(score)), color='r')
    plt.bar(class_names, score)
    plt.subplot(1, 2, 2)
    plt.title('Image: {}'.format(correct_answer))
    plt.imshow(image)
    plt.show()

# Odpowiedz sieci na podany obraz (wesja dla GUI)
def predict(model, image_path, class_names):
    image = tf.keras.utils.load_img(image_path, target_size=(256, 256))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    for i in range(0, len(class_names)):
        class_names[i] = class_names[i][:3]

    figure = plt.figure(figsize=(21, 3))
    plt.subplots_adjust(wspace=0)
    plt.subplot(1, 2, 1)
    plt.title('Answer: {} with a {:.2f}% confidence.'.format(class_names[np.argmax(score)], 100 * np.max(score)))

    plt.bar(class_names, score)
    plt.subplot(1, 2, 2)
    plt.imshow(image)

    return figure

# wykresy dokładności i strat (wersja dla GUI)
def accuracyAndLoss(history):
    acc = [round(i*100, 1) for i in history['accuracy']]
    val_acc = [round(i*100, 1) for i in history['val_accuracy']]
    loss = [round(i*100, 1) for i in history['loss']]
    val_loss = [round(i*100, 1) for i in history['val_loss']]
    epochs = []
    for i in range(len(acc)):
        epochs.append(i + 1)

    figure = plt.figure(figsize=(10, 4))
    if len(epochs) > 1:
        plt.subplot(1, 2, 1)
        plt.plot(epochs, acc, label='Training Accuracy')
        plt.plot(epochs, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')
        plt.locator_params(axis="x", integer=True, tight=True)
        plt.xlabel("Number of epochs")
        plt.ylabel("Accuracy percentage")
        plt.subplot(1, 2, 2)
        plt.plot(epochs, loss, label='Training Loss')
        plt.plot(epochs, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.locator_params(axis="x", integer=True, tight=True)
        plt.xlabel("Number of epochs")
        plt.ylabel("Loss percentage")
    else:
        plt.subplot(1, 2, 1)
        plt.scatter(epochs, acc, label='Training Accuracy')
        plt.scatter(epochs, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')
        plt.locator_params(axis="x", integer=True, tight=True)
        plt.xlabel("Number of epochs")
        plt.ylabel("Accuracy percentage")
        plt.subplot(1, 2, 2)
        plt.scatter(epochs, loss, label='Training Loss')
        plt.scatter(epochs, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.locator_params(axis="x", integer=True, tight=True)
        plt.xlabel("Number of epochs")
        plt.ylabel("Loss percentage")

    return figure
