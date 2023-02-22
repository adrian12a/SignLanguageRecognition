import string

import model as ml
import graphs as gr
import gui
import tensorflow as tf

data = ml.Model()

# uczenie sieci
# data.training()
# data.save_model('model.h5', 'history.npy')
# data.testing()

# wczytanie wyuczonej sieci z pliku
model = data.load_model('4layers10epochs\\model.h5')
history = data.load_history('4layers10epochs\\history.npy')
data.testing()

# struktura sieci
# model.summary()

# wykresy uczenia sieci
# gr.accuracy(history)
# gr.loss(history)

# wykresy odpowiedzi sieci dla zdjec testowych
# letters = list(string.ascii_uppercase)
# for i in range(0, len(letters)):
#     path = 'dataset\\test1\\' + letters[i] + '\\' + letters[i] + '_test.jpg'
#     gr.predict(model, path, data.class_names, letters[i])

# uruchomienie graficznego interfejsu uzytkownika
ui = gui.Gui(data)
