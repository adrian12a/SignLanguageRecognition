import os
import tkinter as tk
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import graphs as gr

class Gui:

    root = tk.Tk()

    history = None
    model = None
    data = None

    # wybor zdjecia z pliku
    def choosePicture(self):
        self.clearFrame()
        path = filedialog.askopenfilename(initialdir=".", title="Wybierz zdjęcie",
                                          filetypes=(("Obrazy", "*.jpg"), ("Wszystkie pliki", "*.*")))
        if path:
            print("Wybrano plik:", path)
            figure = gr.predict(self.model, path, self.data.class_names)
            canvas = FigureCanvasTkAgg(figure, self.root)
            canvas.get_tk_widget().place(height=300, width=1240, x=20, y=200)

    # wybor modelu z pliku
    def chooseModel(self):
        self.clearFrame()
        path = filedialog.askdirectory(initialdir=".", title="Wybierz folder")
        if path:
            print("Wybrano folder:", path)
            self.setModel(path)

    # wyswietlenie statystyk uczenia wybranego modelu (grafy)
    def showStats(self):
        self.clearFrame()
        figure = gr.accuracyAndLoss(self.history)
        canvas = FigureCanvasTkAgg(figure, self.root)
        canvas.get_tk_widget().pack()
        canvas.get_tk_widget().place(height=400, width=1240, x=20, y=200)

    def __init__(self, data):
        self.root.title('System rozpoznawania języka migowego na obrazach')
        self.root.geometry('1280x720')
        self.root.resizable(False, False)
        self.root.configure(bg='#fef9e7')
        self.data = data
        self.setModel('4layers10epochs')
        self.clearFrame()
        self.root.mainloop()

    # ustawienie wybranego modelu sieci
    def setModel(self, folder):
        self.model = self.data.load_model(os.path.join(folder, 'model.h5'))
        self.history = self.data.load_history(os.path.join(folder, 'history.npy'))

    # wyswietlenie opisu modelu
    def showModelDescription(self):
        summary = []
        self.model.summary(print_fn=lambda x: summary.append(x))
        summary = "\n".join(summary)
        label = tk.Label(self.root, text=summary)
        label.pack()
        label.place(height=600, width=1240, x=20, y=125)
        label.configure(bg='#fef9e7')

    # narysowanie ponownie glownego ekranu
    def clearFrame(self):
        for widget in self.root.winfo_children():
            widget.destroy()

        label = tk.Label(self.root, text="System rozpoznawania języka migowego na obrazach")
        label.configure(bg='#fef9e7')
        choosePictureButton = tk.Button(self.root, text="Wybierz zdjęcie", command=self.choosePicture)
        chooseModelButton = tk.Button(self.root, text="Wybierz inny model sieci neuronowej", command=self.chooseModel)
        showStatsButton = tk.Button(self.root, text="Wyświetl statystyki uczenia", command=self.showStats)
        modelDescriptionButton = tk.Button(self.root, text="Wyświetl opis sieci", command=self.showModelDescription)

        label.pack()
        choosePictureButton.pack()
        chooseModelButton.pack()
        showStatsButton.pack()
        modelDescriptionButton.pack()

        label.place(height=30, width=1000, x=150, y=20)
        choosePictureButton.place(height=30, width=250, x=60, y=75)
        chooseModelButton.place(height=30, width=250, x=360, y=75)
        showStatsButton.place(height=30, width=250, x=660, y=75)
        modelDescriptionButton.place(height=30, width=250, x=960, y=75)
