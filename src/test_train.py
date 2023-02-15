# Laboratorio 3
# Inteligencia Artificial
# Authors: YongBum Park 20117, Santiago Taracena 20017, Pedro Arriola 20188, Oscar López 20679

import random
import re

SEED = 123

# Definición de la clase TestTrain.
class TestTrain(object):

    # Método constructor de la clase TestTrain.
    def __init__(self, path):

        # Declaración de una seed y lectura del archivo.
        random.seed(SEED)
        self.path = path
        self.read()

    # Método para leer el archivo a limpiar.
    def read(self):

        # Leer el archivo de entrenamiento.
        with open(self.path, "r") as f:
            self.lines = f.readlines()

        # Limpieza previa de los mensajes.
        self.clear_message()

    # Método para limpiar los mensajes.
    def clear_message(self):

        # Lista con mensajes limpios.
        self.messages = []

        # Limpiar y separar los mensajes
        for line in self.lines:
            self.label, self.text = line.strip().split("\t")
            # Limpiar el texto eliminando caracteres especiales y convirtiéndolo a minúsculas.
            self.text = re.sub(r"[^\w\s]", "", self.text.lower())
            if (len(self.text) > 2):
                self.messages.append((self.label, self.text))

        # Mezcla de los mensajes.
        self.split_messages()

    # Método para seleccionar aleatoriamente los mensajes.
    def __random_selection(self, messages, size, result):
        for _ in range(size):
            element = random.choice(messages)
            result.append(element)
            messages.remove(element)
        return messages, result

    # Método para separar los mensajes en conjuntos de entrenamiento y prueba.
    def split_messages(self):

        # Tamaño de muestras y declaración de conjuntos de mensajes.
        train_size = int(len(self.messages) * 0.8)
        validation_size = int(len(self.messages) * 0.1)
        test_size = int(len(self.messages) * 0.1)
        self.train_messages = []
        self.validation_messages = []
        self.test_messages = []

        # Conjunto de entrenamiento.
        self.messages, self.train_messages = self.__random_selection(self.messages, train_size, self.train_messages)
        self.messages, self.validation_messages = self.__random_selection(self.messages, validation_size, self.validation_messages)
        self.messages, self.test_messages = self.__random_selection(self.messages, test_size, self.test_messages)

        # Almacenamiento de conjuntos en archivos separados.
        self.save()

    # Método para guardar los conjuntos de entrenamiento, prueba y validación en archivos separados.
    def save(self):
        # Guardar los conjuntos de entrenamiento, prueba y validación en archivos separados
        with open("./data/train.txt", "w") as f:
            for label, text in self.train_messages:
                f.write(f"{label}\t{text}\n")
        with open("./data/test.txt", "w") as f:
            for label, text in self.test_messages:
                f.write(f"{label}\t{text}\n")
        with open("./data/validation.txt", "w") as f:
            for label, text in self.validation_messages:
                f.write(f"{label}\t{text}\n")
