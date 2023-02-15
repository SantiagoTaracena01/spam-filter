# Laboratorio 3
# Inteligencia Artificial
# Authors: YongBum Park 20117, Santiago Taracena 20017, Pedro Arriola 20188, Oscar López 20679

from test_train import *
from laplace import LaplaceClassifier

entrenamiento = TestTrain("./data/entrenamiento.txt")

# Crear una instancia del clasificador de spam con Laplace smoothing
classifier = LaplaceClassifier()

# Entrenar el modelo con los datos de entrenamiento
classifier.train("./data/train.txt")

results = {}

# Abrir el archivo de la información
with open("./data/test.txt", "r") as f:
    test_messages = f.readlines()

# Iterar sobre los mensajes obtenidos por el archivo
for message in test_messages:
    label, text = message.strip().split("\t")
    result = classifier.classify(text)
    results[text] = (label, result[0])

spam_correct = len([1 for text in results if results[text] == ("spam", "spam")])
spam_incorrect = len([1 for text in results if results[text] == ("spam", "ham")])
ham_correct = len([1 for text in results if results[text] == ("ham", "ham")])
ham_incorrect = len([1 for text in results if results[text] == ("ham", "spam")])

# Realizar la impresión de datos
print([spam_correct, spam_incorrect])
print([ham_incorrect, ham_correct])