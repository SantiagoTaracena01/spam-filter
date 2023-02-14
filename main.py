from test_train import *
from laplace import LaplaceClassifier

entrenamiento = TestTrain("entrenamiento.txt")

# Crear una instancia del clasificador de spam con Laplace smoothing
classifier = LaplaceClassifier()

# Entrenar el modelo con los datos de entrenamiento
classifier.train("train.txt")

sample = "este es un mensaje de prueba"

# Obtener la predicción para una muestra
prediction = classifier.classify(sample)

print("\nTraining")
print(prediction[0])
print("Probabilidad de spam: ", prediction[1])
print("Probabilidad de ham: ", prediction[2])

# Entrenar el modelo con los datos de entrenamiento
classifier.train("test.txt")

# Obtener la predicción para una muestra
prediction = classifier.classify(sample)

print("\nTesting")
print(prediction[0])
print("Probabilidad de spam: ", prediction[1])
print("Probabilidad de ham: ", prediction[2])
