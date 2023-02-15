from test_train import *
from laplace import LaplaceClassifier

entrenamiento = TestTrain("entrenamiento.txt")

# Crear una instancia del clasificador de spam con Laplace smoothing
classifier = LaplaceClassifier()

# Entrenar el modelo con los datos de entrenamiento
classifier.train("train.txt")

results = {}

with open("test.txt", "r") as f:
    test_messages = f.readlines()

for message in test_messages:
    label, text = message.strip().split("\t")
    result = classifier.classify(text)
    results[text] = (label, result[0])

print(results)

# Función para contar los valores correctos/incorrectos
def cI_values(tuple):
    return len([1 for text in results if results[text] == tuple])


# Definición de variables
spam_correct = cI_values(("spam", "spam"))
spam_incorrect = cI_values(("spam", "ham"))
ham_correct = cI_values(("ham", "ham"))
ham_incorrect = cI_values(("ham", "spam"))


print([spam_correct, spam_incorrect])
print([ham_incorrect, ham_correct])

print(len(results))
print(sum([spam_correct, spam_incorrect]) + sum([ham_incorrect, ham_correct]))
