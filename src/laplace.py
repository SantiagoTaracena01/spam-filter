# Laboratorio 2
# Inteligencia Artificial
# Authors: YongBum Park 20117, Santiago Taracena 20017, Pedro Arriola 20188, Oscar López 20679

import re

# Definición de la clase LaplaceClassifier.
class LaplaceClassifier(object):

    # Método constructor de la clase LaplaceClassifier.
    def __init__(self):

        # Declaración de propiedades iniciales.
        self.spam_prob = 0
        self.ham_prob = 0
        self.spam_words = {}
        self.ham_words = {}

    # Método para entrenar el modelo.
    def train(self, train_path):

        # Leer el archivo de entrenamiento
        with open(train_path, "r") as f:
            train_messages = f.readlines()

        # Contar la frecuencia de cada palabra en los mensajes de spam y de ham
        for message in train_messages:

            # Separación del mensaje en palabras.
            label, text = message.strip().split("\t")
            words = re.findall(r'\b\w+\b', text.lower())

            # Conteo de palabras.
            if (label == "spam"):
                self.spam_prob += 1
                for word in words:
                    self.spam_words[word] = self.spam_words.get(word, 0) + 1
            elif (label == "ham"):
                self.ham_prob += 1
                for word in words:
                    self.ham_words[word] = self.ham_words.get(word, 0) + 1

        # Calcular la probabilidad de cada palabra en pertenecer a spam o ham con Laplace Smoothing
        num_spam_words = len(self.spam_words)
        num_ham_words = len(self.ham_words)

        # Probabilidad de ser spam.
        for word in self.spam_words:
            self.spam_words[word] = (self.spam_words[word] + 1) / (num_spam_words + 2)

        # Probabilidad de ser ham.
        for word in self.ham_words:
            self.ham_words[word] = (self.ham_words[word] + 1) / (num_ham_words + 2)

        # Probabilidad final de que sea spam o ham.
        self.spam_prob = self.spam_prob / len(train_messages)
        self.ham_prob = self.ham_prob / len(train_messages)

    # Método para clasificar un mensaje.
    def classify(self, message):
        
        # Probabilidades iniciales.
        p_spam = 1
        p_ham = 1

        # Calcular la probabilidad de que el mensaje sea spam y de que sea ham.
        for word in re.findall(r'\b\w+\b', message.lower()):
            p_spam *= self.spam_words.get(word, 1)
            p_ham *= self.ham_words.get(word, 1)
        p_spam *= self.spam_prob
        p_ham *= self.ham_prob

        # Clasificar el mensaje según la probabilidad resultante
        if (p_spam >= p_ham):
            return "spam", p_spam, p_ham
        else:
            return "ham", p_spam, p_ham
