import random
import re

class TestTrain:
    
    def __init__(self, path):
        self.path = path
        random.seed(123)
        self.read()
        
    def read(self):
        # Leer el archivo de entrenamiento
        with open(self.path, "r") as f:
            self.lines = f.readlines()
        self.clearMessage()
            
    def clearMessage(self):
        # Limpiar y separar los mensajes
        self.messages = []
        for line in self.lines:
            self.label, self.text = line.strip().split("\t")
            # Limpiar el texto eliminando caracteres especiales y convirtiéndolo a minúsculas
            self.text = re.sub(r"[^\w\s]", "", self.text.lower())
            self.messages.append((self.label, self.text))
        self.train_test()
            
    def train_test(self):
        # Separar los mensajes en conjuntos de entrenamiento y prueba
        train_size = int(len(self.messages) * 0.8)
        self.train_messages = self.messages[:train_size]
        self.test_messages = self.messages[train_size:]
        self.validation_test()
        
    def validation_test(self):
        # Opcionalmente, subdividir el conjunto de prueba en conjuntos de validación y prueba
        val_size = int(len(self.test_messages) * 0.5)
        self.val_messages = self.test_messages[:val_size]
        self.test_messages = self.test_messages[val_size:]
        self.save()
        
    def save(self):
        # Guardar los conjuntos de entrenamiento, prueba y validación en archivos separados
        with open("train.txt", "w") as f:
            for label, text in self.train_messages:
                f.write(f"{label}\t{text}\n")
        with open("test.txt", "w") as f:
            for label, text in self.test_messages:
                f.write(f"{label}\t{text}\n")
        with open("val.txt", "w") as f:
            for label, text in self.val_messages:
                f.write(f"{label}\t{text}\n")
                