import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import os
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

model_path="./mmlde.keras"

threshold = 0.3

# Funkcja do wczytania danych testowych
def load_test_data(folder_path):
    files = [os.path.join(folder_path, plik) for plik in os.listdir(folder_path) ]
    X_test = []
    for pelna_sciezka in files:
        with open(pelna_sciezka, 'r') as f:
            liczby = [float(linia.strip()) for linia in f if linia.strip().replace('.', '', 1).isdigit()]
            X_test.append(liczby)
    # Paduj sekwencje do długości input_dim
    X_test_padded = tf.keras.preprocessing.sequence.pad_sequences(X_test, padding='post', dtype='float32')
    return X_test_padded


# Ścieżka do foldera z danymi testowymi
test_folder_path_0 = "./test/benign/pe32"
test_folder_path_1 = "./test/malware/pe32"
test_folder_path_0_1 = "./test/benign/pe32p"
test_folder_path_1_1 = "./test/malware/pe32p"

# Wczytaj dane testowe
X_test_0 = load_test_data(test_folder_path_0) 
X_test_0_1= load_test_data(test_folder_path_0_1)
X_test_1 = load_test_data(test_folder_path_1) 
X_test_1_1 = load_test_data(test_folder_path_1_1)
print(len(X_test_0))
print(len(X_test_1))
print(len(X_test_0_1))

print(f"Model: {model_path}")

model = load_model(model_path)

# Przewidywanie dla danych testowych
predictions_0 = model.predict(X_test_0)
predictions_0_0 = model.predict(X_test_0_1)
predictions_1 = model.predict(X_test_1)
predictions_1_1 = model.predict(X_test_1_1)
labels0 = [0 if i < threshold else 1 for i in predictions_0 ]
labels1 = [1 if i >= threshold else 0 for i in predictions_1]
T = len(labels1)
print(T)
F = len(labels0)
print(T)
TP = labels1.count(1)
TN = labels0.count(0)

FP = labels0.count(1)
FN = labels1.count(0)
conf = {"TP": TP, "FP": FP, "TN": TN, "FN": FN}
accuracy = (TP + TN) / (T + F)
precision = TP / (TP + FP) if (TP + FP) > 0 else 0
recall = TP / (TP + FN) if (TP + FN) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
    }
accuracy_0 = np.mean(predictions_0 < threshold) 
accuracy_1 = np.mean(predictions_1 >= threshold)
y_true = [0] * len(X_test_0) + [1] * len(X_test_1)
y_pred = labels0 + labels1
# Oblicz macierz pomyłek
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malware'], yticklabels=['Benign', 'Malware'])
plt.xlabel('Predicted')
plt.ylabel('True')    
plt.title(f'Confusion Matrix for {model_path}')
plt.savefig(f'confusion_matrix_{os.path.basename(model_path)}.png')
plt.close()
print(conf)
print(metrics)
print(f"Dokładność dla klasy 0: {accuracy_0}")
print(f"Dokładność dla klasy 1: {accuracy_1}")
print(f"Macierz pomyłek:\n{cm}")
