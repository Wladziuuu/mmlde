import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import StratifiedKFold
import numpy as np
from keras_tuner import HyperModel
from keras_tuner.tuners import RandomSearch
gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[0], 'GPU')
batch_size = 32
input_dim = 4096
num_classes = 1

class VirusDetectionHyperModel(HyperModel):
    def build(self, hp):
        model = Sequential()
        
        model.add(Dense(units=hp.Int('units_2', min_value=128, max_value=256, step=8),
                        activation='relu', kernel_regularizer=l2(hp.Float('l2_2', 0.001, 0.01, sampling='log'))))
        model.add(BatchNormalization())
        model.add(Dropout(rate=hp.Float('dropout_2', 0.2, 0.5, step=0.1)))
        
        model.add(Dense(units=hp.Int('units_3', min_value=64, max_value=128, step=8),
                        activation='relu', kernel_regularizer=l2(hp.Float('l2_3', 0.001, 0.01, sampling='log'))))
        model.add(BatchNormalization())
        model.add(Dropout(rate=hp.Float('dropout_3', 0.2, 0.5, step=0.1)))
        
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer=Adam(learning_rate=hp.Float('learning_rate', 0.4, 1e-3, sampling='log')),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])

        return model

def load_data(folder_path_0, folder_path_2):
    files_0 = [os.path.join(folder_path_0, plik) for plik in os.listdir(folder_path_0) if os.path.isfile(os.path.join(folder_path_0, plik))]
    #files_1 = [os.path.join(folder_path_1, plik) for plik in os.listdir(folder_path_1) if os.path.isfile(os.path.join(folder_path_1, plik))]
    files_2 = [os.path.join(folder_path_2, plik) for plik in os.listdir(folder_path_2) if os.path.isfile(os.path.join(folder_path_2, plik))]
    #files_3 = [os.path.join(folder_path_3, plik) for plik in os.listdir(folder_path_3) if os.path.isfile(os.path.join(folder_path_3, plik))]
    X = []
    Y = []
    for file in files_0:
        with open(file, 'r') as f:
            liczby = [float(linia.strip()) for linia in f if linia.strip().replace('.', '', 1).isdigit()]
            X.append(liczby)
        Y.append(0)
    
    #for file in files_1:
    #    with open(file, 'r') as f:
     #       liczby = [float(linia.strip()) for linia in f if linia.strip().replace('.', '', 1).isdigit()]
      #      X.append(liczby)
       # Y.append(2)
    for file in files_2:
        with open(file, 'r') as f:
            liczby = [float(linia.strip()) for linia in f if linia.strip().replace('.', '', 1).isdigit()]
            X.append(liczby)
        Y.append(1)
    #for file in files_3:
     #   with open(file, 'r') as f:
      #      liczby = [float(linia.strip()) for linia in f if linia.strip().replace('.', '', 1).isdigit()]
       #     X.append(liczby)
        #Y.append(4)    
    X_padded = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=input_dim, padding='post', dtype='float32')
    return np.array(X_padded), np.array(Y)

# Załaduj dane
X, Y = load_data("./train/benign/pe32",  "./train/malware/pe32")

# Ustawienia tunera
tuner = RandomSearch(
    VirusDetectionHyperModel(),
    objective='val_accuracy',
    max_trials=60,
    executions_per_trial=2,
    directory='tuner_dir',
    project_name='virus_detection')

# Definicja StratifiedKFold
skf = StratifiedKFold(n_splits=5)

# Listy do zapisywania wyników
val_scores = []
best_models = []
rlr = ReduceLROnPlateau(monitor='val_accuracy', patience=10, mode='max', factor=0.5)
csv_logger = CSVLogger('log.csv', append=True, separator=';')
# Pętla cross-validation
for train_index, val_index in skf.split(X, Y):
    X_train, X_val = X[train_index], X[val_index]
    Y_train, Y_val = Y[train_index], Y[val_index]
    
    early_stop = EarlyStopping(monitor="val_loss", patience=45, verbose=1, restore_best_weights=True)
    checkpoint = ModelCheckpoint('model_best_fold.keras', monitor='val_loss', save_best_only=True, verbose=1)
    tuner.search(X_train, Y_train,
                 epochs=500,
                 validation_data=(X_val, Y_val),
                 callbacks=[early_stop, checkpoint,rlr])

    # Wyświetl najlepsze hiperparametry dla tego folda
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"""
    Najlepsze hiperparametry dla folda:
      - units_2: {best_hps.get('units_2')}
      - l2_2: {best_hps.get('l2_2')}
      - dropout_2: {best_hps.get('dropout_2')}
      - units_3: {best_hps.get('units_3')}
      - l2_3: {best_hps.get('l2_3')}
      - dropout_3: {best_hps.get('dropout_3')}
      - learning_rate: {best_hps.get('learning_rate')}
    """)

    # Zbudowanie modelu z najlepszymi hiperparametrami
    model = Sequential([
        Dense(units=best_hps.get('units_2'), activation='relu', kernel_regularizer=l2(best_hps.get('l2_2'))),
        BatchNormalization(),
        Dropout(rate=best_hps.get('dropout_2')),

        Dense(units=best_hps.get('units_3'), activation='relu', kernel_regularizer=l2(best_hps.get('l2_3'))),
        BatchNormalization(),
        Dropout(rate=best_hps.get('dropout_3')),

        Dense(num_classes, activation='sigmoid')
    ])

    model.compile(optimizer=Adam(learning_rate=best_hps.get('learning_rate')),
                  loss='binary_crossentropy',
                  metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

    # Trenowanie modelu
    model.fit(X_train, Y_train,
              epochs=500,
              validation_data=(X_val, Y_val),
              callbacks=[early_stop, checkpoint],
              batch_size=batch_size)

    # Zapisz najlepszy wynik walidacji i model
    val_score = model.evaluate(X_val, Y_val, verbose=0)
    val_scores.append(val_score)
    best_models.append(model)

# Wyświetl wyniki walidacji
for i, score in enumerate(val_scores):
    print(f"Fold {i+1} - Loss: {score[0]}, Accuracy: {score[1]}, Precision: {score[2]}, Recall: {score[3]}")

# Zapisz model z najlepszym wynikiem walidacji
best_model_index = np.argmax([score[1] for score in val_scores])
best_model = best_models[best_model_index]
best_model.save('final_best_model.keras')
