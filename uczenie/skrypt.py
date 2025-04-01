import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, ReduceLROnPlateau
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.utils import Sequence

# Parameters
input_dim = 4096  
num_classes = 1  # Binary output: virus or no virus
num_folds = 5  # Number of folds for K-Fold Cross Validation
batch_size = 128

# Data generator class
class DataGenerator(Sequence):
    def __init__(self, file_paths, labels, batch_size, input_dim):
        self.file_paths = file_paths
        self.labels = labels
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.indices = np.arange(len(self.file_paths))
        
    def __len__(self):
        return int(np.floor(len(self.file_paths) / self.batch_size))
    
    def __getitem__(self, index):
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        batch_files = [self.file_paths[i] for i in batch_indices]
        X, Y = self.__data_generation(batch_files)
        return X, np.array(Y)
    
    def on_epoch_end(self):
        np.random.shuffle(self.indices)
        
    def __data_generation(self, batch_files):
        X = []
        Y = []
        for file_path in batch_files:
            with open(file_path, 'r') as f:
                numbers = [float(line.strip()) for line in f if line.strip().replace('.', '', 1).isdigit()]
                X.append(numbers)
                Y.append(0 if 'benign' in file_path else 1)  # Assuming folder names include 'benign' and 'malware'
        X_padded = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=self.input_dim, padding='post', dtype='float32')
        return np.array(X_padded), np.array(Y)

# Custom step activation function
def step_activation(x):
    return tf.cast(tf.greater_equal(x, 0), tf.float32)

# Load file paths
folder_path_0 = "./train/benign/pe32"
folder_path_1 = "./train/malware/pe32"
folder_path_2 = "./train/benign/pe32+"
folder_path_3 = "./train/malware/pe32+"
files_0 = [os.path.join(folder_path_0, file) for file in os.listdir(folder_path_0) if os.path.isfile(os.path.join(folder_path_0, file))]
files_1 = [os.path.join(folder_path_1, file) for file in os.listdir(folder_path_1) if os.path.isfile(os.path.join(folder_path_1, file))]
files_2 = [os.path.join(folder_path_2, file) for file in os.listdir(folder_path_2) if os.path.isfile(os.path.join(folder_path_2, file))]
files_3 = [os.path.join(folder_path_3, file) for file in os.listdir(folder_path_3) if os.path.isfile(os.path.join(folder_path_3, file))]
all_files = files_0 + files_1 + files_2 + files_3
all_labels = [0]*len(files_0) + [1]*len(files_1) + [0]*len(files_2) + [1]*len(files_3)

# K-Fold Cross Validation
kf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=42)

fold_no = 1
for train_index, val_index in kf.split(all_files, all_labels):
    print(f'Training on fold {fold_no}...')
    
    train_files = [all_files[i] for i in train_index]
    val_files = [all_files[i] for i in val_index]
    train_labels = [all_labels[i] for i in train_index]
    val_labels = [all_labels[i] for i in val_index]
    
    train_generator = DataGenerator(train_files, train_labels, batch_size, input_dim)
    val_generator = DataGenerator(val_files, val_labels, batch_size, input_dim)
    
    model = Sequential()

    model.add(Dense(128, activation='relu', input_shape=(input_dim,), kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
 
    model.add(Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))

    model.add(Dense(2, activation='sigmoid'))
 


    model.compile(optimizer=Adam(learning_rate=0.001),
                  loss='binary_crossentropy',  # 'binary_crossentropy' for binary classification
                  metrics=['accuracy'])
 
    early_stop = EarlyStopping(monitor="val_loss", patience=25, verbose=1, restore_best_weights=True)
    rlr = ReduceLROnPlateau(monitor='val_accuracy', patience=10, mode='max', faactor=0.5)
    checkpoint = ModelCheckpoint(
        filepath=f'./model/finalsf/best_model_fold_{fold_no}.keras',
        monitor='val_loss',
        save_best_only=True,
        save_weights_only=False,
        mode='min',
        verbose=1
    )
    csv_logger = CSVLogger('log.csv', append=True, separator=';')
    model.fit(
        train_generator,
        epochs=1500,
        batch_size=batch_size,
        validation_data=val_generator,
        verbose=1,
        shuffle=True,
        callbacks=[early_stop, checkpoint, csv_logger, rlr]
    )
 
    fold_no += 1
 
# Save the final model
model.save('final_model.h5')