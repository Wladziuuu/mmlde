import sys
import os
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QMessageBox, QVBoxLayout, QPushButton
from PyQt5.QtCore import Qt
from collections import Counter
import tensorflow as tf
from tensorflow.keras.models import load_model

class FileProcessorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('MMLDE')
        self.setGeometry(100, 100, 400, 200)

        layout = QVBoxLayout()

        self.label = QLabel('Przeciągnij plik tutaj aby przeprowadzić analizę', self)
        self.label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.label)

        self.setLayout(layout)

        # Ustawienie okna jako miejsce do przeciągania plików
        self.setAcceptDrops(True)

        self.model = load_model('mmlde.keras')

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files:
            file_path = files[0]
            self.process_file(file_path)

    def generate_data(self, n, input_filepath):
        # Initialize n-grams dictionary
        ngrams = {f"{i:0{2*n}x}": 0 for i in range(256**n)}
        
        try:
            # Read the file content
            with open(input_filepath, 'rb') as file:
                file_content = file.read()
            
            # Convert file content to hexadecimal string
            hex_string = file_content.hex()
            
            # Extract n-grams
            ngrams_list = [hex_string[i:i+2*n] for i in range(0, len(hex_string)-(2*n - 1), 2)]
            
            # Create histogram
            histogram = Counter(ngrams_list)
            
            # Update ngrams dictionary with counts from histogram
            for ngram, count in histogram.items():
                if ngram in ngrams:
                    ngrams[ngram] += count
            
            # Convert ngrams dictionary values to a list (histogram)
            histogram_values = list(ngrams.values())
            
            interval_length = 16**n
            intervals = [histogram_values[i:i+interval_length] for i in range(0, len(histogram_values), interval_length)]
            means = [sum(interval) / len(interval) for interval in intervals]

            return tf.expand_dims(tf.convert_to_tensor(means, dtype=tf.float32), axis=0)
        
        except Exception as e:
            QMessageBox.critical(self, 'Błąd', f'Nie udało się przetworzyć pliku: {e}')
            return None

    def process_file(self, file_path):
        try:
            values = self.generate_data(3, file_path)

            prediction = self.model.predict(values)[0]

            threshold = 0.7 # change if necessary or if model returns 0/1 use result = prediction

            result = 1 if prediction >= threshold else 0

            if result == 0:
                QMessageBox.information(self, 'Wynik', f'Plik jest bezpieczny.\nProcent zgodności: {prediction}')
            else:
                reply = QMessageBox.question(self, 'Wynik', f'Plik jest niebezpieczny! Usunąć?\nProcent zgodności: {prediction}',
                                            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
                if reply == QMessageBox.Yes:
                    try:
                        os.remove(file_path)
                        QMessageBox.information(self, 'Wynik', 'Plik został usunięty')
                    except Exception as e:
                        QMessageBox.critical(self, 'Błąd', f'Nie udało się usunąć pliku: {e}')
        
        except Exception as e:
            QMessageBox.critical(self, 'Błąd', f'{e}')

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = FileProcessorApp()
    ex.show()
    sys.exit(app.exec_())
