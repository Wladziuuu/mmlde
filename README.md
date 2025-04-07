## Projekt naukowy wykrywania złośliwego oprogramowania

MMLDE to zaawansowany system detekcji malware'u wykorzystujący głębokie uczenie maszynowe. Projekt łączy w sobie nowoczesne techniki analizy plików binarnych z interfejsem graficznym ułatwiającym wykorzystanie modelu przez użytkowników końcowych.

## Kluczowe cechy projektu

### Zaawansowana analiza plików
- **Metoda n-gramowa**: System wykorzystuje analizę 3-gramów z plików binarnych, konwertując je do reprezentacji heksadecymalnej
- **Ekstrakcja cech**: Automatyczne generowanie histogramów i statystyk dystrybucji bajtów
- **Przetwarzanie plików PE**: Specjalizacja w analizie plików wykonywalnych (PE32/PE32+)

### Architektura modelu
- **Sieć neuronowa**: Wielowarstwowy perceptron (MLP) z regularyzacją L2
- **Optymalizacja**: Zastosowanie Batch Normalization i Dropout dla lepszej generalizacji
- **Funkcja aktywacji**: Warstwa wyjściowa z sigmoidą dla klasyfikacji binarnej

### Zaawansowane techniki ML
- **Walidacja krzyżowa**: 5-fold stratified cross-validation
- **Strojenie hiperparametrów**: Automatyczne poszukiwanie optymalnej konfiguracji modelu
- **Wczesne zatrzymywanie**: Zapobieganie przeuczeniu poprzez monitoring metryk walidacyjnych

## Zastosowania praktyczne
- **Detekcja w czasie rzeczywistym**: Możliwość analizy pojedynczych plików poprzez interfejs GUI
- **Skalowalność**: Architektura pozwalająca na trenowanie modelu na dużych zbiorach danych
- **Bezpieczeństwo**: Funkcja automatycznego usuwania wykrytego złośliwego oprogramowania

## Metryki wydajności
System osiąga wysoką skuteczność w detekcji złośliwego oprogramowania, mierzoną poprzez:
- Dokładność (accuracy)
- Precyzję (precision)
- Czułość (recall)
- F1-score
- Analizę macierzy pomyłek
