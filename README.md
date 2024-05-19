Analiza Regresji w Pythonie

Opis projektu

Projekt demonstruje zastosowanie różnych metod regresji (wielomianowej, 1/x i liniowej) do analizy danych. Wykorzystano bibliotekę NumPy do obliczeń matematycznych, matplotlib do wizualizacji wyników oraz scikit-learn do podziału danych na zbiory treningowe i testowe oraz do oceny modelu za pomocą miary RMSE (Root Mean Squared Error).

Pliki

dane7.txt - plik z danymi wejściowymi zawierający dwie kolumny: wartości x i odpowiadające im wartości y.
regression_analysis.py - główny skrypt Pythona wykonujący analizę regresji i rysujący wykresy.
Wymagania

Python 3.x
NumPy
Matplotlib
Scikit-learn
Możesz zainstalować wymagane biblioteki za pomocą pip:

bash
Skopiuj kod
pip install numpy matplotlib scikit-learn
Instrukcja uruchomienia

Upewnij się, że plik dane7.txt znajduje się w tym samym katalogu co skrypt regression_analysis.py.
Uruchom skrypt za pomocą Python:
bash
Skopiuj kod
python regression_analysis.py
Opis skryptu

Wczytywanie danych
Skrypt rozpoczyna się od wczytania danych z pliku dane7.txt:

python
Skopiuj kod
data = np.loadtxt('dane7.txt')
x = data[:, [0]]
y = data[:, [1]]
Podział danych
Dane są podzielone na zbiory treningowe i testowe:

python
Skopiuj kod
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, shuffle=True)
Regresja wielomianowa (stopnia 3)
Macierz cech dla regresji wielomianowej trzeciego stopnia jest tworzona i współczynniki są obliczane za pomocą równań normalnych:

python
Skopiuj kod
c = np.hstack([x_train**3, x_train**2, x_train, np.ones(x_train.shape)])
v = np.linalg.inv(c.T @ c) @ c.T @ y_train
Regresja 1/x
Macierz cech dla regresji 1/x jest tworzona i współczynniki są obliczane za pomocą pseudoodwrotnej macierzy:

python
Skopiuj kod
c1 = np.hstack([1/x_train, np.ones(x_train.shape)])
v1 = np.linalg.pinv(c1) @ y_train
Regresja liniowa
Macierz cech dla regresji liniowej jest tworzona i współczynniki są obliczane za pomocą pseudoodwrotnej macierzy:

python
Skopiuj kod
c2 = np.hstack([x_train, np.ones(x_train.shape)])
v2 = np.linalg.pinv(c2) @ y_train
Obliczanie RMSE
Skrypt oblicza RMSE dla każdej metody regresji:

python
Skopiuj kod
rmse_poly = np.sqrt(mean_squared_error(y_test, y_pred_test_poly))
rmse_inv = np.sqrt(mean_squared_error(y_test, y_pred_test_inv))
rmse_lin = np.sqrt(mean_squared_error(y_test, y_pred_test_lin))

print(f'RMSE dla regresji wielomianowej: {rmse_poly}')
print(f'RMSE dla regresji 1/x: {rmse_inv}')
print(f'RMSE dla regresji liniowej: {rmse_lin}')
Wizualizacja wyników
Wyniki są wizualizowane za pomocą biblioteki matplotlib:

python
Skopiuj kod
plt.figure(figsize=(12, 6))
plt.scatter(x, y, color='red', label='Dane')
plt.plot(x_range, y_pred_range_poly, label='Regresja wielomianowa (stopnia 3)')
plt.plot(x_range, y_pred_range_inv, label='Regresja 1/x')
plt.plot(x_range, y_pred_range_lin, label='Regresja liniowa')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Różne typy regresji')
plt.grid(True)
plt.show()
Kontakt

W razie pytań lub uwag prosimy o kontakt na adres email@example.com.
