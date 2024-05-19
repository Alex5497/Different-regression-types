import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Wczytanie danych z pliku
data = np.loadtxt('dane7.txt')

x = data[:, [0]]
y = data[:, [1]]

# Podział danych na zbiór treningowy i testowy
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, shuffle=True)

# Regresja wielomianowa (stopnia 3)
c = np.hstack([x_train**3, x_train**2, x_train, np.ones(x_train.shape)])
v = np.linalg.inv(c.T @ c) @ c.T @ y_train

y_pred_train = v[0] * x_train ** 3 + v[1] * x_train ** 2 + v[2] * x_train + v[3]
y_pred_test_poly = v[0] * x_test ** 3 + v[1] * x_test ** 2 + v[2] * x_test + v[3]

# RMSE dla regresji wielomianowej
rmse_poly = np.sqrt(mean_squared_error(y_test, y_pred_test_poly))
print(f'RMSE dla regresji wielomianowej: {rmse_poly}')

# Regresja 1/x
c1 = np.hstack([1/x_train, np.ones(x_train.shape)])
v1 = np.linalg.pinv(c1) @ y_train

y_pred_test_inv = v1[0] / x_test + v1[1]

# RMSE dla regresji 1/x
rmse_inv = np.sqrt(mean_squared_error(y_test, y_pred_test_inv))
print(f'RMSE dla regresji 1/x: {rmse_inv}')

# Regresja liniowa
c2 = np.hstack([x_train, np.ones(x_train.shape)])
v2 = np.linalg.pinv(c2) @ y_train

y_pred_test_lin = v2[0] * x_test + v2[1]

# RMSE dla regresji liniowej
rmse_lin = np.sqrt(mean_squared_error(y_test, y_pred_test_lin))
print(f'RMSE dla regresji liniowej: {rmse_lin}')

# Rysowanie wykresów
plt.figure(figsize=(12, 6))

# Dane wyjściowe
plt.scatter(x, y, color='red', label='Dane')

# Regresja wielomianowa
x_range = np.linspace(x.min(), x.max(), 100)
y_pred_range_poly = v[0] * x_range ** 3 + v[1] * x_range ** 2 + v[2] * x_range + v[3]
plt.plot(x_range, y_pred_range_poly, label='Regresja wielomianowa (stopnia 3)')

# Regresja 1/x
y_pred_range_inv = v1[0] / x_range + v1[1]
plt.plot(x_range, y_pred_range_inv, label='Regresja 1/x')

# Regresja liniowa
y_pred_range_lin = v2[0] * x_range + v2[1]
plt.plot(x_range, y_pred_range_lin, label='Regresja liniowa')

# Ustawienia wykresu
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Różne typy regresji')
plt.grid(True)

# Wyświetlenie wykresu
plt.show()
