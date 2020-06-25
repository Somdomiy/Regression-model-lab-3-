import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

df = pd.read_csv('abs.csv')
df = df.replace('кирпичный', 0).replace('панельный', 1).replace('монолитный', 2).replace('блочный', 3).replace('деревянный', 4)

y = df['price'].values # Результаты
del df['price'] # удаляем столбец с ценой квартиры
#del df['district']
x = df.values # Данные

model = LinearRegression().fit(x, y)
#С помощью .fit() вычисляются оптимальные значение весов

r_sq = model.score(x, y) # Результат обучения модели (точность)
print('r_sq: ' + str(r_sq))

x_pred = [
    [60, 2, 4, 0] # 60квм, 2 комнаты , 4 этаж, кирпичный
]
y_pred = model.predict(x_pred) # Предсказание по придуманным данным
print('predict: ', y_pred, sep='\n')


y_p = model.predict(x)
plt.scatter(y,y_p,color = 'red')
plt.plot(y, y, color='blue', linewidth=2)
plt.show() # Визуализация результатов


