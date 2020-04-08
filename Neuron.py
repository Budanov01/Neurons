import math


# При инициализации нейрона в конструктор передаются параметры в следующем порядке - матрица возможных входов,
# вектор правильного входа, веса, кол-во эпох и скорость обучения
class Neuron:
    def __init__(self, x_matrix, t_vector, weights, age, n):
        self.__x_matrix = x_matrix  # Матрица возможных входов
        self.__t_vector = t_vector  # Вектор целевых значений нейрона
        self.__weights = weights  # Веса нейрона, последнее значение - смещение
        self.__age = age  # Колличество эпох
        self.__n = n  # Скорость обучения

    # функция активации (сигмоид)
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    # Тут нейрон будет обучается когда подрастет
    def learning(self):
        while self.__age >= 0:
            # Пройдемся по каждым возможным входным значениям перемножив их на веса / считаем полином
            z = []  # Вектор полученных значений
            for row in self.__x_matrix:
                y = 0
                for i in range(len(self.__weights)):
                    y += row[i] * self.__weights[i]
                z.append(self.sigmoid(y))

            # Посчитаем ошибку и вычтим ее из весов
            for j in range(len(self.__weights) - 1):
                delta = 0
                for i in range(len(z)):
                    delta += (z[i] - self.__t_vector[i]) * z[i] * (1 - z[i]) * self.__x_matrix[i][j]
                delta *= 2
                self.__weights[j] -= delta * self.__n
            self.__age -= 1

    # Пришло время показать свои знания Mr Neuron
    def reaction(self, x):
        total = 0
        for j in range(len(x)):
            total += x[j] * self.__weights[j]

        return self.sigmoid(total)