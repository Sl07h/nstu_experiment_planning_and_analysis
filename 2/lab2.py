import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.linalg import det, inv


pd.set_option('precision', 3)
plt.rcParams.update({'font.size': 14})
k = 2       # число переменных 2: (x,y)
m = 6       # число параметров a + b*x + c*y + d*x*y + e*x^2 + f*y^2 
n = 25      # число точек сетки
width = 5   # сторона сетки

def f(theta, x):
    return  theta[0] + \
            theta[1]*x[0] + \
            theta[2]*x[1] + \
            theta[3]*x[0]*x[1] + \
            theta[4]*x[0]**2 + \
            theta[5]*x[1]**2

def f_vector(x):
    return np.array([
        [1],
        [x[0]],
        [x[1]],
        [x[0]*x[1]],
        [x[0]**2],
        [x[1]**2]
    ])

def f_vector_T(x):
    return np.array([
        1,
        x[0],
        x[1],
        x[0]*x[1],
        x[0]**2,
        x[1]**2
    ])


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
class Lab2():
    '''
    Класс для 2 лабораторной работы
    Для стандартизации все критерии ищут минимальное значение
    '''
    def __init__(self):
        ''' Выделение памяти под массивы '''
        self.x = np.ndarray((n, k))
        self.p = np.ndarray(n)
        self.t = np.linspace(-1, 1, width)
        self.M = np.ndarray((m, m))
        self.D = np.ndarray((m, m))
        self.alpha = 1 / n
        self.gamma = 2
        self.max_iter_s = 30
        self.max_iter_alpha = 20
        self.to_report = [1, 10, 20, 30]

    def generate_initial_guess(self):
        ''' Задаём начальное приближение '''
        i = 0
        for x1 in self.t:
            for x2 in self.t:
                self.x[i] = np.array([x1, x2])
                i+=1
        self.p = np.full(n, 1/n)

    def fi(self, x):
        ''' Значение функции fi в точке x '''
        return f_vector_T(x) @ self.D @ f_vector(x)

    def max_fi(self):
        ''' Поиск максимального значения fi '''
        max_fi = -9000
        for point in self.x:
            fi = self.fi(point)
            if fi > max_fi:
                max_fi = fi
        return max_fi

    def is_plan_optimal(self):
        '''
        Проверяем выполнение необходимых и достаточных
        условий оптимальности плана
        '''
        max_fi = self.max_fi()
        delta = 0.01 * abs(max_fi)
        if abs(-max_fi + np.trace(self.M @ self.D)) <= delta:
            return True
        else:
            return False

    def remove_points(self):
        ''' 
        Удаление из плана незначительных точек. План после очистки
        +-+-+
        -----
        +-+-+
        -----
        +-+-+        
        '''
        global n, width
        n = 9
        width = 3

        x = np.copy(self.x)
        p = np.copy(self.p)
        self.t = np.linspace(-1,1,3)

        weight_to_mult = p[[1,3,11,13,21,23]].sum() + \
                        np.sum(p[5:10]) + \
                        np.sum(p[15:20])
        p /= 1 - weight_to_mult 

        self.x = np.copy(x[[0,2,4,10,12,14,20,22,24]])
        self.p = np.copy(p[[0,2,4,10,12,14,20,22,24]])

    def clear_plan(self):
        ''' Процедура очистки плана '''
        global n
        x = np.ndarray((width**2, k))
        p = np.ndarray(width**2)

        i = 0
        for x1 in self.t:
            for x2 in self.t:
                x[i] = np.array([x1, x2])
                i+=1
        p = np.zeros(width**2)

        for point, weight in zip(self.x, self.p):
            for i in range(width):
                for j in range(width):
                    a = point
                    b = x[i*width+j]
                    if a[0]==b[0] and a[1]==b[1]:
                        p[i*width+j] += weight
        self.x = np.copy(x)
        self.p = np.copy(p)
        n = width**2

    def calc_new_point(self):
        ''' Выбираем новую точку плана '''
        max_fi = -9000
        new_point = self.x[0]
        for point in self.x:
            fi = self.fi(point)
            if fi > max_fi:
                max_fi = fi
                new_point = point
        return new_point

    def add_new_point(self):
        ''' Добавляем в план новую точку x_s '''
        global n
        n += 1
        x_s = self.calc_new_point()
        self.x = np.append(self.x, [x_s], axis=0)
        self.p = np.append(self.p * (1 - self.alpha), self.alpha) 
        
    def draw_plan(self, iteration):
        ''' Отрисовка весов плана эксперимента '''
        x, y = np.hsplit(self.x, 2)
        plt.scatter(x, y)
        for i, txt in enumerate(self.p):
            plt.annotate(str(int(txt*100)), (x[i] + 0.05, y[i] - 0.05))
        plt.xticks(self.t)
        plt.yticks(self.t)
        plt.title('План на шаге: ' + str(iteration), fontsize=20)
        plt.grid(alpha=0.4)
        plt.margins(0.1)
        plt.savefig('pics/plan' + str(iteration) + '.png')
        plt.clf()

    def build_table(self, iteration):
        ''' Построение таблицы весов плана эксперимента'''
        tmp = np.copy(self.p)
        t = tmp.reshape((width, width))
        d = pd.DataFrame(data = t, columns=self.t, index=self.t)
        filename = 'tables/' + str(iteration) + '.tex' 
        with open(filename, 'w') as f:
            f.writelines(d.to_latex())
        
    def sequential_algorithm(self):
        '''
        Последовательный алгоритм синтеза непрерывного
        оптимального плана эксперимента
        '''
        self.generate_initial_guess()
        do_calc = True
        s = 0

        while do_calc == True and s < self.max_iter_s:
            a = 0
            self.alpha = 1 / n
            self.build_matrix_M()
            self.build_matrix_D()
            psi = self.calc_D()
            self.add_new_point()
            psi_next = self.calc_D()

            # Уменьшаем шаг, если метод расходится
            while psi_next >= psi and a < self.max_iter_alpha:
                a += 1
                self.alpha /= self.gamma
                psi = psi_next
                self.add_new_point()
                psi_next = self.calc_D()

            print(s+1, a)
            self.clear_plan()
            
            # Строим таблицы и графики
            if s in self.to_report:
                self.draw_plan(s)
                self.build_table(s)

            do_calc = not self.is_plan_optimal()
            s += 1

        self.remove_points()
        self.draw_plan(s)
        self.build_table(s)

    def build_matrix_M(self):
        ''' Построение информационной матрицы M '''
        self.M = np.zeros((m, m))
        for i in range(n):
            self.M += self.p[i] * f_vector(self.x[i]) * f_vector_T(self.x[i])

    def build_matrix_D(self):
        ''' Построение дисперсионной матрицы D '''
        self.D = inv(self.M)

    def calc_D(self):
        ''' 
        Критерий D - оптимальности. (D - determinant)
        Эллипсоид рассеивания имеет минимальный объём
        '''
        return np.log(det(self.M))



#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

l2 = Lab2()
l2.sequential_algorithm()