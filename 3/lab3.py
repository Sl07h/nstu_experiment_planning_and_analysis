import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import det, inv, norm

k = 2       # число переменных 2: (x,y)
m = 6       # число параметров a + b*x + c*y + d*x*y + e*x^2 + f*y^2 
MAX_ITER = 50

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
class Lab3():
    '''
    Класс для 3 лабораторной работы
    Для стандартизации все критерии ищут минимальное значение
    '''
    def __init__(self, N, width, delta):
        ''' Выделение памяти под массивы '''
        n = width**2
        self.x_grid = np.ndarray((n, k))
        self.x_plan = np.ndarray((N, k))
        self.p = np.full(N, 1/N)
        self.M = np.ndarray((m, m))
        self.D = np.ndarray((m, m))
        self.width = width
        self.epsilon = 0.001
        self.n = n
        self.delta = delta
        self.max_iter = MAX_ITER
        self.N = N

    def Fedorov_algorithm(self, do_visualisation = False):
        '''
        Алгоритм Фёдорова синтеза дискретного
        оптимального плана эксперимента
        '''
        self.generate_initial_guess()
        do_calc = True
        s = 0
        result = np.zeros(self.max_iter)

        while do_calc == True and s < self.max_iter:
            self.build_matrix_M()
            self.build_matrix_D()
            D = self.calc_D()
            max_delta, i, j = self.find_two_points()
            self.x_plan[i] = self.x_grid[j]
            do_calc = not (max_delta < self.delta)
            result[s] = D
            if s % 10 == 0 and do_visualisation:
                self.draw_plan_on_iteration(s)
            s += 1
            print(s, i, j, max_delta, D)

        if do_calc == False:
            for i in range(s, self.max_iter):
                result[i] = result[s-1]

        if do_visualisation:
            self.draw_plan_on_iteration(s)
        return result

    def draw_plan_on_iteration(self, s):
        t = np.linspace(-1, 1, 11)
        plt.title('Алгоритм Фёдорова на шаге ' + str(s))
        plt.scatter([self.x_plan[i][0] for i in range(len(self.x_plan))],[self.x_plan[i][1] for i in range(len(self.x_plan))], )
        plt.xticks(t)
        plt.yticks(t)
        plt.savefig('pics/plan_Fedorov_{}_{}_{:.3f}_{}.png'.format(self.N, self.width, self.delta, s))
        plt.clf()

    def find_two_points(self):
        ''' Поиск точки плана и сетки для их замены '''
        # для начала:
        indicies = np.ndarray((self.N, 2), dtype = np.int64) # Пары i,j
        max_deltas = np.full(self.N, -9000.0)

        # перебираем все точки плана и все точки сетки
        for i, int_point in enumerate(self.x_plan):
            for j, ext_point in enumerate(self.x_grid):
                delta = self.Delta(int_point, ext_point)[0]
                # выбрали пару точек получше
                if delta > max_deltas[i]:
                    max_deltas[i] = delta
                    indicies[i][0] = i
                    indicies[i][1] = j

        max_delta_res = -9000.0
        I = 0
        J = 0
        for i, max_delta in enumerate(max_deltas):
            if max_delta > max_delta_res:
                max_delta_res = max_delta
                I = indicies[i][0]
                J = indicies[i][1]
        
        return max_delta_res, I, J

    def generate_initial_guess(self):
        ''' Задаём начальное приближение '''
        # создаём сетку
        self.t = np.linspace(-1, 1, self.width)
        i = 0
        for x1 in self.t:
            for x2 in self.t:
                self.x_grid[i] = np.array([x1, x2])
                i+=1

        # случайно выбираем точки плана и сохраняем
        # for i in range(self.N):
        #     self.x_plan[i] = self.x_grid[np.random.choice(self.n)]
        # np.savetxt('plans/plan_{}x{}_{}.txt'.format(self.width, self.width, self.N), self.x_plan)
        
        # или же загружаем
        self.x_plan = np.loadtxt('plans/plan_{}x{}_{}.txt'.format(self.width, self.width, self.N), dtype=np.float)

    def build_matrix_M(self):
        ''' Построение информационной матрицы M '''
        self.M = np.zeros((m, m))
        for i in range(self.N):
            self.M += (self.p[i] * f_vector(self.x_plan[i]) * f_vector_T(self.x_plan[i]))

    def build_matrix_D(self):
        ''' Построение дисперсионной матрицы D '''
        self.D = inv(self.M)

    def calc_D(self):
        ''' 
        Критерий D - оптимальности. (D - determinant)
        Эллипсоид рассеивания имеет минимальный объём
        '''
        return np.log(det(self.D))

    def Delta(self, x_j, x):
        N = float(self.N)
        return ((self.d(x) - self.d(x_j)) / N) - \
            ((self.d(x) * self.d(x_j) - self.d_2(x,x_j)**2)/(N**2))

    def d(self, x):
        return f_vector_T(x) @ self.D @ f_vector(x)

    def d_2(self, x, x_j):
        return f_vector_T(x) @ self.D @ f_vector(x_j)


    def draw_Mitchell_on_s(self, s):
        t = np.linspace(-1, 1, 11)
        plt.title('Алгоритм Митчелла на шаге ' + str(s))
        plt.scatter([self.x_plan[i][0] for i in range(len(self.x_plan))],[self.x_plan[i][1] for i in range(len(self.x_plan))], )
        plt.xticks(t)
        plt.yticks(t)
        plt.savefig('pics/plan_Mitchell_{}_{}_{:.3f}_{}.png'.format(self.N, self.width, self.delta, s))
        plt.clf()


    def Mitchell_algorithm(self, do_visualisation = True):
        '''
        Алгоритм Митчелла синтеза дискретного
        оптимального плана эксперимента
        '''
        self.generate_initial_guess()
        do_calc = True
        s = 0
        result = np.zeros(self.max_iter)

        while do_calc == True and s < self.max_iter:
            self.build_matrix_M()
            self.build_matrix_D()
            new_p, i = self.find_new_point()
            old_p, j = self.find_old_point()
            do_calc = norm(new_p - old_p) >= self.epsilon
            print(s+1, self.calc_D(), new_p, old_p)
            self.x_plan[j] = self.x_grid[i]
            result[s] = self.calc_D()
            
            if s % 10 == 0 and do_visualisation:
                self.draw_Mitchell_on_s(s)
            s += 1

            if do_calc == False:
                for i in range(s, self.max_iter):
                    result[i] = result[s-1]

        if do_visualisation:
            self.draw_Mitchell_on_s(s)
        return result

    def find_new_point(self):
        ''' Выбор новой точки плана  max d(x) '''
        new_i = 0
        new_point = self.x_grid[0]
        max_d = self.d(new_point)

        for i, point in enumerate(self.x_grid):
            d = self.d(point)
            if d > max_d:
                new_i = i
                new_point = point
        
        return new_point, new_i

    def find_old_point(self):
        ''' Выбор страрой точки плана min d(x) '''
        new_i = 0
        new_point = self.x_plan[0]
        min_d = self.d(new_point)

        for i, point in enumerate(self.x_plan):
            d = self.d(point)
            if d < min_d:
                new_i = i
                new_point = point
        
        return new_point, new_i



#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
t = np.linspace(0, MAX_ITER, 11)

def perform_experiment(N, width, delta):
    l3 = Lab3(N, width, delta)
    return l3.Fedorov_algorithm()

def research_delta():
    ''' Исследование скорости сходимости и устойчивости от delta '''
    print('Исследование скорости сходимости и устойчивости от delta')
    width = 21
    for N in [20, 40]:
        for delta in [0.001, 0.01, 0.1]:
            y = perform_experiment(N, width, delta)
            plt.plot(y, label=str(delta))    

        plt.title('D-оптимальность плана от delta')
        plt.legend(title='сетка: {}x{}\nN: {}'.format(width, width, N))
        plt.xticks(t)
        plt.xlabel('итерации')
        plt.ylabel(r'$\log(\left| M^{-1}(\varepsilon) \right|)$')
        plt.savefig('pics/research_delta_{}x{}_{}.png'.format(width, width, N), dpi=200)
        plt.clf()

def research_N():
    ''' Исследование скорости сходимости и устойчивости от N '''
    print('Исследование скорости сходимости и устойчивости от N')
    delta = 0.01
    for width in [11, 21]:
        for N in [20, 25, 30, 35, 40]:
            y = perform_experiment(N, width, delta)
            plt.plot(y, label=str(N))    
        plt.title('Влияние числа узлов плана')
        plt.legend(title='сетка: {}x{}\ndelta: {:.3f}'.format(width, width, delta))
        plt.xticks(t)
        plt.xlabel('итерации')
        plt.ylabel(r'$\log(\left| M^{-1}(\varepsilon) \right|)$')
        plt.savefig('pics/research_N_{}x{}.png'.format(width, width), dpi=200)
        plt.clf()

def research_width():
    ''' Исследование скорости сходимости и устойчивости от числа узлов сетки '''
    print('Исследование скорости сходимости и устойчивости от числа узлов сетки')
    delta = 0.001
    for width in [11, 21]:
        for N in [20, 40]:
            y = perform_experiment(N, width, delta)
            plt.plot(y, label='N: {}  n: {}'.format(N, width**2))    

    plt.title('Влияние числа узлов сетки')
    plt.legend(title='delta: {:.3f}'.format(delta))
    plt.xticks(t)
    plt.xlabel('итерации')
    plt.ylabel(r'$\log(\left| M^{-1}(\varepsilon) \right|)$')
    plt.savefig('pics/research_width.png', dpi=200)
    plt.clf()

def show_convergence_Fedorov(N, width, delta):
    print(N, width, delta)
    l3 = Lab3(N, width, delta)
    y = l3.Fedorov_algorithm(True)
    plt.plot(y)
    plt.title('Сходимость метода Фёдорова')
    plt.text(24, 6, 'сетка: {}x{}\nN: {}\ndelta: {:.3f}'.format(width, width, N, delta))
    plt.xticks(t)
    plt.xlabel('итерации')
    plt.ylabel(r'$\log(\left| M^{-1}(\varepsilon) \right|)$')
    plt.savefig('pics/convergence_Fedorov_{}_{}_{:.3f}.png'.format(N, width, delta), dpi=200)
    plt.clf()




#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# research_delta()
# research_N()
# research_width()
# show_convergence_Fedorov(30, 11, 0.01)
# show_convergence_Fedorov(30, 21, 0.01)



#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
def show_convergence_Mitchell(N, width, delta):
    print(N, width, delta)
    l = Lab3(N, width, delta)
    y = l.Mitchell_algorithm()
    plt.plot(y)
    plt.title('Сходимость метода Митчелла')
    plt.text(24, 6, 'сетка: {}x{}\nN: {}\ndelta: {:.3f}'.format(width, width, N, delta))
    plt.xticks(t)
    plt.xlabel('итерации')
    plt.ylabel(r'$\log(\left| M^{-1}(\varepsilon) \right|)$')
    plt.savefig('pics/convergence_Mitchell_{}_{}_{:.3f}.png'.format(N, width, delta), dpi=200)
    plt.clf()


show_convergence_Mitchell(30, 21, 0.001)