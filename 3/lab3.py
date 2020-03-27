import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import det, inv, norm

k = 2       # число переменных 2: (x,y)
m = 6       # число параметров a + b*x + c*y + d*x*y + e*x^2 + f*y^2 
MAX_ITER = 50
t = np.linspace(0, MAX_ITER, 11)


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

    def draw_plan_on_s(self, s, alg_name, path):
        t = np.linspace(-1, 1, 11)
        plt.title(alg_name + str(s))
        plt.scatter([self.x_plan[i][0] for i in range(len(self.x_plan))],[self.x_plan[i][1] for i in range(len(self.x_plan))], )
        plt.xticks(t)
        plt.yticks(t)
        plt.savefig(path + str(s) + '.png')
        plt.clf()

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
    
    def find_new_point(self):
        ''' Выбор новой точки плана  max d(x), x in grid '''
        new_i = 0
        new_point = self.x_grid[0]
        max_f = self.d(new_point)

        for i, point in enumerate(self.x_grid):
            f = self.d(point)
            if f > max_f:
                new_i = i
                new_point = point
        
        return new_point, new_i

    def find_old_point(self):
        ''' Выбор старой точки плана min d(x), x in plan '''
        new_i = 0
        new_point = self.x_plan[0]
        min_f = self.d(new_point)

        for i, point in enumerate(self.x_plan):
            f = self.d(point)
            if f < min_f:
                new_i = i
                new_point = point
        
        return new_point, new_i

    def build_matrix_M(self):
        ''' Построение информационной матрицы M '''
        self.M = np.zeros((m, m))
        for i in range(self.N):
            x = self.x_plan[i]
            self.M += self.p[i] * f_vector(x) * f_vector_T(x)

    def build_matrix_M_without_p(self):
        ''' Построение информационной матрицы M без использования весов плана'''
        self.M = np.zeros((m, m))
        for i in range(self.N):
            x = self.x_plan[i]
            self.M += f_vector(x) * f_vector_T(x)
        self.M /= self.N

    def build_matrix_D(self):
        ''' Построение дисперсионной матрицы D '''
        self.D = inv(self.M)

    def calc_D(self):
        ''' 
        Критерий D - оптимальности. (D - determinant)
        Эллипсоид рассеивания имеет минимальный объём
        '''
        return np.log(det(self.D))

    def calc_A(self):
        '''
        Критерий A - оптимальности. (A - average variance)
        Эллипсоид рассеивания с наименьшей суммой квадратов длин осей
        '''
        return np.trace(self.D)


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
    def Delta(self, x_j, x):
        N = float(self.N)
        return ((self.d(x) - self.d(x_j)) / N) - \
            ((self.d(x) * self.d(x_j) - self.d_2(x,x_j)**2)/(N**2))

    def d(self, x):
        return f_vector_T(x) @ self.D @ f_vector(x)

    def d_2(self, x, x_j):
        return f_vector_T(x) @ self.D @ f_vector(x_j)

#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
    def Fedorov_algorithm(self, do_visualisation = False):
        '''
        Алгоритм Фёдорова синтеза дискретного
        оптимального плана эксперимента
        '''
        alg_name = 'Алгоритм Фёдорова на шаге '
        path = 'pics/plan_Fedorov_{}_{}_{:.3f}_'.format(self.N, self.width, self.delta)

        self.generate_initial_guess()
        do_calc = True
        s = 0
        result = np.ndarray(self.max_iter)

        while do_calc == True and s < self.max_iter:
            self.build_matrix_M()
            self.build_matrix_D()
            D = self.calc_D()
            max_delta, i, j = self.find_two_points()
            self.x_plan[i] = self.x_grid[j]
            do_calc = not (max_delta < self.delta)
            result[s] = D
            if s % 10 == 0 and do_visualisation:
                self.draw_plan_on_s(s, alg_name, path)
            s += 1
            print('{}   det(D): {:.2f}   max_delta: {}   i: {}   j: {}'.format(s, D, max_delta, i, j))

        if do_calc == False:
            for i in range(s, self.max_iter):
                result[i] = result[s-1]

        self.draw_plan_on_s(s, alg_name, path)
        return result

    def Mitchell_algorithm(self, do_visualisation = False):
        '''
        Алгоритм Митчелла синтеза дискретного
        оптимального плана эксперимента
        '''
        alg_name = 'Алгоритм Митчелла на шаге '
        path = 'pics/plan_Mitchell_{}_{}_{:.3f}_'.format(self.N, self.width, self.delta)

        self.generate_initial_guess()
        do_calc = True
        s = 0
        result = np.ndarray(self.max_iter)

        while do_calc == True and s < self.max_iter:
            self.build_matrix_M()
            self.build_matrix_D()
            new_p, i = self.find_new_point()
            old_p, j = self.find_old_point()
            do_calc = norm(new_p - old_p) >= self.epsilon
            self.x_plan[j] = self.x_grid[i]
            result[s] = self.calc_D()
            if s % 5 == 0 and do_visualisation:
                self.draw_plan_on_s(s, alg_name, path)
            s += 1
            print('{}   det(D): {:.2f}   new_p: {}   old_p: {}'.format(s, self.calc_D(), new_p, old_p))

            if do_calc == False:
                for i in range(s, self.max_iter):
                    result[i] = result[s-1]
            
        self.draw_plan_on_s(s, alg_name, path)
        return result

    def gradient_algorithm(self, do_visualisation = False):
        '''
        Градиентный алгоритмы синтеза дискретного
        оптимального плана эксперимента
        '''
        alg_name = 'Градиентный алгоритм на шаге '
        path = 'pics/plan_grad_alg_{}_{}_{:.3f}_'.format(self.N, self.width, self.delta)
        
        # шаг 1
        # задаём начальное приближение
        self.generate_initial_guess()
        do_calc = True
        s = 0
        A_prev = 0.0
        A = 0.0
        result = np.ndarray(self.max_iter)

        while do_calc == True and s < self.max_iter:
            # шаг 2
            # вычисляем элементы вектора градиента 
            self.build_matrix_M_without_p()
            self.build_matrix_D()
            D = self.D

            grad_plan = np.ndarray(self.N)
            for i in range(self.N):
                x = self.x_plan[i]
                grad_plan[i] = f_vector_T(x) @ D @ D @ f_vector(x)
            
            grad_grid = np.ndarray(self.n)
            for i in range(self.n):
                x = self.x_grid[i]
                grad_grid[i] = f_vector_T(x) @ D @ D @ f_vector(x)

            i_counter = 0 # счётчик успешных замен
            do_replaces = True
            N_indicies = {i for i in range(self.N)}
            n_indicies = {i for i in range(self.n)}


            while do_replaces and i_counter < self.N:
                grad_grid = np.ndarray(self.n)
                D = self.D
                for i in range(self.n):
                    x = self.x_grid[i]
                    grad_grid[i] = f_vector_T(x) @ D @ D @ f_vector(x)
                
                # шаг 3
                # поиск x*
                max_j = 0
                max_val = grad_grid[max_j]
                for j in n_indicies:
                    if grad_grid[j] > max_val:
                        max_val = grad_grid[j]
                        max_j = j

                # шаг 4
                # поиск x**
                min_i = 0
                min_val = grad_plan[min_i]
                for i in N_indicies:
                    if grad_plan[i] < min_val:
                        min_val = grad_plan[i]
                        min_i = i

                # шаг 5
                # замена x** на x*
                self.x_plan[min_i] = self.x_grid[max_j]

                self.build_matrix_M_without_p()
                self.build_matrix_D()
                A = self.calc_A()

                # шаг 6
                # сравниваем функционал на 2-х итерациях
                if A > A_prev:
                    i_counter += 1
                    print(min_i, max_j)
                    N_indicies.remove(min_i)
                    n_indicies.remove(max_j)
                else:
                    do_replaces = False
                A_prev = A

            result[s] = A
            

            # if s % 10 == 0 and do_visualisation:
            self.draw_plan_on_s(s, alg_name, path)
            s += 1
            print('{}   det(M^-2): {:.2f}'.format(s, A))




            if do_calc == False:
                for i in range(s, self.max_iter):
                    result[i] = result[s-1]

        self.draw_plan_on_s(s, alg_name, path)
        return result


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
def perform_experiment(N, width, delta, method = 'Fedorov', do_visualisation = False):
    ''' 
    Отдельный эксперимент:
    N       - число узлов плана,
    width   - ширина сетки,
    delta   - точность,
    method  - Fedorov, Mitchell, gradient.
    '''
    l = Lab3(N, width, delta)
    algorithm = {
        'Fedorov': l.Fedorov_algorithm,
        'Mitchell': l.Mitchell_algorithm,
        'gradient': l.gradient_algorithm
    }
    algorithm = algorithm[method]
    return algorithm(do_visualisation)

def research_delta(method = 'Fedorov'):
    ''' Исследование скорости сходимости и устойчивости от delta '''
    print('Исследование скорости сходимости и устойчивости от delta')
    width = 21
    for N in [20, 40]:
        for delta in [0.001, 0.01, 0.1]:
            y = perform_experiment(N, width, delta, method, False)
            plt.plot(y, label=str(delta))    

        plt.title('D-оптимальность плана от delta')
        plt.legend(title='сетка: {}x{}\nN: {}'.format(width, width, N))
        plt.xticks(t)
        plt.xlabel('итерации')
        plt.ylabel(r'$\log(\left| M^{-1}(\varepsilon) \right|)$')
        plt.savefig('pics/research_delta_{}x{}_{}.png'.format(width, width, N), dpi=200)
        plt.clf()

def research_N(method = 'Fedorov'):
    ''' Исследование скорости сходимости и устойчивости от N '''
    print('Исследование скорости сходимости и устойчивости от N')
    delta = 0.01
    for width in [11, 21]:
        for N in [20, 25, 30, 35, 40]:
            y = perform_experiment(N, width, delta, method, False)
            plt.plot(y, label=str(N))    
        plt.title('Влияние числа узлов плана')
        plt.legend(title='сетка: {}x{}\ndelta: {:.3f}'.format(width, width, delta))
        plt.xticks(t)
        plt.xlabel('итерации')
        plt.ylabel(r'$\log(\left| M^{-1}(\varepsilon) \right|)$')
        plt.savefig('pics/research_N_{}x{}.png'.format(width, width), dpi=200)
        plt.clf()

def research_width(method = 'Fedorov'):
    ''' Исследование скорости сходимости и устойчивости от числа узлов сетки '''
    print('Исследование скорости сходимости и устойчивости от числа узлов сетки')
    delta = 0.001
    for width in [11, 21]:
        for N in [20, 40]:
            y = perform_experiment(N, width, delta, method, False)
            plt.plot(y, label='N: {}  n: {}'.format(N, width**2))    

    plt.title('Влияние числа узлов сетки')
    plt.legend(title='delta: {:.3f}'.format(delta))
    plt.xticks(t)
    plt.xlabel('итерации')
    plt.ylabel(r'$\log(\left| M^{-1}(\varepsilon) \right|)$')
    plt.savefig('pics/research_width.png', dpi=200)
    plt.clf()

def show_convergence_of_method(N, width, delta, method = 'Fedorov', do_visualisation = False):
    ''' 
    Отрисовка сходимости метода:
    N       - число узлов плана,
    width   - ширина сетки,
    delta   - точность,
    method  - Fedorov, Mitchell, gradient.
    '''
    print(N, width, delta)
    l = Lab3(N, width, delta)
    
    algorithm = {
        'Fedorov': l.Fedorov_algorithm,
        'Mitchell': l.Mitchell_algorithm,
        'gradient': l.gradient_algorithm
    }
    algorithm = algorithm[method]
    
    title = {
        'Fedorov': 'Сходимость метода Фёдорова',
        'Mitchell': 'Сходимость метода Митчелла',
        'gradient': 'Сходимость градиентного алгоритма'
    }
    title = title[method]

    if method == 'gradient':
        ylabel = r'$\log(\left| M^{-2}(\varepsilon) \right|)$'
    else:
        ylabel = r'$\log(\left| M^{-1}(\varepsilon) \right|)$'

    path = {
        'Fedorov': 'pics/convergence_Fedorov_{}_{}_{:.3f}.png'.format(N, width, delta),
        'Mitchell': 'pics/convergence_Mitchell_{}_{}_{:.3f}.png'.format(N, width, delta),
        'gradient': 'pics/convergence_grad_alg_{}_{}_{:.3f}.png'.format(N, width, delta)
    }
    path = path[method]

    y = algorithm(do_visualisation)
    plt.plot(y)
    plt.title(title)
    plt.text(24, 6, 'сетка: {}x{}\nN: {}\ndelta: {:.3f}'.format(width, width, N, delta))
    plt.xticks(t)
    plt.xlabel('итерации')
    plt.ylabel(ylabel)
    plt.savefig(path, dpi=200)
    plt.clf()


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
# method = 'Fedorov'
# method = 'Mitchell'
method = 'gradient'

# research_delta(method)
# research_N(method)
# research_width(method)
show_convergence_of_method(30, 21, 0.001, method, True)