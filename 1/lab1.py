import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from numpy.linalg import inv, det, eigvals, norm


pd.set_option('precision', 2)
n = 3
m = 3
points_count = 51

def f(theta, x):
    return theta[0] + theta[1]*x + theta[2]*x**2

def f_vector(x):
    return np.array([ [1], [x], [x**2] ])

def f_vector_T(x):
    return np.array([ 1, x, x**2 ])


#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
class Lab1():
    '''
    Класс для 1 лабораторной работы
    Для стандартизации все критерии ищют минимальное значение
    '''
    def __init__(self):
        ''' Выделение памяти под массивы '''
        self.x = np.ndarray(n)
        self.p = np.ndarray(n)
        self.M = np.ndarray((m, m))

    def read_plan_from_file(self, filename):
        ''' Считываем данные из файла '''
        data = np.loadtxt(filename)
        self.x = data[0]
        self.p = data[1]
        
    def create_plan(self, q):
        ''' Создание плана '''
        self.p = np.array([q, 1-2*q, q])
    
    def build_matrix_M(self):
        ''' Построение информационной матрицы M '''
        self.M = np.zeros((m,m))
        for i in range(n):
            self.M += self.p[i] * f_vector(self.x[i]) * f_vector_T(self.x[i])

    def build_matrix_D(self):
        ''' Построение дисперсионной матрицы D '''
        self.D = inv(self.M)

    def find_optimal_q(self):
        ''' Поиск оптимального значения q '''
        self.x_axis   = np.linspace(0, 0.5, points_count)
        self.D_from_E = np.ndarray(points_count)
        i = 0
        for q in self.x_axis:
            self.create_plan(q)
            self.build_matrix_M()
            # self.build_matrix_D() # т.к. матрица сингулярная
            self.D_from_E[i] = self.calc_M()
            i+=1
        
        i = np.where(self.D_from_E == max(self.D_from_E))
        print('Оптимальное значение параметра q: ' + str(self.x_axis[i][0]))

    def compare_plans(self, filename):
        ''' Сравнение планов различными критериями '''
        plan_files = [
            '1.txt',
            '2.txt',
            '3.txt',
            '4.txt'
            ]
        with open('report/matricies_M.tex', 'w') as f:
            f.write('\\begin{tabular}{|c|c|c|c|}\n')
            f.write('\hline\n')
            f.write('\tvariant 1 & variant 2 & variant 3 & variant 4 \\\ \n')
            f.write('\hline\n')
            s = ''
            for plan_file in plan_files:
                s += '&'
                self.read_plan_from_file('plans/' + plan_file)
                self.build_matrix_M()
                self.build_matrix_D()
                df = pd.DataFrame(data = self.M)
                s += df.to_latex(index=False, header = False, column_format='ccc')
            s = re.sub(r'\\[a-z]+rule\n', '', s)# удаляем \toprule и \bottomrule
            s = re.sub(r'\n&', ' &\n', s)       # переносим &
            f.write(s[1:-1] + ' \\\\ \n')
            f.write('\hline\n\end{tabular}')            

        with open('report/matricies_D.tex', 'w') as f:
            f.write('\\begin{tabular}{|c|c|c|c|}\n')
            f.write('\hline\n')
            f.write('\tvariant 1 & variant 2 & variant 3 & variant 4 \\\ \n')
            f.write('\hline\n')
            s = ''
            for plan_file in plan_files:
                s += '&'
                self.read_plan_from_file('plans/' + plan_file)
                self.build_matrix_M()
                self.build_matrix_D()
                df = pd.DataFrame(data = self.D)
                s += df.to_latex(index=False, header = False, column_format='ccc')
            s = re.sub(r'\\[a-z]+rule\n', '', s)# удаляем \toprule и \bottomrule
            s = re.sub(r'\n&', ' &\n', s)       # переносим &
            f.write(s[1:-1] + ' \\\\ \n')
            f.write('\hline\n\end{tabular}')



        criterion_files = [
            'D.txt',
            'A.txt',
            'E.txt',
            'Phi_2.txt',
            'Lambda.txt',
            'MV.txt',
            'G.txt'
            ]
        criterion_functions = {
            'D.txt': self.calc_D,
            'A.txt': self.calc_A,
            'E.txt': self.calc_E,
            'Phi_2.txt': self.calc_Phi_2,
            'Lambda.txt': self.calc_Lambda,
            'MV.txt': self.calc_MV,
            'G.txt': self.calc_G
        }

        i_vec = np.ndarray(4)
        cr_vec = np.ndarray(4)
        table = pd.DataFrame()

        # для каждого критерия
        for criterion_file in reversed(criterion_files):
            criterion_name = criterion_file[:-4]
            # рассмотрим 4 плана
            for plan_file in plan_files:
                self.read_plan_from_file('plans/' + plan_file)
                self.build_matrix_M()
                self.build_matrix_D()
                result = criterion_functions[criterion_file]()
                i = int(plan_file[:-4]) - 1
                i_vec[i] = i + 1
                cr_vec[i] = result
            i_vec = np.argsort(cr_vec) + 1
            cr_vec = np.sort(cr_vec)
            d = {(criterion_name, 'i'): i_vec,
                 (criterion_name, 'value'): cr_vec}
            df = pd.DataFrame(data = d)
            table = df.join(table, lsuffix='_caller', rsuffix='_other')
        with open(filename, 'w') as f:
            f.writelines(table.to_latex(index=False))    

    def draw_plot(self):
        ''' Отрисовка зависимости D(E) от q '''
        plt.title('D - оптимальность плана в зависимости от q')
        plt.plot(self.x_axis, self.D_from_E)
        plt.savefig('report/opt_q.png')

    def calc_M(self):
        ''' 
        Критерий D - оптимальности. (D - determinant)
        Эллипсоид рассеивания имеет минимальный объём
        '''
        return det(self.M)

    def calc_D(self):
        ''' 
        Критерий D - оптимальности. (D - determinant)
        Эллипсоид рассеивания имеет минимальный объём
        '''
        return det(self.D)

    def calc_A(self):
        '''
        Критерий A - оптимальности. (A - average variance)
        Эллипсоид рассеивания с наименьшей суммой квадратов длин осей
        '''
        return np.trace(self.D)

    def calc_E(self):
        '''
        Критерий E - оптимальности. (E - eigenvalue)
        Минимизация максимального собственного значения дисперсионной матрицы
        '''
        return np.max(eigvals(self.D))

    def calc_Phi_2(self):
        '''
        Критерий \Phi_2 - оптимальности. (\Phi_2 - функционал класса \Phi_p, p in (0, inf))
        Минимизация максимального собственного значения дисперсионной матрицы
        '''
        return np.sqrt(np.trace(self.D**2) / m)

    def calc_Lambda(self):
        '''
        Критерий \Lambda - оптимальности. (Lambda - собственное значение)
        Минимизация дисперсии собственных значений
        '''
        eig_vec = eigvals(self.D)
        avg_eig = np.mean(eig_vec)
        return np.sum((eig_vec[:]-avg_eig)**2)

    def calc_MV(self):
        '''
        Критерий MV - оптимальности. (MV - maximum variation)
        Минимизация максимального диагонального значения дисперсионной матрицы
        '''
        return max(np.diag(self.D))

    def calc_G(self):
        '''
        Критерий G - оптимальности. (G - general varience)
        Минимизация максимального значения общей дисперсии
        '''
        return np.max([f_vector_T(x) * self.D * f_vector(x) for x in self.x])



#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------
#-------------------------------------------------------------------------------

l1 = Lab1()
# выбор оптимального плана
l1.compare_plans('report/plans_comparison.tex')
# выбор оптимального значения q
l1.read_plan_from_file('plans/4.txt')
l1.find_optimal_q()
l1.draw_plot()
