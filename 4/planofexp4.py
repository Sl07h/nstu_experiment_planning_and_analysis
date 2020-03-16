import numpy as np
import random
import matplotlib.pyplot as plt

#Функция Ф, которая как производная от \Eta.
def function(x):
    return np.array([
        [x[0] ** theta[1] * x[1] ** theta[2]],
        [theta[0] * theta[1] * x[0] ** (theta[1] - 1) * x[1] ** theta[2]],
        [theta[0] * theta[2] * x[0] ** theta[1] * x[1] ** (theta[2] - 1)]
    ])

#Построение массива откликов с зашумлением
def makeY(plan, p = 0.2):
    Y = list(map(lambda x: theta[0] * x[0]**theta[1] * x[1]**theta[2], plan))
    Y = list(map(lambda y: y + random.normalvariate(0, p * y), Y))
    return Y

#Построение сдучайного плана экспримента на сетке
def makePlan(grid = np.linspace(1, 10, 1001), m = 9):
    return list(map(lambda x: [random.choice(grid), random.choice(grid)], range(m)))


#Построение матрицы Икс по плану
def makeX(plan):
    return np.array(list(map(lambda x: [1.0, np.math.log(x[0]), np.math.log(x[1])], plan)))

N = 15
grid = np.linspace(0.1, 10, 100)
plan = makePlan(grid = grid)
theta = [0.2, 0.4, 0.4]
Y = makeY(plan)
Y = np.log(Y)
X = makeX(plan)

#функция вычисления информационной матрицы
def InfMatrix(plan):
  m = len(function(plan[0]))
  M = np.zeros((m,m))
  for j in range(m): 
    M += (1 / N) * function(plan[j]) * function(plan[j]).T
  return M

#функция вычисления дисперсионной матрицы
def DispMatrix(a):
  return np.linalg.inv(a)

M = InfMatrix(plan)
D = DispMatrix(M)

#дисперсия модели
def DisM(x, M):
    return np.dot(np.dot(function(x).T, DispMatrix(M)), function(x))

def DisM_(x,x_j,M):
    return np.dot(np.dot(function(x).T, DispMatrix(M)), function(x_j))

def Delta(x,x_j,M):
    return ((1/N) * (DisM(x,M) - DisM(x_j,M))) - ((1/(N**2)) * (DisM(x,M) * DisM(x_j,M) - DisM_(x,x_j,M)**2))

#нахождения максимального значения дельты на для одной точки из плана, но для всейй сетки.
def findMaxforOneX(x, M):
    maxdot = [grid[0], grid[0]]
    maxvalue = Delta(maxdot,x, M)
    for x1 in grid:
        for x2 in grid:
            value = Delta([x1, x2], x, M)
            if value > maxvalue:
                maxvalue = value
                maxdot = [x1, x2]
    return [maxvalue, maxdot]

#функция нахождения максимумов для всех точек плана и выбора из него наибольшего
def findMaxforAll(X, M):
    listofmax = [findMaxforOneX(x,M) for x in X]
    return [*max(listofmax),listofmax.index(max(listofmax))]




#Построение оптимального плана эксперимента по некоторому плану plan
def makeOptimalPlan(plan):
    eps = 0.0001
    iteration = 0
    while True:
        M = InfMatrix(plan)
        print("det", np.linalg.det(M))
        delta = findMaxforAll(plan, M)
        if delta[0] > eps:
            plan[delta[2]] = delta[1]
        else:
            break
        iteration += 1
    return plan

plan = makePlan(grid = grid, m = N)
first_plan = plan
plt.scatter([first_plan[i][0] for i in range(len(first_plan))],[first_plan[i][1] for i in range(len(first_plan))],)
plt.show()
optim_plan = makeOptimalPlan(plan)

plt.clf()

plt.scatter([optim_plan[i][0] for i in range(len(optim_plan))],[optim_plan[i][1] for i in range(len(optim_plan))],)
plt.show()

def MNK(x,y):
    exp_theta =  np.dot(np.dot(np.linalg.inv(np.dot(x.T,x)),x.T),y)
    exp_theta[0] = np.math.e ** exp_theta[0]
    return exp_theta

def RSS(x,y,theta_):
    y_exp = np.dot(x,theta_)
    return np.dot(y - y_exp, y - y_exp)

def Experiment(plan):
    rss = 0
    dif_norma = 0
    for i in range(100):
        X = makeX(plan) 
        Y = makeY(plan)
        Y = np.log(Y)
        exp_theta = MNK(X,Y)
        rss += RSS(X,Y,exp_theta)
        dif_norma += np.dot(exp_theta - theta,exp_theta - theta)
    rss /= 100
    dif_norma /= 100
    return rss, dif_norma

opt_rss, opt_norma = Experiment(optim_plan)
print('Оптимальный план')
print('RSS: ', opt_rss)
print('отклонение нормы оценок: ', opt_norma)

rand_rss, rand_norma = Experiment(first_plan)
print('Случайный план')
print('RSS: ', rand_rss)
print('отклонение нормы оценок: ', rand_norma)

