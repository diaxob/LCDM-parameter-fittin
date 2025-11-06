import numpy as np
from collections import namedtuple

Result = namedtuple('Result', ('nfev', 'cost', 'gradnorm', 'x'))
Result.__doc__ = """Результаты оптимизации

Attributes
----------
nfev : int
    Полное число вызовов модельной функции
cost : 1-d array
    Значения функции потерь 0.5 sum(y - f)^2 на каждом итерационном шаге.
    В случае метода Гаусса—Ньютона длина массива равна nfev, в случае ЛМ-метода
    длина массива — менее nfev(шаг может уменьшаться);
gradnorm : float
    Норма градиента на финальном итерационном шаге
theta : 1-d array
    Финальное значение вектора, минимизирующего функцию потерь
y — это массив с измерениями,
f(*theta) — это функция от неивестных параметров, возвращающая значения, 
    рассчитанные в соответствии с моделью, в виде одномерного массива размера y.size,
r = y - f(*theta),
j(*theta) — это функция от неизвестных параметров, возвращающая якобиан 
    в виде двумерного массива (y.size, theta0.size),
theta0 — массив с начальными приближениями параметров,
k — положительное число меньше единицы, параметр метода,
    видимо, коэффициент длины шага,
tol — относительная ошибка, условие сходимости, параметр метода,
    критерий остановки,
lmbd0 - начальное значение параметра лямбда - множителя Лагранжа,
nu - мультипликатор для лямбды.
"""

def gn_mod(r, j, theta0, k=1, tol=1e-4):
    max_iter = 10000
    nfev = 0
    theta = np.array(theta0, dtype=float)
    cost = []

    while True:
        nfev += 1
        res = r(theta)
        cost.append(0.5 * np.dot(res, res))
        c = 0.5 * float(res @ res)

        # якобиан + колоночное масштабирование
        J = j(theta)
        scale = np.linalg.norm(J, axis=0) + 1e-12
        Jn = J / scale

        grad = Jn.T @ res
        gradnorm = np.linalg.norm(grad)

        A = Jn.T @ Jn

        # маленькая стабилизация, для наших данных имеет место большая обусловленность
        A += 1e-6 * np.eye(A.shape[0])

        try:
            delta_scaled = np.linalg.solve(A, grad)
            # возвращаем масштаб
            delta = delta_scaled / scale    
        except np.linalg.LinAlgError:
            break

        # первоначальная попытка побороть огромные выходные данные
        # backtracking line search
        # step = float(k)
        # for _ in range(20):
        #     theta_try = theta - step * delta
        #     res_try = r(theta_try)
        #     c_try = 0.5 * float(res_try @ res_try)
        #     if c_try < c:
        #         theta = theta_try
        #         c = c_try
        #         break
        #     step *= 0.5
        # else:
        #     break

        theta_new = theta- k * delta

        if len(cost) > 1 and abs(cost[-1] - cost[-2]) < tol * (cost[-2] + 1e-12):
            break
        
        theta = theta_new

        if nfev >= max_iter:
            break

    return Result(nfev, np.asarray(cost), gradnorm, theta)

def lm_mod(r, j, theta0, lmbd0 = 1e-2, nu = 2, tol = 1e-4):
    max_iter = 10000
    nfev = 0
    theta = np.array(theta0, dtype = float)
    cost = []

    lmbd = lmbd0

    while True:
        nfev += 1
        res = r(theta)
        cost.append(0.5 * np.dot(res, res))

        J = j(theta)
        grad = J.T @ res
        gradnorm = np.linalg.norm(grad)

        A = J.T @ J
        N = A.shape[0]
        A_lm = A + lmbd * np.eye(N)

        try:
            delta = np.linalg.solve(A_lm, grad)
        except np.linalg.LinAlgError:
            lmbd *= nu
            if nfev >= max_iter:
                break
            continue

        if  np.linalg.norm(delta) < tol or (len(cost) > 1 and 
                                            abs(cost[-1] - cost[-2]) < tol * (cost[-2] + 1e-12)):
            break
            
        theta_new = theta - delta
        res_new = r(theta_new)
        cost_new = 0.5 * np.dot(res_new, res_new)

        # Попытка учета условий на лямбду
        if cost_new < cost[-1]:
            theta = theta_new   # Улучшение при уменьшении лямбда - уменьшаем
            lmbd /= nu  
        elif cost_new >= cost[-1] and lmbd != lmbd0:
            # Старая лямбда была лучше - не меняем лямбду и theta
            pass
        else:
            # Увеличиваем лямбду, пока не станет лучше
            w = 0
            while cost_new >= cost[-1] and w < 10:
                lmbd *= nu
                A_lm = A + lmbd * np.eye(len(theta))
                try:
                    delta = np.linalg.solve(A_lm, grad)
                except np.linalg.LinAlgError:
                    w += 1
                    continue
                theta_new = theta + delta
                res_new = r(theta_new)
                cost_new = 0.5 * np.dot(res_new, res_new)
                w += 1

            if cost_new < cost[-1]:
                theta = theta_new

        if nfev >= max_iter:
            break
    
    return Result(nfev, np.asarray(cost), gradnorm, theta)

if __name__ == "__main__":
    pass