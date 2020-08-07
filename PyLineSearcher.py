import numpy as np
import math
import random
import time


class CGSSearch(object):
    def __init__(self, costfun, x=0, direction=1, eps=1e-2, delta=1e-1):
        self.costfun = costfun
        self.x = x
        self.d = direction
        self.eps = eps
        self.delta = delta

    def set_costfun(self, costfun):
        self.costfun = costfun

    def set_x(self, x):
        self.x = x

    def set_d(self, d):
        self.d = d

    def set_eps(self, eps):
        self.eps = eps

    def set_delta(self, delta):
        self.delta = delta

    def __Phase1(self):
        alpha_g = 0
        alpha_g_list = list()
        f_alpha_g = list()
        self.alpha_lower = 0
        self.alpha_upper = 0
        self.interval = 0

        g = 0
        while True:
            alpha_g = alpha_g + self.delta * (1.618)**g
            alpha_g_list.append(alpha_g)
            f_alpha_g.append(self.costfun(self.x + alpha_g * self.d))
            if g > 1:
                if f_alpha_g[g - 1] < f_alpha_g[g] and f_alpha_g[g - 1] < f_alpha_g[g - 2]:
                    self.alpha_lower = alpha_g_list[g - 2]
                    self.alpha_upper = alpha_g_list[g]
                    self.interval = self.alpha_upper - self.alpha_lower
                    # self.interval = 2.618 * (1.618 ** (g - 1)) * self.delta
                    break
            g += 1

    def __Phase2(self):
        # ========== TestLineFun1's theoretical value test ========== #

        # self.alpha_lower = 0
        # self.alpha_upper = 2
        # self.interval = self.alpha_upper - self.alpha_lower

        # ========== TestLineFun1's theoretical value test ========== #

        # p_plus, p_minus = (3 + math.sqrt(5)) / 2, (3 - math.sqrt(5)) / 2
        p = (3 - math.sqrt(5)) / 2
        # p = 0.382

        # N is the worst case, NOT necessary
        reductionFactor = self.eps / self.interval
        N = 0
        while True:
            if 0.61803**N <= abs(reductionFactor):
                break
            N += 1

        alpha_a = 0
        alpha_b = 0
        f_alpha_a = 0
        f_alpha_b = 0
        iteration = 0

        while True:
            if abs(self.interval) < self.eps or iteration == N:
                x = (self.alpha_lower + self.alpha_upper) / 2
                return x, iteration
            else:
                alpha_a = self.alpha_lower + p * self.interval
                alpha_b = self.alpha_upper - p * self.interval
                f_alpha_a = self.costfun(self.x + alpha_a * self.d)
                f_alpha_b = self.costfun(self.x + alpha_b * self.d)

                if f_alpha_a < f_alpha_b:  # step3
                    self.alpha_upper = alpha_b

                elif f_alpha_a > f_alpha_b:  # step4
                    self.alpha_lower = alpha_a

                elif f_alpha_a == f_alpha_b:  # step5
                    self.alpha_lower = alpha_a
                    self.alpha_upper = alpha_b

            self.interval = np.subtract(self.alpha_upper, self.alpha_lower)
            iteration += 1

        # step = 1
        # while True:
        #     if abs(self.interval) < self.eps:  # or iteration == N:
        #         return (self.alpha_lower + self.alpha_upper) / 2, iteration
        #     else:
        #         if step == 1:
        #             alpha_a = self.alpha_lower + p * self.interval
        #             alpha_b = self.alpha_upper - p * self.interval
        #             f_alpha_a = self.costfun(self.x + alpha_a * self.d)
        #             f_alpha_b = self.costfun(self.x + alpha_b * self.d)
        #             step = 2

        #         if step == 2:
        #             if f_alpha_a < f_alpha_b:  # step3
        #                 self.alpha_lower = self.alpha_lower
        #                 self.alpha_upper = alpha_b
        #                 alpha_b = alpha_a

        #                 alpha_a = self.alpha_lower + p * (self.alpha_upper - self.alpha_lower)
        #                 f_alpha_b = f_alpha_a
        #                 f_alpha_a = self.costfun(self.x + alpha_a * self.d)

        #                 self.interval = np.subtract(self.alpha_upper, self.alpha_lower)

        #             elif f_alpha_a > f_alpha_b:  # step4
        #                 self.alpha_lower = alpha_a
        #                 self.alpha_upper = self.alpha_upper
        #                 alpha_a = alpha_b

        #                 alpha_b = self.alpha_lower + (1 - p) * (self.alpha_upper - self.alpha_lower)
        #                 f_alpha_a = f_alpha_b
        #                 f_alpha_b = self.costfun(self.x + alpha_b * self.d)

        #                 self.interval = np.subtract(self.alpha_upper, self.alpha_lower)

        #             elif f_alpha_a == f_alpha_b:  # step5
        #                 self.alpha_lower = alpha_a
        #                 self.alpha_upper = alpha_b
        #                 step = 1

        #     iteration += 1

    def RunSearch(self):
        self.__Phase1()
        return self.__Phase2()


class CFiSearch(object):
    def __init__(self, costfun, x=0, direction=1, eps=1e-2, delta=1e-1):
        self.costfun = costfun
        self.x = x
        self.d = direction
        self.eps = eps
        self.delta = delta

    def set_costfun(self, costfun):
        self.costfun = costfun

    def set_x(self, x):
        self.x = x

    def set_d(self, d):
        self.d = d

    def set_eps(self, eps):
        self.eps = eps

    def set_delta(self, delta):
        self.delta = delta

    def __Phase1(self):
        alpha_g = 0
        alpha_g_list = list()
        f_alpha_g = list()
        self.alpha_lower = 0
        self.alpha_upper = 0
        self.interval = 0

        g = 0
        while True:
            alpha_g = alpha_g + self.delta * (1.618)**g
            alpha_g_list.append(alpha_g)
            f_alpha_g.append(self.costfun(self.x + alpha_g * self.d))
            if g > 1:
                if f_alpha_g[g - 1] < f_alpha_g[g] and f_alpha_g[g - 1] < f_alpha_g[g - 2]:
                    self.alpha_lower = alpha_g_list[g - 2]
                    self.alpha_upper = alpha_g_list[g]
                    self.interval = self.alpha_upper - self.alpha_lower
                    # self.interval = 2.618 * (1.618 ** (g - 1)) * self.delta
                    break
            g += 1

    def __Phase2(self):
        # ========== TestLineFun1's theoretical value test ========== #

        # self.alpha_lower = 0
        # self.alpha_upper = 2
        # self.interval = self.alpha_upper - self.alpha_lower

        # ========== TestLineFun1's theoretical value test ========== #

        # FibSeq = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584,
        #           4181, 6765, 10946, 17711, 28657, 46368, 75025, 121393, 196418, 317811, 514229]

        random_eps = random.uniform(0, self.eps)

        reductionFactor = (1 + 2 * random_eps) / (self.eps / self.interval)

        FibSeq, F_pos = [1, 1], 2

        while True:
            FibNum_pre = FibSeq[F_pos - 1]
            FibNum_cur = FibSeq[F_pos - 2]
            FibNum_nxt = FibNum_pre + FibNum_cur
            FibSeq.append(FibNum_nxt)
            F_pos += 1
            if FibSeq[F_pos - 1] > reductionFactor:
                break

        N = min(FibSeq, key=lambda x: abs(x - reductionFactor))
        if N < reductionFactor:
            N = FibSeq.index(N)
        else:
            N = FibSeq.index(N) - 1

        iteration = 0
        p = 0
        alpha_a = 0
        alpha_b = 0
        f_alpha_a = 0
        f_alpha_b = 0

        while True:
            if abs(self.interval) < self.eps or iteration > N:
                x = (self.alpha_lower + self.alpha_upper) / 2
                return x, iteration
            else:
                if iteration == N:
                    p = 0.5 - random_eps
                else:
                    p = 1 - (FibSeq[N - iteration] / FibSeq[N + 1 - iteration])

                alpha_a = self.alpha_lower + p * self.interval
                alpha_b = self.alpha_upper - p * self.interval

                f_alpha_a = self.costfun(self.x + alpha_a * self.d)
                f_alpha_b = self.costfun(self.x + alpha_b * self.d)

                if f_alpha_a < f_alpha_b:  # step3
                    self.alpha_upper = alpha_b

                elif f_alpha_a > f_alpha_b:  # step4
                    self.alpha_lower = alpha_a

                elif f_alpha_a == f_alpha_b:  # step5
                    self.alpha_lower = alpha_a
                    self.alpha_upper = alpha_b

                self.interval = self.alpha_upper - self.alpha_lower
                iteration += 1

    def RunSearch(self):
        self.__Phase1()
        return self.__Phase2()


def TestLineFun1(x):
    return x**4 - 14 * (x**3) + 60 * (x**2) - 70 * x
    # x* = 0.780885825794867, f(x*) = -24.369601567258060


def TestLineFun2(x):
    return (0.65 - 0.75 / (1 + x**2)) - 0.65 * x * np.arctan2(1, x)
    # x* = 0.4808678353168805, f(x*) = -0.310020501948328


def TestLineFun3(x):
    return -(108 * x - x**3) / 4
    # x* = 6, f(x*) = -108


def Fibonacci(x):
    if x in [0, 1]:
        return 1
    return Fibonacci(x - 1) + Fibonacci(x - 2)


if __name__ == "__main__":
    # input_x = random.uniform(-0.1, 0.1)
    input_x = 0
    input_direction = 1
    input_eps = 1e-4
    input_delta = 1e-1
    input_func = [TestLineFun1, TestLineFun2, TestLineFun3]
    test_CGSSearch = CGSSearch(input_func[0], input_x, input_direction, input_eps, input_delta)
    test_CFiSearch = CFiSearch(input_func[0], input_x, input_direction, input_eps, input_delta)
    print("CGSSearch================================", "CFiSearch=============================")

    start_total = time.time()

    for i in range(0, len(input_func)):

        test_CGSSearch.set_costfun(input_func[i])
        test_CFiSearch.set_costfun(input_func[i])

        x_GSS, i_GSS = test_CGSSearch.RunSearch()
        fx_GSS = input_func[i](x_GSS)

        x_FiS, i_FiS = test_CFiSearch.RunSearch()
        fx_FiS = input_func[i](x_FiS)

        print("x* = %7.4f, f(x*) = %9.4f, i = %d   " % (x_GSS, fx_GSS, i_GSS),
              "x* = %7.4f, f(x*) = %9.4f, i = %d   " % (x_FiS, fx_FiS, i_FiS))

    end_total = time.time()
    print("--- total time: %f seconds ---" % (end_total - start_total))

    # TestLineFun1: x* = 0.780885825794867 , f(x*) = -24.369601567258060
    # TestLineFun2: x* = 0.4808678353168805, f(x*) = -0.310020501948328
    # TestLineFun3: x* = 6                 , f(x*) = -108
