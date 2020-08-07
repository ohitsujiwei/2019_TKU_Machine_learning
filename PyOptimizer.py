import numpy as np
import time
import random
import PyLineSearcher


class CForwardDiff(object):
    def __init__(self, costfun, x, dim, eps=1e-5, percent=1e-5):
        self.costfun = costfun
        self.x = x
        self.dim = dim
        self.eps = eps
        self.percent = percent

    def set_costfun(self, costfun):
        self.costfun = costfun

    def set_x(self, x):
        self.x = x

    def set_dim(self, dim):
        self.dim = dim

    def set_eps(self, eps):
        self.eps = eps

    def set_percent(self, percent):
        self.percent = percent

    def GetGrad(self):
        diff_fx_list = list()
        res = list()
        self.x = np.add(self.x, self.eps)

        for i in range(0, self.dim):
            for j in range(0, self.dim):
                if j == i:
                    res.append(self.x[j] + self.x[j] * self.percent)
                else:
                    res.append(self.x[j])
            diff_fx_list.append((self.costfun(res[i * self.dim:(1 + i) * self.dim]) - self.costfun(self.x)) / (self.x[i] * self.percent))

        return diff_fx_list


class CBackwardDiff(object):
    def __init__(self, costfun, x, dim, eps=1e-5, percent=1e-5):
        self.costfun = costfun
        self.x = x
        self.dim = dim
        self.eps = eps
        self.percent = percent

    def set_costfun(self, costfun):
        self.costfun = costfun

    def set_x(self, x):
        self.x = x

    def set_dim(self, dim):
        self.dim = dim

    def set_eps(self, eps):
        self.eps = eps

    def set_percent(self, percent):
        self.percent = percent

    def GetGrad(self):
        diff_fx_list = list()
        res = list()
        self.x = np.add(self.x, self.eps)

        for i in range(0, self.dim):
            for j in range(0, self.dim):
                if j == i:
                    res.append(self.x[j] - self.x[j] * self.percent)
                else:
                    res.append(self.x[j])
            diff_fx_list.append((self.costfun(self.x) - self.costfun(res[i * self.dim:(1 + i) * self.dim])) / (self.x[i] * self.percent))

        return diff_fx_list


class CCentralDiff(object):
    def __init__(self, costfun, x, dim, eps=1e-5, percent=1e-5):
        self.costfun = costfun
        self.x = x
        self.dim = dim
        self.eps = eps
        self.percent = percent

    def set_costfun(self, costfun):
        self.costfun = costfun

    def set_x(self, x):
        self.x = x

    def set_dim(self, dim):
        self.dim = dim

    def set_eps(self, eps):
        self.eps = eps

    def set_percent(self, percent):
        self.percent = percent

    def GetGrad(self):
        diff_fx_list = list()
        res1 = list()
        res2 = list()
        self.x = np.add(self.x, self.eps)

        for i in range(0, self.dim):
            for j in range(0, self.dim):
                if j == i:
                    res1.append(self.x[j] + self.x[j] * self.percent / 2)
                    res2.append(self.x[j] - self.x[j] * self.percent / 2)
                else:
                    res1.append(self.x[j])
                    res2.append(self.x[j])

            a = self.costfun(res1[i * self.dim:(1 + i) * self.dim])
            b = self.costfun(res2[i * self.dim:(1 + i) * self.dim])
            diff_fx_list.append((a - b) / (self.x[i] * self.percent))

        return diff_fx_list


class CGradDecent(object):
    def __init__(self, costfun, x0, dim, Gradient='Backward', LineSearch='FiS', MinNorm=1e-3, MaxIter=1e+3):
        self.costfun = costfun
        self.x0 = x0
        self.dim = dim
        self.MinNorm = MinNorm
        self.MaxIter = MaxIter

        if Gradient == "Forward":
            self.Gradient = CForwardDiff(self.costfun, self.x0, self.dim)
        elif Gradient == "Backward":
            self.Gradient = CBackwardDiff(self.costfun, self.x0, self.dim)
        elif Gradient == "Central":
            self.Gradient = CCentralDiff(self.costfun, self.x0, self.dim)
        else:
            print("Gradient select Error")
            return

        if LineSearch == "FiS":
            self.LineSearch = PyLineSearcher.CFiSearch(self.costfun, self.x0)
        elif LineSearch == "GSS":
            self.LineSearch = PyLineSearcher.CGSSearch(self.costfun, self.x0)
        else:
            print("LineSearch select Error")
            return

    def set_costfun(self, costfun):
        self.costfun = costfun
        self.Gradient.set_costfun(costfun)
        self.LineSearch.set_costfun(costfun)

    def set_x0(self, x0):
        self.x0 = x0
        self.Gradient.set_x(x0)
        self.LineSearch.set_x(x0)

    def set_dim(self, dim):
        self.dim = dim
        self.Gradient.set_dim(dim)

    def set_MinNorm(self, MinNorm):
        self.MinNorm = MinNorm

    def set_Maxlter(self, MaxIter):
        self.MaxIter = MaxIter

    def RunOptimize(self):
        iteration = 0
        iteration_LS = 0
        x = self.x0
        alpha = 0
        gradient = 0
        norm = 0
        error_list = list()

        while True:
            gradient = self.Gradient.GetGrad()
            norm = np.linalg.norm(gradient)
            error_list.append(self.costfun(x))
            if norm < self.MinNorm or iteration > self.MaxIter:
                return x, iteration_LS, iteration, error_list
            else:
                direction = np.multiply(gradient, -1)
                check = np.dot(gradient, direction)
                self.LineSearch.set_d(direction)
                alpha, i_LS = self.LineSearch.RunSearch()
                x = x + alpha * direction
                self.Gradient.set_x(x)
                self.LineSearch.set_x(x)
                iteration += 1
                iteration_LS += i_LS


def Test2VarFun0(x):
    return 3 * x[0] ** 2 + 2 * x[0] * x[1] + 2 * x[1] ** 2 + 7
    # x* = [0, 0], f(x*) = 7;


def Test2VarFun1(x):
    return (x[0] - x[1] + 2 * (x[0]**2) + 2 * x[0] * x[1] + x[1]**2)
    # x* = [-1, 1.5], f(x*) = -1.25;


def Test2VarFun2(x):
    return 0.5 * (100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2)
    # x* = [1, 1], f(x*) = 0;


def Test2VarFun3(x):
    return -x[0] * x[1] * np.exp(-x[0]**2 - x[1]**2)
    # x* = [0.7071, 0.7071] or x* = [-0.7071, -0.7071], f(x*) = -0.1839;


def Test2VarFun4(x):
    return -3 * x[1] / (x[0]**2 + x[1]**2 + 1)
    # x* = [0, 1], f(x*) = -1.5;


def PowellFun(x):
    f1 = x[0] + 10 * x[1]
    f2 = np.sqrt(5.0) * (x[2] - x[3])
    f3 = (x[1] - 2 * x[2])**2
    f4 = np.sqrt(10.0) * ((x[0] - x[3])**2)
    return np.sqrt(f1 * f1 + f2 * f2 + f3 * f3 + f4 * f4)
    # x* = [0, 0, 0, 0], f(x*) = 0;


def testFun(x):
    return x[0]**2 + x[1]


if __name__ == "__main__":
    # test_x = [2, 1]
    # testClass = CForwardDiff(testFun, test_x, len(test_x), percent=1e-2)
    # x, res = testClass.GetGrad()
    # print("[", ', '.join('{:4.2f}'.format(f) for f in res), "], ", "[", ', '.join('{:4.2f}'.format(f) for f in x), "]")
    # del testClass
    # testClass = CBackwardDiff(testFun, test_x, len(test_x), percent=1e-2)
    # x, res = testClass.GetGrad()
    # print("[", ', '.join('{:4.2f}'.format(f) for f in res), "], ", "[", ', '.join('{:4.2f}'.format(f) for f in x), "]")
    # del testClass
    # testClass = CCentralDiff(testFun, test_x, len(test_x), percent=1e-2)
    # x, res1, res2 = testClass.GetGrad()
    # print("[", ', '.join('{:5.3f}'.format(f) for f in res1), "], ",
    #       "[", ', '.join('{:5.3f}'.format(f) for f in res2), "], ",
    #       "[", ', '.join('{:4.2f}'.format(f) for f in x), "]")

    input_x, number_x = list(), 2
    for i in range(0, number_x):
        input_x.append(random.uniform(-0.1, 0.1))
    input_func = [Test2VarFun0, Test2VarFun1, Test2VarFun2, Test2VarFun3, Test2VarFun4]
    input_Diff = "Central"
    input_LS = "FiS"
    input_MinNorm = 1e-3
    input_MaxIter = 1e+3
    test_CGradDecent = CGradDecent(input_func[0], input_x, number_x, input_Diff, input_LS, input_MinNorm, input_MaxIter)
    test_CGradDecent.LineSearch.set_delta(1e-6)

    start_total = time.time()

    for i in range(0, len(input_func)):
        test_CGradDecent.set_costfun(input_func[i])
        x, iter_LS, iter_K, errors = test_CGradDecent.RunOptimize()
        fx = input_func[i](x)
        print("x = [", ', '.join('{:7.4f}'.format(f) for f in x), "],",
              "fx = %7.4f," % fx,
              "iter = %d, search = %d" % (iter_LS, iter_K))

    input_x, number_x = list(), 4
    for i in range(0, number_x):
        input_x.append(1)
    test_CGradDecent.set_x0(input_x)
    test_CGradDecent.set_dim(number_x)
    test_CGradDecent.set_costfun(PowellFun)
    x, iter_LS, iter_K, errors = test_CGradDecent.RunOptimize()
    fx = PowellFun(x)
    print("x = [", ', '.join('{:7.4f}'.format(f) for f in x), "],",
          "fx = %7.4f," % fx,
          "iter = %d, search = %d" % (iter_LS, iter_K))

    end_total = time.time()
    print("--- total time: %f seconds ---" % (end_total - start_total))

    # O Test2VarFun0: x* = [0, 0]                                     , f(x*) = 7;
    # O Test2VarFun1: x* = [-1, 1.5]                                  , f(x*) = -1.25;
    # X Test2VarFun2: x* = [1, 1]                                     , f(x*) = 0;
    # O Test2VarFun3: x* = [0.7071, 0.7071] or x* = [-0.7071, -0.7071], f(x*) = -0.1839;
    # O Test2VarFun4: x* = [0, 1]                                     , f(x*) = -1.5;
    # X PowellFun   : x* = [0, 0, 0, 0]                               , f(x*) = 0;

    input_x, number_x = list(), 2
    for i in range(0, number_x):
        input_x.append(random.uniform(-0.1, 0.1))
    input_func = [Test2VarFun0, Test2VarFun1, Test2VarFun2, Test2VarFun3, Test2VarFun4]
    input_Diff = "Central"
    input_LS = "FiS"
    input_MinNorm = 1e-3
    input_MaxIter = 1e+3
    test_CGradDecent = CGradDecent(input_func[0], input_x, number_x, input_Diff, input_LS, input_MinNorm, input_MaxIter)
    test_CGradDecent.LineSearch.set_delta(1e-6)
    for i in range(0, len(input_func)):
        test_CGradDecent.set_costfun(input_func[i])
        x, iter_LS, iter_K, errors = test_CGradDecent.RunOptimize()
        fx = input_func[i](x)
    input_x, number_x = list(), 4
    for i in range(0, number_x):
        input_x.append(1)
    test_CGradDecent.set_x0(input_x)
    test_CGradDecent.set_dim(number_x)
    test_CGradDecent.set_costfun(PowellFun)
    x, iter_LS, iter_K, errors = test_CGradDecent.RunOptimize()
    fx = PowellFun(x)
