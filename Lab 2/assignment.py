import numpy as np, random, math
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.widgets import *


class Kernel:
    def linear(xi, xj):
        return np.dot(xi, xj)

    def polynomial(xi, xj, p=2):
        return pow((np.dot(xi,xj) + 1), p) 

    def rbf(xi, xj, a=2):
        return np.exp(-((np.linalg.norm(xi-xj)**2))/(2*a**2))

def zero_fun(vec_a):
    toRtn = np.dot(vec_a, targets)
    #print("DEBUG Zerofun %f" % (toRtn))
    return toRtn

def save_matrix(targets, kern, inputs):
    matrix_p = np.empty((N,N))
    for i in range(N):
        for j in range(N):
            matrix_p[i][j] = targets[i]*targets[j]*kern(inputs[i], inputs[j])
    return matrix_p

def objective(a):
    global P
    val = 0
    # np.sum(np.dot(a, some_matrix))
    for i in range(len(a)):
        for j in range(len(a)):
            val += a[i]*a[j]*P[i][j]
    val = val * 0.5

    return val - np.sum(a)

def ind(s, ext_a, ker, b):
    to_rtn = 0
    for i in ext_a:
        to_rtn += i['a']*i['t']*ker(s,i['x'])

    to_rtn = to_rtn - b
    return to_rtn

def perform_b(ext_alpha, ker):
    b=0
    #print(len(ext_alpha))
    for s in ext_alpha:
        #Check that point is on the margin
        if C is None or (s['a']>0 and s['a']<C):
            print("DEBUG: ENTRATO in IF")
            for i in ext_alpha:
                b += i['a']*i['t']*ker(s['x'], i['x'])
            b = b - s['t']
            return b

# Create training set
var=0.5
center=[1.5, 0.5]
classA = np.concatenate(
    (np.random.randn(10, 2) * var + center,
     np.random.randn(10, 2) * var + [-center[0], center[1]]))
classB = np.random.randn(20, 2) * var + [0.0, -0.5]
inputs = np.concatenate((classA, classB))
targets = np.concatenate((np.ones(classA.shape[0]), -np.ones(classB.shape[0])))

# Computer p matrix

N = inputs.shape[0]
start = np.zeros(N)
C = 1

permute = list(range(N))
random.shuffle(permute)
inputs = inputs[permute, :]
targets = targets[permute]

fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.2)
nz = 0
ker = Kernel.linear
num_nz = 0

def compute():
    global P, num_nz
    bounds= [(0, C) for b in range(N)]
    constraint={'type':'eq', 'fun':zero_fun}

    P = save_matrix(targets, ker, inputs)

    ret = minimize(objective, start, bounds=bounds, constraints=constraint)
    alpha = ret['x']

    non_zero = list(map(lambda x: {'a':alpha[x], 't':targets[x], 'x':inputs[x]}, [i for i in range(len(alpha)) if 10 ** -5 < alpha[i] < C]))
    # print([non_zero[0][2] for i in range(len(non_zero))])
    # print(non_zero[:2])

    num_nz = len(non_zero)
    ax.clear()
    # for nz in non_zero:
    b = perform_b(non_zero, ker)
    print('b', b)
    #ind = indicator(non_zero[nz][2], non_zero, ker, b)

    # b = np.dot(non_zero[:0], np.dot(non_zero[:1], ker([non_zero[0][2] for i in range(len(non_zero))], non_zero[:2])))
    # plotting

    xgrid = np.linspace(-5, 5)
    ygrid = np.linspace(-4, 4)
    grid = np.array([[ind((x, y), non_zero, ker, b) for x in xgrid] for y in ygrid])

    ax.contour(xgrid, ygrid, grid, (-1.0, 0.0, 1.0), colors=('red', 'black', 'blue'), linewidths=(1, 3, 1))
    #ax.scatter(non_zero[nz][2][0], non_zero[nz][2][1])

    ax.plot([p[0] for p in classA],
            [p[1] for p in classA],
            'b.')
    ax.plot([p[0] for p in classB],
            [p[1] for p in classB],
            'r.')

    ax.axis('equal')

    # fig.savefig('svmplot.pdf')
    plt.show()
        # break

gui_objects = []

def changeC(cc):
    global C
    C = cc
    compute()

def changeNZ(nzp):
    global nz
    nz = nzp
    compute()

def kernel(val):
    global ker
    if val == 'linear':
        ker = Kernel.linear
    elif val == 'polynomial':
        ker = Kernel.polynomial
    else:
        ker = Kernel.rbf
    compute()

def draw_gui():
    gui_objects.clear()
    sfreq_b = Slider(plt.axes([0.25, 0.1, 0.2, 0.03]), 'C', 0.1, 10, valinit=1, valstep=0.1)
    sfreq_b.on_changed(changeC)
    gui_objects.append(sfreq_b)
    #sfreq_u = Slider(plt.axes([0.25, 0.05, 0.2, 0.03]), 'NonZeroPt', 0, num_nz, valinit=0, valstep=1)
    #sfreq_u.on_changed(changeNZ)
    #gui_objects.append(sfreq_u)
    rax = plt.axes([0.05, 0.4, 0.15, 0.15])
    radio2 = RadioButtons(rax, ('linear', 'polynomial', 'rbf'))
    radio2.on_clicked(kernel)
    gui_objects.append(radio2)

plt.ion()
compute()
draw_gui()
input()