from scipy.optimize import minimize
import random, math
import numpy as np
import matplotlib.pyplot as plt

#N will be modified by the code after random is called
N = 0

matrix_p = np.empty((1,1))
targets = np.empty(0)

C = 10

sigma = None

def objective(vec_a):
    first_term = 0
    second_term = 0
    for i in range(N):
        for j in range(N):
            first_term += vec_a[i]*vec_a[j]*matrix_p[i][j]
        second_term += vec_a[i]
    #toRtn = (first_term/2) - second_term
    return (first_term/2) - second_term


#Implement different kernel functions
def linear_kernel_1(xi,xj):
    return np.dot(xi, xj)

def polynomial_kernel(xi,xj, p=2):
    return pow((np.dot(xi,xj) + 1), p) 

def RBF_kernel(xi, xj, a=sigma):
    return np.exp(-((np.linalg.norm(xi-xj)**2))/(2*a**2))

def kernel_fun(xi,xj):
    return linear_kernel_1(xi,xj)
    #return polynomial_kernel(xi,xj)
    #return RBF_kernel(xi,xj,2)

def zerofun(vec_a):
    toRtn = np.dot(vec_a, targets)
    #print("DEBUG Zerofun %f" % (toRtn))
    return toRtn

def save_matrix(inputs, targets):
    matrix_p = np.empty((N,N))
    for i in range(N):
        for j in range(N):
            matrix_p[i][j] = targets[i]*targets[j]*kernel_fun(inputs[i], inputs[j])
    return matrix_p

def perform_b(ext_alpha):
    b=0
    #print(len(ext_alpha))
    for s in ext_alpha:
        #Check that point is on the margin
        if C is None or (s['a']>0 and s['a']<C):
            print("DEBUG: ENTRATO in IF")
            for i in ext_alpha:
                b += i['a']*i['t']*kernel_fun(s['x'], i['x'])
            b = b - s['t']
            return b

def ind(s, ext_a, targets, b):
    to_rtn = 0
    for i in ext_a:
        to_rtn += i['a']*i['t']*kernel_fun(s,i['x'])

    to_rtn = to_rtn - b
    return to_rtn


#MAIN

#Generating data
np.random.seed()

classA = np.concatenate(
    (np.random.randn(10,2) * 0.2 + [1.5, 0.5],
    np.random.randn(10,2) * 0.2 + [-1.5, 0.5]))

classB = np.random.randn(20, 2) * 0.2 + [0.0, -0.5]

inputs = np.concatenate((classA, classB))
targets = np.concatenate(
    (np.ones(classA.shape[0]),
    -np.ones(classB.shape[0])))

N = inputs.shape[0]

#Compute sigma (std deviation)
sigma = np.std(inputs)

permute = list(range(N))
random.shuffle(permute)
inputs = inputs[permute, :]
targets = targets[permute]

zeros = np.zeros(N)

#Save matrix before calling minimiza
matrix_p = save_matrix(inputs, targets)

#print(matrix_p)

ret = minimize(objective, zeros, method='SLSQP' ,bounds=[(0,C) for b in range(N)], constraints={'type':'eq', 'fun':zerofun}) 

alpha = ret['x']

print("DEBUG: Result %r" % ret['success'])

#Extract the nonzero values
ext_alpha = list()
#print("DEBUG", len(alpha))
#print("DEBUG", alpha)
for i in range(len(alpha)):
    if abs(alpha[i]) > 0.0000001:
        #print("DEBUG Maggiore")
        #Save            
        ext_alpha.append({'a':alpha[i], 'x':inputs[i], 't':targets[i]})

#Calculate b using eq (7)
b = perform_b(ext_alpha)

print("DEBUG: b: %f" % (b))


#PLOTTING

plt.plot([p[0] for p in classA],
        [p[1] for p in classA], 'b.')

plt.plot([p[0] for p in classB],
        [p[1] for p in classB], 'r.')

#PLOTTING DEC BOUNDARY
x_grid = np.linspace(-3,3)
y_grid = np.linspace(-2,2)

grid=np.array([[ind(np.array((x,y)), ext_alpha, targets, b)
                    for x in x_grid] for y in y_grid])

plt.contour(x_grid, y_grid, grid, 
            (-1.0, 0.0, 1.0),
            colors=('red', 'black', 'blue'),
            linewidth=(1,3,1))

plt.axis('equal')
#plt.savefig('svmplot.pdf')
plt.show()




'''
minimize(objective, start, bounds=B, constraint=XC)
alpha = ret['x']

objective is a function we have to define that takes a vector a as argument and return a scalar value.
It should implement equation (4)

start is a vector with the initial guess of the a vector.
Can simply yse numpy.zeros(N)

B is a list of pairs of the same length as the a-vector
statint the lower and upper bounds for the corresponding element in a
e.g: B=[(0,C) for b in range(N)]
B=[(0,none) for b in range(N)] to only have lower bound

XC is used to impose other constraint.
We will use it to impose the equality constraint. Parameters is given
as a dictionare with the fields type and fun stating the type and
the implementation of the costraint
con = {'type':'eq', 'fun', zerofun}
zerofun is a function defined by us that calculate the value which should
be constrained to zero.
It takes a vector as argument and return a scalar value.

We have to implement:
- A suitable kernel function
    Start with the linear kernel (scalary product) but also esplore other kernels

- The function objective
    t and K can be global
    hint: pre-compute a matrix outside of the function and then store in a global variable
        Inside the function use numpy.dot and numpy.sum

- The function zerofun
    Should implement the equality costraint of equation (10)
    Use numpy.dot

- Call minimize
    It return a dictionare with some interesting keys
    'x' to pick out the actual a values
    'success' hold a boolean which is True if the optimizer
    actually found a solution

- Extract the non-zero a values
Only a few of the a values will be non-zero
We are dealing with floating point values so they will be 
approximately zero. Use a threshold (10^-5) to determine which
are to be regarded as non-zer0
Save the non-zerp a along with the corresponding data points (x) 
and target values (t) in a separata data structure

- Calculate the b value using equation (7)
    Must use a point ON the margin
    A point with a larger than zero but less than C

- Implement the indicator function, equation (6)
    Uses the non-zero a together with their x and t to classify new points


- Code for generating test data and visualizing the results in order
to test our support vector machine
'''