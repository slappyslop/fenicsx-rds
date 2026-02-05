
a_1 = 0.01
b_1 = 1.5
c_1 = 1.8
d_1 = 0.07

A_1 = 10
B_1 = 1.0
D_X1 = 0.01*1.25
D_Y1 = 0.1*1.25

a_2 = 0.0006
b_2 = 0.025
c_2 = 0.005
d_2 = 0.21
e_2 = 0.27
A_2 = 0.4
B_2 = 0.4
D_X2 = 0.0004*1.25
D_Y2 = 0.008*1.25

x1_star = 1.42857
y1_star = 0.58333
x2_star = 0.36079 #these were my calculated steady states but the qualitiative results only come when they are both set to 0
y2_star = 0.04821

n = 0.00125 
c_g = 0.001
dt = 0.01
eps = 1e-8

# Reaction kinetics
def f_1(x,y):
    return a_1*A_1 - b_1*B_1*x + c_1* (x**2) * y - d_1*x
def g_1(x,y):
    return b_1*B_1*x - c_1*(x**2)*y
def f_2(x, y, x2, y2):
    c_2 = n*x
    return a_2*A_2 + c_2*A_2*(x2**2)/ (y2 + eps) - d_2*x2
def g_2(x, y, x2, y2):
    return b_2*B_2*(x2**2) - e_2*y2


T = 300

