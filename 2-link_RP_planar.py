#!/usr/bin/env python
# coding: utf-8

# Inverse Dynamics of a Planar Two-Link RP Robot Arm
from sympy import *
from sympy.physics.mechanics import LagrangesMethod, dynamicsymbols
import numpy as np
from IPython.display import display, Markdown, Math
import matplotlib.pyplot as plt

# Defining the necessary symbols/functions:
m1, m2, l1, l2, g, t = symbols('m1, m2, l1, l2, g, t')
th, la = dynamicsymbols('theta, lambda')

# Defining the positions of the centers of mass based on the angle and the extension:
x1 = l1 * cos(th) / 2
y1 = l1 * sin(th) / 2
x2 = la * cos(th)
y2 = la * sin(th)

# Differentiating the positions with respect to time:
x1d = x1.diff(t)
y1d = y1.diff(t)
x2d = x2.diff(t)
y2d = y2.diff(t)

# Introducing shorthands for derivatives:
thd = Derivative(th, t)
thdd = Derivative(thd, t)
lad = Derivative(la, t)
ladd = Derivative(lad, t)

# Computing the kinetic and potential energies:
Ek1 = simplify(m1 * (x1d**2 + y1d**2) / 2 + m1 * l1**2 * thd**2 / 24)
Ek2 = simplify(m2 * (x2d**2 + y2d**2) / 2 + m2 * l2**2 * thd**2 / 24)

Ep1 = m1 * g * y1
Ep2 = m2 * g * y2

# The Lagrangian function:
L = simplify(Ek1 + Ek2 - Ep1 - Ep2)

# Forming Lagrange's equations of motion:
LM = LagrangesMethod(L, (th, la))
eqns = LM.form_lagranges_equations()

# Turning the equations of motion into functions that take numerical values:
params = [m1, m2, l1, l2, g, th, thd, thdd, la, lad, ladd]
T_lambda = utilities.lambdify(params, eqns[0])
F_lambda = utilities.lambdify(params, eqns[1])

# Introducing numerical values:
M1 = 1.4
M2 = 1.1
L1 = 0.9
L2 = 0.8
G = 9.81

# Plotting the general shape of the trajectory:
dx = 0.01
xs = np.arange(0, 10 + dx, dx)
ys = np.sin(3*xs) * np.cos(2*xs) * 2 * np.sin(3*xs)
plt.plot(xs, ys)
plt.grid()
plt.show()

# Fitting the assumed trajectory into a rectangle within the work area:
th_n = np.pi / 6
x_min = L1 * np.cos(th_n)
y_min = L1 * np.sin(th_n)
x_max = x_min + L2 * np.cos(th_n)
y_max = y_min + L2 * np.sin(th_n)

def fit(x, x_min, x_max):
    fitted = (x - np.min(x)) * (x_max - x_min) / (np.max(x) - np.min(x)) + x_min
    return fitted

x_traj = fit(xs, x_min, x_max)
y_traj = fit(ys, y_min, y_max)
plt.plot(x_traj, y_traj)
plt.grid()
plt.xlim([-0.1, 2])
plt.ylim([-0.1, 2])
plt.show()

# Conversion from the trajectory points to the joint variables:
TH = np.arctan(y_traj / x_traj)
LA = np.sqrt(x_traj**2 + y_traj**2) - L2/2

def interpolate(array, num_elements):
    old_indices = np.arange(len(array))
    new_indices = np.linspace(0, len(array) - 1, num_elements)
    array_interpolated = np.interp(new_indices, old_indices, array)
    return array_interpolated

# Numerical differentiation:
ts = np.linspace(0, 10, len(TH))
dt = ts[1] - ts[0]

THd = np.diff(TH) / dt
THd = interpolate(THd, len(TH))
THdd = np.diff(THd) / dt
THdd = interpolate(THdd, len(TH))

LAd = np.diff(LA) / dt
LAd = interpolate(LAd, len(LA))
LAdd = np.diff(LAd) / dt
LAdd = interpolate(LAdd, len(LA))

# Calculating the required torque and force:
T_req = T_lambda(M1, M2, L1, L2, G, TH, THd, THdd, LA, LAd, LAdd)
F_req = F_lambda(M1, M2, L1, L2, G, TH, THd, THdd, LA, LAd, LAdd)

# Plotting the torque:
plt.plot(ts, T_req)
plt.grid()
plt.xlabel('t [$s$]')
plt.ylabel('T [$N \cdot m$]')
plt.show()

# Plotting the force:
plt.plot(ts, F_req)
plt.grid()
plt.xlabel('t [$s$]')
plt.ylabel('F [$N$]')
plt.show()
