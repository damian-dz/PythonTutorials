#!/usr/bin/env python
# coding: utf-8

from sympy import *
from sympy.physics.mechanics import LagrangesMethod, dynamicsymbols

# Define the necessary symbols and functions:
t, g, m, l = symbols('t, g, m, l')
th = dynamicsymbols('theta')

# Define the position of the mass point based on the angle:
x = l * sin(th)
y = -l * cos(th)

# Compute the time derivatives:
xd = x.diff(t)
yd = y.diff(t)

# Calculate the kinetic energy and the potential energy:
KE = simplify(m * (xd ** 2 + yd ** 2) / 2)
PE = m * g * y

# Find the Lagrangian function and its derivatives:
L = simplify(KE - PE); dispmath('L', L)

LM = LagrangesMethod(L, [th])
eqns = LM.form_lagranges_equations()
T = simplify(eqns[0])

# Solve for $\ddot{\theta}$ (for $T = 0$) and display it the form of a second-order differential equation:
THdd = solve(T, Derivative(th, (t, 2)))[0]
