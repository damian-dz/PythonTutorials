#!/usr/bin/env python
# coding: utf-8

from sympy import *
from sympy.physics.mechanics import LagrangesMethod, dynamicsymbols

# Symbols/functions:
m, l, r, g, t = symbols('m, l, r, g, t')
th_y, th_z = dynamicsymbols('theta_y, theta_z')

# Position of the center of mass based on the angles:
x = l * cos(th_y) * cos(th_z) / 2
y = l * cos(th_y) * sin(th_z) / 2
z = l * sin(th_y) / 2

# Time derivatives:
xd = simplify(x.diff(t))
yd = simplify(y.diff(t))
zd = z.diff(t)

# Kinetic energy component related to linear motion (translation):
KE_lin = simplify(m * (xd**2 + yd**2 + zd**2) / 2)

# Angular velocities in matrix form:
omega = Matrix([[0], [Derivative(th_y, t)], [Derivative(th_z, t)]])

# Moment of inertia matrix for a uniform cylinder:
I = Matrix([[m*l**2/12 + m*r**2/4, 0, 0], [0, m*l**2/12 + m*r**2/4, 0], [0, 0, m*r**2/2]])

# Kinetic energy component related to rotational motion (rotation):
KE_rot = simplify(omega.T * I * omega)[0]

# Total kinetic energy and potential energy:
KE = simplify(KE_lin + KE_rot)
PE = m * g * z

# Lagrangian function:
L = simplify(KE - PE)

# Equations of motion:
LM = LagrangesMethod(L, (th_y, th_z))
eqns = LM.form_lagranges_equations()
