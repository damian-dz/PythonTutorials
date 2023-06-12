#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

dx = 0.01
x = np.arange(0, 10 + dx, dx)
y = np.sin(x)
plt.plot(x, y)
plt.grid()
plt.show()

yd = np.diff(y) / dx
# y[1] - y[0], y[2] - y[1], y[3] - y[2]

def interpolate(array, num_elements):
    old_indices = np.arange(len(array))
    new_indices = np.linspace(0, len(array) - 1, num_elements)
    array_interpolated = np.interp(new_indices, old_indices, array)
    return array_interpolated

yd = interpolate(yd, len(x))

plt.plot(x, yd)
plt.grid()
plt.show()

ydd = np.diff(yd) / dx
ydd = interpolate(ydd, len(x))

plt.plot(x, ydd)
plt.grid()
plt.show()
