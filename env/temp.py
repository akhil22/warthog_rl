import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import time
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_box_aspect(1)
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
x = np.linspace(0., 6 * np.pi, 100)
y = np.sin(x)
ax.plot(x, y, 'r-')
rect = Rectangle((0, 0), 0.5, 0.3, 45.0)
ax.add_artist(rect)
t = []
for i in range(0, 1000):
    t1 = time.time()
    fig.canvas.draw()
    fig.canvas.flush_events()
    #rect.set_xy((x[i], y[i]))
    #ax.set_xlim([x[i] - 0.5, x[i] + 0.5])
    #ax.set_ylim([y[i] - 0.5, y[i] + 0.5])
    t2 = time.time()
    t.append(t2 - t1)
    print(rect.get_transform())
print(t)
