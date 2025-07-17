import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def plot_mesh(X, Y, x, y, X_mid, Y_mid):
    """
    Plot original mesh and rectangular mesh over original mesh
    :param X:
    :param Y:
    :param x:
    :param y:
    :param X_mid:
    :param Y_mid:
    :return:
    """
    # Plot original mesh
    fig, ax = plt.subplots()
    ax.scatter(X, Y, marker='.', color='k')
    ax.set_title('Original Mesh')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Plot rectangular mesh over original mesh
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(np.zeros_like(X), cmap='gray', extent=[x.min(), x.max(), y.min(), y.max()], origin='lower')
    ax.scatter(X, Y, marker='.', color='k')
    ax.plot(X_mid, Y_mid, '--k', lw=1)
    ax.plot(X_mid.T, Y_mid.T, '--k', lw=1)
    ax.set_title('Rectangular Mesh over Original Mesh')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    fig.colorbar(im)
    ax.set(xlim=(-1.2, 1.2), ylim=(0.9, 5.3))

    return plt.show()


def plot_areas(X_mid, Y_mid, areas):
    """
    Plot relative areas of sub-rectangles
    :param X_mid:
    :param Y_mid:
    :param areas:
    :return:
    """
    fig, ax = plt.subplots()
    ax.pcolormesh(X_mid, Y_mid, areas, cmap='viridis', shading='auto')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Relative areas of sub-rectangles')
    return plt.show()