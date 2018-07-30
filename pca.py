import sys
import numpy as np
from sklearn.decomposition import PCA
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def pca_axes(X):
    pca = PCA(n_components = 2)
    pca.fit(X)

    eigen_vectors = pca.components_
    eigen_values = pca.explained_variance_

    major_scale = np.sqrt(2 * eigen_values[0])
    minor_scale = np.sqrt(2 * eigen_values[1])

    major_axis_unit = eigen_vectors[0, :]
    minor_axis_unit = eigen_vectors[1, :]

    major_axis = major_axis_unit * major_scale
    minor_axis = minor_axis_unit * minor_scale

    return major_axis, minor_axis

def rotate(X, theta):
    R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return np.dot(R, X.T).T

def ellipse(semi_major_axis, semi_minor_axis, sigma, n_points):
    t = np.linspace(0, 1, n_points, endpoint=True)
    noise = np.random.multivariate_normal([0, 0], [[sigma, 0], [0, sigma]], n_points)
    return np.c_[semi_major_axis * np.cos(2 * np.pi * t), semi_minor_axis * np.sin(2 * np.pi * t)] + noise

def draw_ellipse(theta, X, ax):
    X = rotate(X, theta)
    major_axis, minor_axis = pca_axes(X)

    plt.cla()
    plt.grid(True)
    plt.scatter(X[:, 0], X[:, 1])
    plt.quiver(0, 0, major_axis[0], major_axis[1], angles = 'xy', scale_units = 'xy', scale = 1, color = 'red')
    plt.quiver(0, 0, minor_axis[0], minor_axis[1], angles = 'xy', scale_units = 'xy', scale = 1, color = 'green')
    plt.draw()
    plt.pause(0.001)

if __name__ == '__main__':
    semi_major_axis = 2
    semi_minor_axis = 7
    n_points = 100
    sigma = 0.05
    n_animation_steps = 100

    fig = plt.figure()
    fig.set_tight_layout(True)
    fig.add_subplot(111)
    ax = fig.axes[0]
    ax.grid(color='k', linestyle='--')

    X = ellipse(semi_major_axis, semi_minor_axis, sigma, n_points)
    callback = partial(draw_ellipse, X = X, ax = ax)
    theta = np.linspace(0, 2 * np.pi, n_animation_steps)
    anim = FuncAnimation(fig, callback, frames = theta, interval=200)
    if len(sys.argv) > 1 and sys.argv[1] == 'save':
        anim.save('pca.gif', dpi=80, writer='imagemagick')
    else:
        plt.show()
