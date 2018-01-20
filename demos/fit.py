import numpy as np
from traits.api import *
from traitschema import Schema


class FitResults(Schema):
    """Stores results from a fit."""
    amplitude = Float()
    frequency = Float()
    phase = Float()
    frequencies = Array(dtype=np.float)
    data = Array(dtype=np.float)
    covariance = Array(dtype=np.float, shape=(3, 3))


def sine(x, omega=1., A=1., phi=0.):
    """Sine function.

    Parameters
    ----------
    omega : float
        Angular frequency (default: 1)
    A : float
        Amplitude (default: 1)
    phi : float
        Phase (default: 0)

    """
    return A * np.sin(omega * x + phi)


def generate_data():
    x = np.linspace(-10, 10, 200)
    y = sine(x, 1, 2, np.pi/2.)
    return x, y + np.random.uniform(-1, 1, len(x))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit

    x, data = generate_data()
    p0 = [1, 1, 0]
    p, cov = curve_fit(sine, x, data, p0)

    plt.plot(x, data, '.', label='data')
    plt.plot(x, sine(x, *p), label='fit')
    plt.legend()
    plt.show()

    print(p)
    print(cov)

    fit = FitResults(frequency=p[0],
                     amplitude=p[1],
                     phase=p[2],
                     frequencies=x,
                     data=data)
    fit.to_hdf("output.h5")
