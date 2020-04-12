#! /usr/bin/env python3

"""Simulate and visualise autoregessive models."""

import numpy as np
import matplotlib.pyplot as plt
import os.path as path
import pickle 

class AR():
    """Class to represent an autoregressive model."""

    def __init__(self, alpha, beta, sigma):
        """
        Initialise class instance.

        Parameters
        ----------
        alpha : float
            Additive constant.
        beta : array-like, float.
            Parameters of the model.
        sigma : float
            Standard deviation of the noise distribution.

        """
        self.alpha = alpha
        self.beta = beta
        self.lag = self.beta.size
        self.sigma = sigma
        self.data = None

    def simulate(self, n=200):
        """
        Simulate the model for n time steps.

        Parameters
        ----------
        n : int, optional
            The number of data points to generate. The default is 200.

        """
        self.data = np.zeros(n)

        # Generate noise for sequence
        noise = np.random.normal(scale=self.sigma, size=n)

        # Generate initial data points
        self.data[:self.lag] = self.alpha + noise[:self.lag]

        # Simulate AR(K) process
        for t in range(self.lag, n):
            self.data[t] = (self.alpha
                           + np.sum(self.beta * self.data[t-self.lag:t][::-1])
                           + noise[t])

    def plot(self, filename=False):
        """
        Plot the time series.

        Parameters
        ----------
        filename : string, optional
            The name to use when saving the plot to file. By default the plot
            is not saved.

        """
        fig, ax = plt.subplots(1, 1)
        ax.scatter(np.r_[:self.data.size], self.data)
        ax.plot(np.r_[:self.data.size], self.data)

        ax.set_xlim(0, self.data.size)
        ax.set_xlabel('Lag')
        ax.set_ylabel('$X$')

        ax.grid(False)

        fig.tight_layout()
        if filename:
            filename += '.pdf' * (filename[-4:] != '.pdf')
            file_path = path.join(path.dirname(path.realpath(__file__)),
                                  'data',
                                  filename)
            fig.savefig(file_path, format='pdf')

    def save(self, filename=None):
        """
        Save this object instance.

        Parameters
        ----------
        filename : string, optional
            The name to use when saving this instance to file. If a file name
            isn't given, the instance well be saved as ARN.pkl where N is the
            order of the model.

        """
        if not filename:
            filename = 'AR{}'.format(self.lag)

        filename += '.pkl' * (filename[-4:] != '.pkl')
        file_path = path.join(path.dirname(path.realpath(__file__)),
                              'data',
                              filename)
        with open(file_path, "wb") as f:
            pickle.dump(self, f, protocol=-1)


if __name__ == "__main__":
    alpha = 0
    beta = np.array([0.75, -0.5, 0.2])
    sigma = 0.01
    AR3 = AR(alpha, beta, sigma)
    AR3.simulate()
    AR3.plot('AR3')
