"""Helpers to plot the output from Stan."""

import matplotlib.pyplot as plt
import numpy as np
import os.path as path
import re


def plot_output(fit, data, filename):
    """
    Create trace plots of chains, histograms of posterior beliefs and and plots
    of our posterior predictive distributions.

    Parameters
    ----------
    fit : StanFit4Model
        Output from pystan.
    data : generate_data.AR
        Class instance which holds observed data.
    filename : string
        Prefix to use in plot naming convention.

    """
    params = ['alpha', 'sigma']
    params += ['beta[{}]'.format(i+1) for i in range(data.beta.size)]

    plot_trace(fit, data, params, filename)
    plot_hist(fit, data, params, filename)
    plot_posterior_predictives(fit, data, params, filename)


def get_true_val(data, param):
    if param.startswith('beta'):
        index = int(re.search('[\d{1}]', param)[0]) - 1
        truth = getattr(data, 'beta')[index]
    else:
        truth = getattr(data, param)
    return truth


def plot_trace(fit, data, params, filename):
    """
    Produce trace plots of the trajectories.

    Parameters
    ----------
    fit : StanFit4Model
        Output from pystan.
    data : generate_data.AR
        Class instance which holds observed data.
    params : iterable
        Names of parameters to plot, as detailed in the fit object.
    filename : string
        Prefix to use in plot naming convention.

    """
    chain_length = fit.extract('alpha', permuted=False)['alpha'].shape[0]

    for param in params:
        fig, ax = plt.subplots(1, 1)
        ax.plot(np.r_[:chain_length],
                      fit.extract(param, permuted=False)[param],
                      alpha=0.85)

        ax.hlines(get_true_val(data, param), 0, chain_length, zorder=3)

        ax.set_ylabel(r'$\{}$'.format(re.sub(r'\[(\d{1})\]', r'_{\1}', param)))

        ax.set_xticks([0, chain_length // 2, chain_length])
        ax.set_xticklabels(['0', 'iteration', chain_length])
        ax.set_xlim(0, chain_length)

        fig.tight_layout()
        fig.savefig(path.join(path.dirname(path.realpath(__file__)),
                    'output',
                    '{}_trace_{}.png'.format(filename, param)),
                    format='png',
                    bbox_inches='tight')


def plot_hist(fit, data, params, filename):
    """
    Plot histograms of posterior beliefs about parameters

    Parameters
    ----------
    fit : StanFit4Model
        Output from pystan.
    data : generate_data.AR
        Class instance which holds observed data.
    params : iterable
        Names of parameters to plot, as detailed in the fit object.
    filename : string
        Prefix to use in plot naming convention.

    """
    chains = len(fit.stan_args)

    for param in params:
        fig, ax = plt.subplots(1, 1)

        n, bins, patches = [chains * [0] for chain in range(3)]

        for i in range(chains):
            n[i], bins[i], patches[i] = ax.hist(
                            fit.extract(param, permuted=False)[param][:, i],
                            25,
                            alpha=0.75,
                            density=1)
        ax.vlines(get_true_val(data, param), 0, np.max(n), zorder=3)

        ax.set_xlim(np.min(bins), np.max(bins))

        ax.set_xlabel(r'$\{}$'.format(re.sub(r'\[(\d{1})\]', r'_{\1}', param)))

        fig.tight_layout()
        fig.savefig(path.join(path.dirname(path.realpath(__file__)),
                    'output',
                    '{}_hist_{}.png'.format(filename, param)),
                    format='png',
                    bbox_inches='tight')


def plot_posterior_predictives(fit, data, params, filename):
    """
    Plot posterior predictive distributions on top of observed data.

    Parameters
    ----------
    fit : StanFit4Model
        Output from pystan.
    data : generate_data.AR
        Class instance which holds observed data.
    params : iterable
        Names of parameters to plot, as detailed in the fit object.
    filename : string
        Prefix to use in plot naming convention.

    """
    fig, ax = plt.subplots(1, 1)

    # Plot data
    ax.scatter(range(data.data.size),
               data.data,
               alpha=0.8,
               label='Observed data')

    y_pred = np.array(list(fit.extract('y_pred').values()))[0]

    # Plot 50% credible interval
    ax.fill_between(range(data.data.size),
                    np.percentile(y_pred, 25, axis=0),
                    np.percentile(y_pred, 75, axis=0),
                    color='C1',
                    alpha=0.75,
                    label = 'Posterior predictive 50%')

    # Plot 95% credible interval
    ax.fill_between(range(data.data.size),
                    np.percentile(y_pred, 2.5, axis=0),
                    np.percentile(y_pred, 97.5, axis=0),
                    color='C1',
                    alpha=0.3,
                    label = 'Posterior predictive 95%')

    # Plot posterior predictive mean
    ax.plot(range(data.data.size),
            y_pred.mean(0),
            c='C2',
            label = 'Posterior predictive mean',
            alpha=0.7)

    # Legend
    ax.legend(loc='upper center',
              bbox_to_anchor=(0.5, 1.1),
              ncol=4)

    ax.set_xlim(0, data.data.size)
    ax.set_xlabel('$t$')
    ax.set_ylabel('$X_t$')

    fig.tight_layout()
    fig.savefig(path.join(path.dirname(path.realpath(__file__)),
                'output',
                '{}_predictives.png'.format(filename)),
                format='png',
                bbox_inches='tight')
