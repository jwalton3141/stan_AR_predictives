#! /usr/bin/env python3

"""Use stan to fit AR(p) model to data and visualise output."""


from hashlib import md5
import pystan
import pickle
import os.path as path
import sys

from generate_data import AR
from plot_output import plot_output


def load_pkl(filename):
    """
    Load pickled object.

    Parameters
    ----------
    filename : string
        Path to file to load.

    Returns
    -------
    obj
        Requested file.

    """
    # Ensure filename has .pkl extension
    filename += '.pkl' * (filename[-4:] != '.pkl')

    # Load data
    with open(filename, 'rb') as f:
        return pickle.load(f)


def proj_dir():
    """
    Get the path to the root of this project.

    Returns
    -------
    string
        Absolute path to the root of this project.

    """
    return path.dirname(path.realpath(__file__))


def StanModel_cache(fit_name, **kwargs):
    """
    Automatically save and re-use compiled models where possible.

    Parameters
    ----------
    fit_name : string
        Name of .stan file.
    **kwargs : dict
        Extra arguments to be passed to pystan.StanMode.

    Returns
    -------
    compiled_model : pystan.model.StanModel
        Model described in Stan's modeling language compiled from C++ code.

    """
    # Get path to model
    model_path = path.join(proj_dir(), '{}.stan'.format(fit_name))

    # Load model as string
    if path.isfile(model_path):
        with open(model_path, 'r') as f:
            model_code = f.read()
    else:
        print("Model {} was not found at {}. "
              "Did you specify a valid model?".format(fit_name, model_path))
        sys.exit(1)

    # Compute hash of model code
    code_hash = md5(model_code.encode('ascii')).hexdigest()
    # Get name of cached model
    cache_path = path.join(proj_dir(),
                           'cached',
                           'compiled-{}-{}.pkl'.format(fit_name, code_hash))

    if path.isfile(cache_path):
        # Load model
        compiled_model = pickle.load(open(cache_path, 'rb'))
        print("Using cached StanModel")
    else:
        # Model cache_fn couldn't be found so compile and save
        compiled_model = pystan.StanModel(file=model_path,
                                          model_name=fit_name,
                                          **kwargs)
        with open(cache_path, "wb") as f:
            pickle.dump(compiled_model, f, protocol=-1)

    return compiled_model


def main(data, iters, warmup, chains):
    """
    Fit data to model with stan

    Parameters
    ----------
    data : generate_data.AR
        Object representing data to fit model to.
    iters : int
        Number of iterations to run stan for.
    warmup : int
        Number of warm-up iterations to use.
    chains : int
        Number of chains to run.

    Returns
    -------
    fit : StanFit4Model
        Stan object representing output of fitting procedure.

    """
    data = {'P': data.lag, 'N': data.data.size, 'y': data.data}

    model = StanModel_cache('ARN')
    fit = model.sampling(data=data,
                         iter=iters,
                         warmup=warmup,
                         chains=chains,
                         n_jobs=-1)

    return fit


if __name__ == "__main__":

    filename = input("Name of data to fit to: ")
    data = load_pkl(path.join(path.dirname(path.realpath(__file__)),
                              'data',
                              filename))

    iters = int(input("Iterations (default = 10,000): ") or "10000")
    warmup = int(input("Warm-up (default = {}): ".format(iters // 2)) 
                 or iters // 2)
    chains = int(input("Chains (default = 4): ") or "4")

    fit = main(data, iters, warmup, chains)
    plot_output(fit, data, filename)
