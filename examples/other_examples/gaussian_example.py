#!/usr/bin/env python
"""
An example of how to use bilby to perform paramater estimation for
non-gravitational wave data consisting of a Gaussian with a mean and variance
"""
from __future__ import division
import bilby
import numpy as np

# A few simple setup steps
label = 'gaussian_example'
outdir = 'outdir'

# Here is minimum requirement for a Likelihood class to run with bilby. In this
# case, we setup a GaussianLikelihood, which needs to have a log_likelihood
# method. Note, in this case we will NOT make use of the `bilby`
# waveform_generator to make the signal.

# Making simulated data: in this case, we consider just a Gaussian

data = np.random.normal(3, 4, 100)


class SimpleGaussianLikelihood(bilby.Likelihood):
    def __init__(self, data):
        """
        A very simple Gaussian likelihood

        Parameters
        ----------
        data: array_like
            The data to analyse
        """
        self.data = data
        self.N = len(data)
        self.parameters = {'mu': None, 'sigma': None}

    def log_likelihood(self):
        mu = self.parameters['mu']
        sigma = self.parameters['sigma']
        res = self.data - mu
        return -0.5 * (np.sum((res / sigma)**2) +
                       self.N * np.log(2 * np.pi * sigma**2))


likelihood = SimpleGaussianLikelihood(data)
priors = dict(mu=bilby.core.prior.Uniform(0, 5, 'mu'),
              sigma=bilby.core.prior.Uniform(0, 10, 'sigma'))

# And run sampler
result = bilby.run_sampler(
    likelihood=likelihood, priors=priors, sampler='dynesty', npoints=500,
    walks=10, outdir=outdir, label=label)
result.plot_corner()
