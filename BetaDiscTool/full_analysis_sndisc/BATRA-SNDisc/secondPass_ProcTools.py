import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from landaupy import langauss
from scipy.stats import median_abs_deviation, norm
from scipy.optimize import curve_fit

import ROOT as root

def binned_fit_langauss(samples, bins, min_x_val, max_x_val, nan='remove'):
  if nan == 'remove':
    samples = samples[~np.isnan(samples)]

  hist, bin_edges = np.histogram(samples, bins, range=(min_x_val,max_x_val), density=True)
  bin_centres = bin_edges[:-1] + np.diff(bin_edges) / 2
  hist = np.insert(hist, 0, sum(samples < bin_edges[0]))
  bin_centres = np.insert(bin_centres, 0, bin_centres[0] - np.diff(bin_edges)[0])
  hist = np.append(hist, sum(samples > bin_edges[-1]))
  bin_centres = np.append(bin_centres, bin_centres[-1] + np.diff(bin_edges)[0])

  hist = hist[1:-1]
  bin_centres = bin_centres[1:-1]
  landau_x_mpv_ansatz = bin_centres[np.argmax(hist)]
  landau_xi_ansatz = median_abs_deviation(samples) / 5
  gauss_sigma_ansatz = landau_xi_ansatz / 10

  popt, pcov = curve_fit(
    lambda x, mpv, xi, sigma: langauss.pdf(x, mpv, xi, sigma),
    xdata=bin_centres,
    ydata=hist,
    p0=[landau_x_mpv_ansatz, landau_xi_ansatz, gauss_sigma_ansatz],
  )
  return popt, pcov, hist, bin_centres

def gaussian(x, A, mu, sigma):
  return A * norm.pdf(x, mu, sigma)

def binned_fit_gaussian(samples, nBins, nan='remove'):
  if nan == 'remove':
    samples = samples[~np.isnan(samples)]

  mu_ansatz = np.mean(samples)
  sigma_ansatz = np.std(samples)
  hist, bin_edges = np.histogram(samples, bins=nBins, range=(max(0, mu_ansatz - 2*sigma_ansatz), min(2*mu_ansatz, mu_ansatz + 2*sigma_ansatz)), density=True)
  bin_centres = bin_edges[:-1] + np.diff(bin_edges) / 2

  A_ansatz = np.trapz(hist, bin_centres)
  print(mu_ansatz)
  print(sigma_ansatz)

  popt, pcov = curve_fit(gaussian, bin_centres, hist, p0=[mu_ansatz, sigma_ansatz, A_ansatz])
  mu, sigma, A = popt

  y_fit = gaussian(bin_centres, mu, sigma, A)
  residuals = hist - y_fit

  sse = np.sum(residuals**2)
  sigma = np.sqrt(hist)
  sigma[sigma == 0] = 1
  chi2 = np.sum((residuals / sigma)**2)
  dof = len(hist) - len(popt)
  rchi2 = chi2 / dof

  return popt, sse, rchi2
