import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import minimize
from scipy.stats import poisson, median_abs_deviation
import ROOT as root
from ROOT import TF1
from scipy.special import gammaln
import math
from math import exp, sqrt, pi
import pandas as pd
import argparse
import glob
import re
import os
import csv
import math
import plotly.graph_objects as go

import sys
from datetime import datetime
import matplotlib.pylab as plt
import matplotlib.axes as axes
from array import array
from landaupy import langauss
from scipy.optimize import curve_fit

from proc_tools import getBias

def binned_fit_langauss(samples, bins, max_x_val, channel, nan='remove'):
  if nan == 'remove':
    samples = samples[~np.isnan(samples)]

  hist, bin_edges = np.histogram(samples, bins, range=(0,max_x_val), density=True)
  bin_centers = bin_edges[:-1] + np.diff(bin_edges) / 2

  hist = np.insert(hist, 0, sum(samples < bin_edges[0]))
  bin_centers = np.insert(bin_centers, 0, bin_centers[0] - np.diff(bin_edges)[0])
  hist = np.append(hist, sum(samples > bin_edges[-1]))
  bin_centers = np.append(bin_centers, bin_centers[-1] + np.diff(bin_edges)[0])

  hist = hist[1:]
  bin_centers = bin_centers[1:]

  landau_x_mpv_guess = bin_centers[np.argmax(hist)]
  landau_xi_guess = median_abs_deviation(samples) / 5
  gauss_sigma_guess = landau_xi_guess / 10

  popt, pcov = curve_fit(
    lambda x, mpv, xi, sigma: langauss.pdf(x, mpv, xi, sigma),
    xdata=bin_centers,
    ydata=hist,
    p0=[landau_x_mpv_guess, landau_xi_guess, gauss_sigma_guess],
  )
  return popt, pcov, hist, bin_centers

def round_to_sig_figs(x, sig):
  if x == 0:
    return 0
  else:
    return round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)

def plot_langaus(file, tree, channel_array, nBins, xLower, xUpper, savename):

  pmax_list = []
  area_list = []

  arr_of_ch = []
  arr_of_biases = []
  arr_of_MPV = []
  arr_of_width = []
  arr_of_sigma = []
  arr_threshold_frac = []

  for ch_ind, ch_val in enumerate(channel_array):
    pmax_list = []
    area_list = []
    sensorType, AtQfactor, (A, B, C, D, E) = ch_val
    if sensorType == 1:
      if B == 0: B = 1000
      if C == 0: C = -100
      if D == 0: D = -50
      if E == 0: E = 50
      bias_of_channel = getBias(file)
      for entry in tree:
        pmax_sig = entry.pmax[ch_ind]
        negpmax_sig = entry.negpmax[ch_ind]
        tmax_sig = entry.tmax[ch_ind]
        if (pmax_sig < A) or (pmax_sig > B) or (negpmax_sig < C) or (tmax_sig < D) or (tmax_sig > E):
          continue
        else:
          area_sig = entry.area_new[ch_ind]
          pmax_list.append(pmax_sig)
          area_list.append(area_sig)
    else:
      continue

    pmax = np.array(pmax_list)
    area = np.array(area_list)
    area = area/AtQfactor
    area = area[(area>=xLower) & (area<=xUpper)]

    plt.figure(figsize=(10, 6))
    histo, bins, _ = plt.hist(area, bins=nBins, range=(xLower, xUpper), color='blue', edgecolor='black', alpha=0.6, density=True)
    #plt.xlabel('Charge [fC]')
    #plt.ylabel('Frequency')
    #plt.title('Distribution charge')
    #plt.yscale('log')
    #plt.show()

    bin_centers = bins[:-1] + np.diff(bins) / 2
    popt, pcov, fitted_hist, bin_centers = binned_fit_langauss(area, nBins, xUpper, ch_ind)
    
    arr_of_ch.append("Ch"+str(ch_ind))
    arr_of_biases.append(bias_of_channel)
    arr_of_MPV.append(popt[0])
    arr_of_width.append(popt[1])
    arr_of_sigma.append(popt[2])

    count_1p0mpv = sum(1 for value in area if value > popt[0])
    count_1p5mpv = sum(1 for value in area if value > 1.5*popt[0])
    #print("fEv@1.5:")
    #print(count_1p5mpv/count_1p0mpv)
    arr_threshold_frac.append(count_1p5mpv/count_1p0mpv)

    fig = go.Figure()
    fig.update_layout(
      xaxis_title='Charge [fC]',
      yaxis_title='Probability Density',
      title='Langauss Fit',
    )

    fig.add_trace(
      go.Histogram(
        x=area,
        name='Histogram of area',
        histnorm='probability density',
        nbinsx=nBins,
        opacity=0.6,
        marker=dict(color='blue', line=dict(color='black', width=1)),
      )
    )

    x_axis = np.linspace(xLower, xUpper, 999)
    fig.add_trace(
      go.Scatter(
        x=x_axis,
        y=langauss.pdf(x_axis, *popt),
        name=f'Langauss Fit<br>MPV={popt[0]:.2e}<br>ξ={popt[1]:.2e}<br>σ={popt[2]:.2e}',
        mode='lines',
      )
    )

    fig.write_image(savename+"_Ch"+str(ch_ind)+".png")
    print("[BETA ANALYSIS]: [LANGAUS PLOTTER] Saved file "+savename+"_Ch"+str(ch_ind)+".png")

  df_of_results = pd.DataFrame({
    "Channel": arr_of_ch,
    "Bias": arr_of_biases,
    "MPV Charge": arr_of_MPV,
    "Landau width": arr_of_width,
    "Gaussian sigma": arr_of_sigma,
    "Frac above 1p5": arr_threshold_frac,
  })
  return df_of_results
