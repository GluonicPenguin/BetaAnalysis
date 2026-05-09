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

def binned_fit_langauss(samples, bins, min_x_val, max_x_val, channel, nan='remove'):
  if nan == 'remove':
    samples = samples[~np.isnan(samples)]
    #samples = samples[~(np.isnan(samples) | np.isinf(samples))]

  hist, bin_edges = np.histogram(samples, bins, range=(min_x_val, max_x_val), density=True)
  bin_centres = bin_edges[:-1] + np.diff(bin_edges) / 2

  mask = np.isfinite(hist) & np.isfinite(bin_centres)
  hist = hist[mask]
  bin_centres = bin_centres[mask]


  '''
  hist, bin_edges = np.histogram(samples, bins, range=(min_x_val,max_x_val), density=True)
  bin_centres = bin_edges[:-1] + np.diff(bin_edges) / 2

  hist = np.insert(hist, 0, sum(samples < bin_edges[0]))
  bin_centres = np.insert(bin_centres, 0, bin_centres[0] - np.diff(bin_edges)[0])
  hist = np.append(hist, sum(samples > bin_edges[-1]))
  bin_centres = np.append(bin_centres, bin_centres[-1] + np.diff(bin_edges)[0])

  hist = hist[1:-1]
  bin_centres = bin_centres[1:-1]

  mask = np.isfinite(hist) & np.isfinite(bin_centres)
  hist = hist[mask]
  bin_centres = bin_centres[mask]
  '''

  landau_x_mpv_guess = bin_centres[np.argmax(hist)]
  landau_xi_guess = median_abs_deviation(samples) / 5
  gauss_sigma_guess = landau_xi_guess / 10

  popt, pcov = curve_fit(
    lambda x, mpv, xi, sigma: langauss.pdf(x, mpv, xi, sigma),
    xdata=bin_centres,
    ydata=hist,
    p0=[landau_x_mpv_guess, landau_xi_guess, gauss_sigma_guess],
  )
  return popt, pcov, hist, bin_centres #hist, bin_centres

def round_to_sig_figs(x, sig):
  if x == 0:
    return 0
  else:
    return round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)

def plot_langaus(var, file, file_index, tree, channel_array, nBins, xLower, xUpper, savename):

  arr_of_ch = []
  arr_of_biases = []
  arr_of_MPV = []
  arr_of_width = []
  arr_of_sigma = []
  arr_mpv_frac = []
  arr_maxbin_frac = []
  arr_of_sse = []
  arr_of_rchi2 = []

  dict_of_vars = {"amplitude": "Amplitude / mV", "area_fitted": "Area / pWb", "dvdt": "dV/dt / mV/ps", "dvdt_2080": "dV/dt[20%:80%] / mV/ps"}
  for ch_ind, ch_val in enumerate(channel_array):
    area_list = []
    pmax_list = []
    dvdt_list = []
    dvdt_2080_list = []

    sensorType, AtQfactor, (A, B, C, D, E, F) = ch_val
    if AtQfactor == 0:
      AtQfactor = 1
    if sensorType == 1:
      if (A == 0.0) or (A == []):
        A = 0.0
      else:
        A = A[file_index]
      if B == 0: B = 1600
      if (C == 0.0) or (C == []):
        C = 0.0
      else:
        C = C[file_index]
      if D == 0: D = 1600
      if (E == 0.0) or (E == []):
        E = -50
      else:
        E = E[file_index]
      if (F == 0.0) or (F == []):
        F = 50
      else:
        F = F[file_index]
      bias_of_channel = getBias(str(file), ch_ind)
      for entry in tree:
        area_sig = entry.area_new[ch_ind]
        pmax_sig = entry.pmax[ch_ind]
        tmax_sig = entry.tmax[ch_ind]
        if (area_sig < A) or (area_sig > B) or (pmax_sig < C) or (pmax_sig > D) or (tmax_sig < E) or (tmax_sig > F):
          #print(f"BAD EVENTS {A} {B} {C} {D} {E} {F}")
          continue
        else:
          #if (area_sig > A) and (area_sig < B) and (pmax_sig > C) and (tmax_sig > D) and (tmax_sig < E):
          #dvdt_sig = entry.dvdt[ch_ind]
          #dvdt_2080_sig = entry.dvdt_2080[ch_ind]
          pmax_list.append(pmax_sig)
          area_list.append(area_sig)
          #dvdt_list.append(dvdt_sig)
          #dvdt_2080_list.append(dvdt_2080_sig)
    else:
      continue

    plt.figure(figsize=(10, 6))
    if var == "area_fitted":
      area = np.array(area_list)
      #area = area/AtQfactor
      data_var = area[(area>=xLower) & (area<=xUpper)]
    elif var == "dvdt":
      dvdt = np.array(dvdt_list)
      data_var = dvdt[(dvdt>=xLower) & (dvdt<=xUpper)]
    elif var == "dvdt_2080":
      dvdt_2080 = np.array(dvdt_2080_list)
      data_var = dvdt_2080[(dvdt_2080>=xLower) & (dvdt_2080<=xUpper)]
    else:
      pmax = np.array(pmax_list)
      data_var = pmax[(pmax>=xLower) & (pmax<=xUpper)]

    histo, bins, _ = plt.hist(data_var, bins=nBins, range=(xLower, xUpper), color='blue', edgecolor='black', alpha=0.6, density=True)
    #print(len(data_var))

    bin_centres = bins[:-1] + np.diff(bins) / 2

    popt, pcov, fitted_hist, bin_centres = binned_fit_langauss(data_var, nBins, xLower, xUpper, ch_ind)
    arr_of_ch.append("Ch"+str(ch_ind))
    arr_of_biases.append(bias_of_channel)
    arr_of_MPV.append(popt[0])
    arr_of_width.append(popt[1])
    arr_of_sigma.append(popt[2])

    count_1p0mpv = sum(1 for value in data_var if value > popt[0])
    count_1p5mpv = sum(1 for value in data_var if value > 1.5*popt[0])
    arr_mpv_frac.append(count_1p5mpv/count_1p0mpv)

    max_bin_index = np.argmax(histo)
    max_bin_centre = bin_centres[max_bin_index]
    count_max_bin = sum(1 for value in data_var if value > max_bin_centre)
    count_1p5max_bin = sum(1 for value in data_var if value > 1.5 * max_bin_centre)
    arr_maxbin_frac.append(count_1p5max_bin / count_max_bin if count_max_bin > 0 else 0)

    mpv, xi, sigma = popt
    y_fit = langauss.pdf(bin_centres, mpv, xi, sigma)
    residuals = histo - y_fit

    SSE = np.sum(residuals**2)
    normSSE = SSE / len(data_var)
    arr_of_sse.append(SSE)
    sigma = np.sqrt(histo)
    sigma[sigma == 0] = 1
    chi2 = np.sum((residuals / sigma) ** 2)

    N = len(histo)
    p = len(popt)
    nu = N - p
    chi2_red = chi2 / nu
    arr_of_rchi2.append(chi2_red)

    fig = go.Figure()
    fig.update_layout(
      xaxis_title=dict_of_vars[var],
      yaxis_title='Probability Density',
      title='Langauss Fit',
    )

    fig.add_trace(
      go.Histogram(
        x=data_var,
        name='Histogram',
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

    if not os.path.exists(var):
      os.makedirs(var)
    fig.write_image(var+"/"+savename+"_Ch"+str(ch_ind)+".png")
    print("[BETA ANALYSIS]: [LANGAUS PLOTTER] Saved file "+var+"/"+savename+"_Ch"+str(ch_ind)+".png")

  if (var != "dvdt") & (var != "dvdt_2080"): var = var.capitalize()
  if var == "Area_fitted": var = "Area"

  df_of_results = pd.DataFrame({
    "Channel": arr_of_ch,
    "Bias": arr_of_biases,
    var + " MPV": arr_of_MPV,
    "Landau width": arr_of_width,
    "Gaussian sigma": arr_of_sigma,
    "Frac above 1p5 MPV": arr_mpv_frac,
    "Frac above 1p5 Max Bin": arr_maxbin_frac,
    "SSE score": arr_of_sse,
    "Red. Chi2": arr_of_rchi2,
  })
  if var != "Area_fitted":
    df_of_results = df_of_results.drop(columns=['Frac above 1p5 Max Bin'])
  return df_of_results
