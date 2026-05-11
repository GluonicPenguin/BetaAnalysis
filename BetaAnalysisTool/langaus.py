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

def get_true_mpv_from_grid(popt, x_grid):
  y_scan = langauss.pdf(x_grid, *popt)
  return x_grid[np.argmax(y_scan)]

def estimate_mpv_and_fraction_uncertainty(data_var, popt, pcov, x_grid, n_toys=200):
  mpv_samples = []
  xi_samples = []
  sigma_samples = []
  frac_mpv_samples = []
  frac_maxbin_samples = []
  ratio_samples = []

  histo, bins = np.histogram(data_var, bins=50)
  bin_centres = bins[:-1] + np.diff(bins) / 2
  max_bin_centre = bin_centres[np.argmax(histo)]
  for _ in range(n_toys):
    try:
      sampled_params = np.random.multivariate_normal(popt, pcov)
    except np.linalg.LinAlgError:
      continue

    mpv_s, xi_s, sigma_s = sampled_params
    # --- true MPV ---
    mpv_true = get_true_mpv_from_grid(sampled_params, x_grid)
    mpv_samples.append(mpv_true)
    xi_samples.append(xi_s)
    sigma_samples.append(sigma_s)
    # --- ratio ---
    if mpv_true != 0:
      ratio_samples.append(xi_s / mpv_true)
    # --- MPV fraction ---
    count_1p0 = np.sum(data_var > mpv_true)
    count_1p5 = np.sum(data_var > 1.5 * mpv_true)
    if count_1p0 > 0:
      frac_mpv_samples.append(count_1p5 / count_1p0)
    # --- max bin fraction ---
    count_max = np.sum(data_var > max_bin_centre)
    count_1p5_max = np.sum(data_var > 1.5 * max_bin_centre)
    if count_max > 0:
      frac_maxbin_samples.append(count_1p5_max / count_max)

  return {"mpv_mean": np.mean(mpv_samples),
          "mpv_std": np.std(mpv_samples),
          "xi_std": np.std(xi_samples),
          "sigma_std": np.std(sigma_samples),
          "frac_mpv_std": np.std(frac_mpv_samples),
          "frac_maxbin_std": np.std(frac_maxbin_samples),
          "ratio_mean": np.mean(ratio_samples),
          "ratio_std": np.std(ratio_samples),
         }

def compute_ltf_extrapolated(popt, mpv_true, x_min, max_factor=20, n_points=3000):
  """
  Compute LTF using extrapolated Langaus fit beyond fit range.

  Parameters:
    popt: fit parameters (mpv, xi, sigma)
    mpv_true: true MPV
    x_min: lower bound (same as your histogram lower edge)
    max_factor: integrate up to max_factor * MPV
    n_points: grid resolution
 
  Returns:
    float: LTF value
    """
  upper_bound = max_factor * mpv_true
  x_grid = np.linspace(x_min, upper_bound, n_points)
  y = langauss.pdf(x_grid, *popt)
  mask_mpv = x_grid >= mpv_true
  mask_1p5 = x_grid >= 1.5 * mpv_true
  area_mpv = np.trapz(y[mask_mpv], x_grid[mask_mpv])
  area_1p5 = np.trapz(y[mask_1p5], x_grid[mask_1p5])
  if area_mpv == 0:
    return 0
  return area_1p5 / area_mpv

def round_to_sig_figs(x, sig):
  if x == 0:
    return 0
  else:
    return round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)

def fmt_val_unc(val, unc, sig_figs_val=3):
  if val == 0:
    val_rounded = 0
    decimals = 0
  else:
    order = int(np.floor(np.log10(abs(val))))
    decimals = max(sig_figs_val - 1 - order, 0)
    val_rounded = round(val, decimals)
  unc_rounded = round(unc, decimals)
  fmt_str = f"{{:.{decimals}f}} ± {{:.{decimals}f}}"
  return fmt_str.format(val_rounded, unc_rounded)

def plot_langaus(var, file, file_index, tree, channel_array, nBins, xLower, xUpper, savename, thickness):

  arr_of_ch = []
  arr_of_biases = []
  arr_of_MPV = []
  arr_of_MPV_unc = []
  arr_of_width = []
  arr_of_sigma = []
  arr_mpv_frac = []
  arr_maxbin_frac = []
  arr_mpv_frac_unc = []
  arr_maxbin_frac_unc = []
  arr_ratio = []
  arr_ratio_unc = []
  arr_of_sse = []
  arr_of_rchi2 = []
  arr_mpv_frac_from_area = []

  x_axis_add = []
  y_fit_counts_add = []

  dict_of_vars = {"amplitude": "Amplitude / mV", "area_fitted": "Area / pWb", "charge": "Charge / fC", "gain": "Gain / fC", "dvdt": "dV/dt / mV/ps", "dvdt_2080": "dV/dt[20%:80%] / mV/ps"}
  for ch_ind, ch_val in enumerate(channel_array):
    area_list = []
    pmax_list = []
    negpmax_list = []
    dvdt_list = []
    dvdt_2080_list = []

    sensorType, AtQfactor, (A, B, C, D, E, F, G) = ch_val
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
      if G == 0:
        G = -100
      bias_of_channel = getBias(str(file), ch_ind)
      for entry in tree:
        area_sig = entry.area_new[ch_ind]
        pmax_sig = entry.pmax[ch_ind]
        tmax_sig = entry.tmax[ch_ind]
        negpmax_sig = entry.negpmax[ch_ind]
        if (area_sig < A) or (area_sig > B) or (pmax_sig < C) or (pmax_sig > D) or (tmax_sig < E) or (tmax_sig > F) or (negpmax_sig < G):
          #print(f"BAD EVENTS {A} {B} {C} {D} {E} {F}")
          continue
        else:
          #if (area_sig > A) and (area_sig < B) and (pmax_sig > C) and (tmax_sig > D) and (tmax_sig < E):
          #dvdt_sig = entry.dvdt[ch_ind]
          #dvdt_2080_sig = entry.dvdt_2080[ch_ind]
          pmax_list.append(pmax_sig)
          area_list.append(area_sig)
          negpmax_list.append(negpmax_sig)
          #dvdt_list.append(dvdt_sig)
          #dvdt_2080_list.append(dvdt_2080_sig)
    else:
      continue

    plt.figure(figsize=(16, 10))
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

    histo, bins, _ = plt.hist(data_var, bins=nBins, range=(xLower, xUpper), color='white', edgecolor='black', alpha=0.6, density=True)
    bin_width = bins[1] - bins[0]
    bin_centres = bins[:-1] + np.diff(bins) / 2

    if (var == "area_fitted") & (int(bias_of_channel.rstrip("V")) >= 300):
      X_cut = 11
      mask_cut = bin_centres >= X_cut
      histo_masked = histo.copy()
      histo_masked[~mask_cut] = 0

    popt, pcov, fitted_hist, bin_centres = binned_fit_langauss(data_var, nBins, xLower, xUpper, ch_ind)
    if (var == "area_fitted") & (int(bias_of_channel.rstrip("V")) >= 300):
      counts = histo_masked
    else:
      counts = fitted_hist * len(data_var) * bin_width
    arr_of_ch.append("Ch"+str(ch_ind))
    arr_of_biases.append(bias_of_channel)

    x_grid = np.linspace(xLower, xUpper, 1000)
    mpv_true = get_true_mpv_from_grid(popt, x_grid)

    arr_of_MPV.append(mpv_true)
    arr_of_width.append(popt[1])
    arr_of_sigma.append(popt[2])

    count_1p0mpv = np.sum(data_var > mpv_true)
    count_1p5mpv = np.sum(data_var > 1.5 * mpv_true)
    frac_mpv = count_1p5mpv/count_1p0mpv
    arr_mpv_frac.append(frac_mpv)
    ltf_fit = compute_ltf_extrapolated(popt, mpv_true, xLower)
    arr_mpv_frac_from_area.append(ltf_fit)

    max_bin_index = np.argmax(histo)
    max_bin_centre = bin_centres[max_bin_index]
    count_max_bin = sum(1 for value in data_var if value > max_bin_centre)
    count_1p5max_bin = sum(1 for value in data_var if value > 1.5 * max_bin_centre)

    frac_maxbin = count_1p5max_bin / count_max_bin if count_max_bin > 0 else 0
    arr_maxbin_frac.append(frac_maxbin)

    unc_dict = estimate_mpv_and_fraction_uncertainty(data_var, popt, pcov, x_grid)
    ratio_val = popt[1] / mpv_true
    ratio_unc = unc_dict["ratio_std"]
    #ratio_val = unc_dict["ratio_mean"]
    arr_ratio.append(ratio_val)
    arr_ratio_unc.append(ratio_unc)
     
    mpv_unc = unc_dict["mpv_std"]
    xi_unc = unc_dict["xi_std"]
    sigma_unc = unc_dict["sigma_std"]
    mpv_val, xi_val, sigma_val = popt
    arr_of_MPV_unc.append(mpv_unc)
    frac_mpv_unc_syst = unc_dict["frac_mpv_std"]
    frac_maxbin_unc_syst = unc_dict["frac_maxbin_std"]
    frac_mpv_stat = np.sqrt(frac_mpv * (1 - frac_mpv) / count_1p0mpv) if count_1p0mpv > 0 else 0
    frac_maxbin_stat = np.sqrt(frac_maxbin * (1 - frac_maxbin) / count_max_bin) if count_max_bin > 0 else 0
 
    frac_mpv_total_unc = np.sqrt(frac_mpv_stat**2 + frac_mpv_unc_syst**2)
    frac_maxbin_total_unc = np.sqrt(frac_maxbin_stat**2 + frac_maxbin_unc_syst**2)
    arr_mpv_frac_unc.append(frac_mpv_total_unc)
    arr_maxbin_frac_unc.append(frac_maxbin_total_unc)

    mpv, xi, sigma = popt
    y_fit_pdf = langauss.pdf(bin_centres, mpv, xi, sigma)
    y_fit = y_fit_pdf * len(data_var) * bin_width
    x_axis = np.linspace(xLower, xUpper, 999)
    y_fit_counts = langauss.pdf(x_axis, *popt) * len(data_var) * bin_width
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

    mpvext, xiext, sigmaext = popt
    y_landau_pdf = langauss.landau.pdf(x_axis, mpvext, xiext)
    y_landau_pdf *= y_fit_pdf.max() / y_landau_pdf.max()
    y_landau_counts = y_landau_pdf * len(data_var) * bin_width

    x_axis_add.append(x_axis)
    y_fit_counts_add.append(y_fit_counts)

    legend_fontsize=42
    bias_of_channel_spaced = bias_of_channel[:-1] + " " + bias_of_channel[-1]

    fig = go.Figure()
    fig.update_layout(
      width=1600,
      height=1000,
      font=dict(
        family="Arial",
        size=legend_fontsize,
        color="black"
      ),
      legend=dict(font=dict(size=legend_fontsize+4),
        x=1,
        y=1,
        xanchor='right',
        yanchor='top',
        bgcolor='white',
        bordercolor='lightgray',
        borderwidth=2,
        itemwidth=80
        ),
      xaxis_title=dict_of_vars[var],
      yaxis_title='Number of events',
    )
    if (var == "charge") & (int(bias_of_channel.rstrip("V")) >= 300):
      fig.add_trace(
        go.Scatter(
          x=bin_centres,
          y=counts,
          mode='markers',
          name=f"{thickness} μm, V<sub>bias</sub> = {bias_of_channel_spaced}",
          marker=dict(color='black', size=15),
          error_y=dict(type='data', array=np.sqrt(histo_masked))
        )
      )
    else:
      fig.add_trace(
        go.Scatter(
          x=bin_centres,
          y=counts,
          mode='markers',
          name=f"{thickness} μm, V<sub>bias</sub> = {bias_of_channel_spaced}",
          marker=dict(color='black', size=15),
          error_y=dict(type='data', array=np.sqrt(histo))
        )
      )
    x_axis = np.linspace(xLower, xUpper, 999)
    fig.add_trace(
    go.Scatter(
        x=x_axis,
        y=y_fit_counts,
        name=(
             f"<i>Q</i>~<b>θ</b>(<i>Q</i><sub>MPV</sub> = {fmt_val_unc(mpv_true, mpv_unc)},<br>"
             f"          ξ = {fmt_val_unc(xi_val, xi_unc)},<br>"
             f"          σ = {fmt_val_unc(sigma_val, sigma_unc)})"
            ),
        mode='lines',
        line=dict(color='red', width=10)
        )
    )
    fig.add_trace(
      go.Scatter(
        x=x_axis,
        y=y_landau_counts,
        mode='lines',
        name=f'  Landau contribution',
        line=dict(color='blue', width=10, dash='dash')
      )
    )

    fig.update_layout(
      plot_bgcolor="white",
      paper_bgcolor="white",
      xaxis=dict(
        showgrid=True,
        gridcolor="lightgray",
        zeroline=False
      ),
      yaxis=dict(
        showgrid=True,
        gridcolor="lightgray",
        zeroline=False,
        range=[0, 1.2*max(counts)]
      )
    )

    if not os.path.exists(var):
      os.makedirs(var)
    fig.write_image(var+"/GS_STUDIES_"+savename+"_Ch"+str(ch_ind)+".png")
    print("[BETA ANALYSIS]: [LANGAUS PLOTTER] Saved file "+var+"/"+savename+"_Ch"+str(ch_ind)+".png")
    
  if (var != "dvdt") & (var != "dvdt_2080"): var = var.capitalize()
  if var == "Area_fitted": var = "Area" 

  df_of_langaus = pd.DataFrame({"BIAS": arr_of_biases,var: x_axis_add,var+"_EVENTS": y_fit_counts_add})
  var = var.capitalize()
  df_of_results = pd.DataFrame({
    "Channel": arr_of_ch,
    "Bias": arr_of_biases,
    var: arr_of_MPV,
    var+" Unc": arr_of_MPV_unc,
    "Landau width": arr_of_width,
    "Gaussian sigma": arr_of_sigma,
    "LTF": arr_mpv_frac,
    "LTF Unc": arr_mpv_frac_unc,
    "LTF from area": arr_mpv_frac_from_area,
    "LTFmax": arr_maxbin_frac,
    "LTFmax Unc": arr_maxbin_frac_unc,
    "Landau Frac": arr_ratio,
    "Landau Frac Unc": arr_ratio_unc,
    "SSE score": arr_of_sse,
    "Red. Chi2": arr_of_rchi2,
  })
  return df_of_results, df_of_langaus
