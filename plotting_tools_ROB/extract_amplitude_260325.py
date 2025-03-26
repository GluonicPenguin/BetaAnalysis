import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.optimize as opt
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
from sklearn.metrics import mean_absolute_error, mean_squared_error


import sys
from datetime import datetime
import matplotlib.pylab as plt
import matplotlib.axes as axes
#from langaus import LanGausFit
from array import array
from landaupy import langauss
from scipy.optimize import curve_fit

var_dict = {"tmax":"t_{max} / 10 ns" , "pmax":"p_max / mV" , "negpmax":"-p_max / mV", "charge":"Q / fC", "area_new":"Area / pWb" , "rms":"RMS / mV"}

def round_to_sig_figs(x, sig):
  if x == 0:
    return 0
  else:
    return round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)

def gaussian(x, A, mu, sigma):
  return A * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

def extract_CFD_time(t_vals, a_vals, a_peak, frac):
  A_CFD = frac * a_peak / 1000
  for i in range(len(a_vals) - 1):
    if a_vals[i] < A_CFD and a_vals[i + 1] >= A_CFD:
      t_CFD = t_vals[i] + (A_CFD - a_vals[i]) * (t_vals[i + 1] - t_vals[i]) / (a_vals[i + 1] - a_vals[i])
      return t_CFD

  return None 

def main():
  parser = argparse.ArgumentParser(description='Read .root files into an array.')
  parser.add_argument('files', metavar='F', type=str, nargs='+',
                      help='List of .root files or wildcard pattern (*.root)')
  args = parser.parse_args()

  files = []
  trees = []

  for pattern in args.files:
    root_files = glob.glob(pattern)
    for root_file in root_files:
      theFile = root.TFile.Open(root_file)
      theTree = theFile.Get("Analysis")
      files.append(theFile)
      trees.append(theTree)

  t_data = [[] for _ in range(len(trees))]
  w_data = [[] for _ in range(len(trees))]
  cfd10_data = []
  cfd20_data = []
  cfd30_data = []
  num_curves = 1000
  ch_sig = 2
  ch_mcp = 3

  for j in range(len(trees)):
    tree = trees[j]
    i = 0
    ev_true_count = 0
    for entry in tree:
      i += 1
      #if i > 10000:
      #  continue
      #entry_index = entry.event
      pmax_sig = entry.pmax[ch_sig]
      negpmax_sig = entry.negpmax[ch_sig]
      pmax_mcp = entry.pmax[ch_mcp]
      cfd10_sig = entry.cfd[ch_sig][0] # 10%
      cfd20_sig = entry.cfd[ch_sig][1] # 20%
      cfd30_sig = entry.cfd[ch_sig][2] # 30%
      if (pmax_sig > 40) and (pmax_mcp < 540):
        # W12 15e14/25e14 (pmax_sig > 10) and (pmax_sig < 30) and (negpmax_sig > -30) and (pmax_mcp < 120) and (peakfind > 9) and (peakfind < 14)
        # W13 35e14 (pmax_sig > 55) and (pmax_sig < 80) and (negpmax_sig > -30) and (pmax_mcp < 120) and (peakfind > 9) and (peakfind < 14)
        w_sig = entry.w[ch_sig]
        t_sig = entry.t[ch_sig]
        cfd10_data.append(cfd10_sig)
        cfd20_data.append(cfd20_sig)
        cfd30_data.append(cfd30_sig)
        w_data[j].extend(w_sig)
        t_data[j].extend(t_sig)
        ev_true_count += 1
        if ev_true_count >= num_curves: 
          print(f"{num_curves} values/curves")
          break

  t_data = [np.array(bias) for bias in t_data]
  w_data = [np.array(bias) for bias in w_data]

  colours = ['black','dodgerblue']
  #colours = ['black','blue']
  alphas = [1.0,0.8]
  edges = ['black','none']
  labels = ["W5 new (140 V)","W13 35e14 (420 V) G = 35"]
  #labels = ["W16 new (210 V)","W12 15e14 (210 V) G = 4"]
  #labels = ["W17 new (160 V)","W12 15e14 (210 V) G = 4"]

  a_max = []
  a_para = []
  a_gaus = []
  para_mae = []
  para_rmse = []
  gaus_mae = []
  gaus_rmse = []

  make_plots = False
  make_populated_plots = False
  cfd_studies = True

  if make_plots:
    plt.figure(figsize=(10, 6))

  for i in range(1):
    #if i == 0: continue
    time_data = t_data[i]*(10**9)
    print(len(time_data))
    

    reshaped_time_data = time_data.reshape(num_curves,502)
    reshaped_ampl_data = w_data[i].reshape(num_curves,502)
    reshaped_perfect_time = []
    reshaped_perfect_ampl = []

    for j in range(num_curves):

      pmax = reshaped_ampl_data[j].max()
      peak_idx = np.argmax(reshaped_ampl_data[j])
      start_idx = max(0, peak_idx - 3)
      end_idx = min(len(reshaped_ampl_data[j]), peak_idx + 4)

      x_peak = reshaped_time_data[j][start_idx:end_idx]
      y_peak = reshaped_ampl_data[j][start_idx:end_idx]

      if len(x_peak) == 0:
        continue

      # parabola
      parabola_coeffs = np.polyfit(x_peak, y_peak, 2)
      a, b, _ = parabola_coeffs
      x_parabola_max = -b / (2 * a)
      y_parabola_max = np.polyval(parabola_coeffs, x_parabola_max)

      y_parabolic_fit = np.polyval(parabola_coeffs, x_peak)
      parabolic_residuals = y_peak - y_parabolic_fit
      parabolic_mae = mean_absolute_error(y_peak, y_parabolic_fit)
      parabolic_rmse = np.sqrt(mean_squared_error(y_peak, y_parabolic_fit))
      parabolic_max_error = np.max(np.abs(parabolic_residuals))

      # gaussian
      p0 = [np.max(y_peak), x_peak[np.argmax(y_peak)], 1]
      try:
        params, _ = opt.curve_fit(gaussian, x_peak, y_peak, p0=p0)
        A, mu, sigma = params

        y_gaussian_fit = gaussian(x_peak, *params)
        gaussian_residuals = y_peak - y_gaussian_fit
        gaussian_mae = mean_absolute_error(y_peak, y_gaussian_fit)
        gaussian_rmse = np.sqrt(mean_squared_error(y_peak, y_gaussian_fit))
        gaussian_max_error = np.max(np.abs(gaussian_residuals))
      except RuntimeError:
        continue

      a_max.append(1000*pmax)
      a_para.append(1000*y_parabola_max)
      a_gaus.append(1000*gaussian(mu, *params))
      para_mae.append(parabolic_mae)
      para_rmse.append(parabolic_rmse)
      gaus_mae.append(gaussian_mae)
      gaus_rmse.append(gaussian_rmse)

      reshaped_perfect_time.append(reshaped_time_data[j])
      reshaped_perfect_ampl.append(reshaped_ampl_data[j])

      if make_plots:
        x_linspace = np.linspace(x_peak.min(), x_peak.max(), 400)

        scatter = plt.scatter(reshaped_time_data[j],reshaped_ampl_data[j],s=10,c=colours[i],marker='o',edgecolor=edges[i],linewidth=0.5,alpha=alphas[i],label=labels[i]+" "+str(round(1000*pmax, 2))+" mV")
        plt.plot(x_linspace, np.polyval(parabola_coeffs, x_linspace), 'r--', label="Parabolic Fit", linewidth = 2)
        plt.plot(x_linspace, gaussian(x_linspace, *params), 'g-', label="Gaussian Fit", linewidth = 2)
        plt.axhline(y_parabola_max, color='r', linestyle=':', label=r"A$_{para}$: "+str(round(1000*y_parabola_max, 2))+" mV", linewidth = 2)
        plt.axhline(gaussian(mu, *params), color='g', linestyle=':', label=r"A$_{Gaus}$: "+str(round(1000*gaussian(mu, *params), 2))+" mV", linewidth = 2)
        #plt.plot(reshaped_time_data[0],reshaped_ampl_data[0],linestyle='-',color=colours[i],linewidth=1.5,alpha=alphas[i],label=labels[i])
        #for j in range(len(reshaped_time_data)-1):
        #  plt.plot(reshaped_time_data[j+1],reshaped_ampl_data[j+1],linestyle='-',color=colours[i],linewidth=1.5,alpha=alphas[i])
        print(f"A_max = {(1000*pmax):.2f} mV")
        print(f"A_para = {(1000*y_parabola_max):.2f} mV")
        print(f"A_Gaus = {(1000*gaussian(mu, *params)):.2f} mV")

        print("\nParabolic Fit Errors:")
        print(f"  MAE  = {parabolic_mae:.4f}")
        print(f"  RMSE = {parabolic_rmse:.4f}")
        print(f"  Max Error = {parabolic_max_error:.4f}")

        print("\nGaussian Fit Errors:")
        print(f"  MAE  = {gaussian_mae:.4f}")
        print(f"  RMSE = {gaussian_rmse:.4f}")
        print(f"  Max Error = {gaussian_max_error:.4f}")

    if make_plots:
      plt.xlabel('Time [ns]',fontsize=14)
      plt.ylabel('Amplitude [mV]',fontsize=14)
      plt.xticks(fontsize=14)
      plt.yticks(fontsize=14)
      plt.xlim(-2,2)
      plt.legend(fontsize=12)
      #plt.yscale('log')
      plt.grid(True, linestyle='--', alpha=0.5)
      plt.tight_layout()
      plt.savefig("./amplitude_analysis.png",dpi=300,facecolor='w')
      #plt.show()
      plt.clf()

  if make_populated_plots:
    fig, axes = plt.subplots(2, 3, figsize=(18, 9), gridspec_kw={'height_ratios': [3, 1]})
    axes[0, 0].scatter(a_max, a_para, c='r', label='Parabola', marker='o', s=20, edgecolors='black')
    axes[0, 0].scatter(a_max, a_gaus, c='g', label='Gaussian', marker='D', s=20, edgecolors='black')
    m_para, b_para = np.polyfit(a_max, a_para, 1)
    m_gaus, b_gaus = np.polyfit(a_max, a_gaus, 1)
    axes[0, 0].legend([f'Parabola (Gradient = {m_para:.3f})', f'Gaussian (Gradient = {m_gaus:.3f})'], fontsize=14)
    axes[0, 0].set_xlabel(r"A$_{max}$ / mV", fontsize=14)
    axes[0, 0].set_ylabel(r"A$_{fit}$ / mV", fontsize=14)
    axes[0, 0].set_xlim(0, None)
    axes[0, 0].set_ylim(0, None)

    ratio_gaus = np.array(a_gaus) / np.array(a_max)
    ratio_para = np.array(a_para) / np.array(a_max)
    axes[1, 0].scatter(a_max, ratio_para, c='r', marker='o', s=20, edgecolors='black', label=r'Parabola / A$_{max}$')
    axes[1, 0].scatter(a_max, ratio_gaus, c='g', marker='D', s=20, edgecolors='black', label=r'Gaussian / A$_{max}$')
    axes[1, 0].axhline(1, color='black', linestyle='dashed', linewidth=1)
    axes[1, 0].set_xlabel(r"A$_{max}$ / mV", fontsize=14)
    axes[1, 0].set_ylabel("Fit / A$_{max}$", fontsize=14)
    axes[1, 0].set_ylim(0.9, 1.1)
    axes[1, 0].legend(fontsize=10)

    axes[0, 1].scatter(para_mae,gaus_mae,c='blue',label=r'Mean absolute error',marker='o',s=20,edgecolors='black')
    axes[0, 1].set_xlabel(r"MAE$_{para}$",fontsize=14)
    axes[0, 1].set_ylabel(r"MAE$_{Gaus}$",fontsize=14)
    axes[0, 1].set_xscale("log")
    axes[0, 1].set_yscale("log")
    axes[0, 1].legend(fontsize=14)

    axes[0, 2].scatter(para_rmse,gaus_rmse,c='blue',label=r'Root mean square error',marker='o',s=20,edgecolors='black')
    axes[0, 2].set_xlabel(r"RMSE$_{para}$",fontsize=14)
    axes[0, 2].set_ylabel(r"RMSE$_{Gaus}$",fontsize=14)
    axes[0, 2].set_xscale("log")
    axes[0, 2].set_yscale("log")
    axes[0, 2].legend(fontsize=14)

    x_min = min(np.array(para_mae).min(), np.array(para_rmse).min())
    x_max = max(np.array(para_mae).max(), np.array(para_rmse).max())
    y_min = min(np.array(gaus_mae).min(), np.array(gaus_rmse).min())
    y_max = max(np.array(gaus_mae).max(), np.array(gaus_rmse).max())
    axes[0, 1].set_xlim(x_min, x_max)
    axes[0, 2].set_xlim(x_min, x_max)
    axes[0, 1].set_ylim(y_min, y_max)
    axes[0, 2].set_ylim(y_min, y_max)
    axes[1, 1].axis("off")
    axes[1, 2].axis("off")

    fig.suptitle(f"Total {len(a_max)} signal events", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("./amplitude_big_data_analysis.png",dpi=300,facecolor='w')
    plt.clf()

  if cfd_studies:

    cfd20_para = []
    cfd20_gaus = []

    for j in range(num_curves):

      idx_para = np.where(np.array(reshaped_ampl_data[j]) > 0.0002*a_para[j])[0][0]
      idx_gaus = np.where(np.array(reshaped_ampl_data[j]) > 0.0002*a_gaus[j])[0][0]
      time_array_event = np.array(reshaped_time_data[j])
      mean_time_para = np.mean(time_array_event[[idx_para-1, idx_para]])
      mean_time_gaus = np.mean(time_array_event[[idx_gaus-1, idx_gaus]])
      cfd20_para.append(mean_time_para)
      cfd20_gaus.append(mean_time_gaus)

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    rms_diff_para = np.sqrt(np.mean((np.array(cfd20_data) - np.array(cfd20_para)) ** 2)).round(3)
    print(rms_diff_para)
    axes[0].hist(cfd20_data, bins=180,range=(-6,3),color='gray',edgecolor='black',label=r"CFD@20% (A$_{max}$)")
    axes[0].hist(cfd20_para, bins=180,range=(-6,3),color='g',edgecolor='black',alpha=0.4,label=r"CFD@20% (A$_{para}$)")
    axes[0].set_xlabel(r"CFD@20% / ns",fontsize=14)
    axes[0].set_ylabel(r"Events",fontsize=14)
    axes[0].set_xlim(-3, 1)
    axes[0].legend(fontsize=14)
    axes[0].grid(True, axis='both', linestyle='--', alpha=0.5)

    rms_diff_gaus = np.sqrt(np.mean((np.array(cfd20_data) - np.array(cfd20_gaus)) ** 2)).round(3)
    print(rms_diff_gaus)
    axes[1].hist(cfd20_data, bins=180,range=(-6,3),color='gray',edgecolor='black',label=r"CFD@20% (A$_{max}$)")
    axes[1].hist(cfd20_gaus, bins=180,range=(-6,3),color='r',edgecolor='black',alpha=0.4,label=r"CFD@20% (A$_{Gaus}$)")
    axes[1].set_xlabel(r"CFD@20% / ns",fontsize=14)
    axes[1].set_ylabel(r"Events",fontsize=14)
    axes[1].set_xlim(-3, 1)
    axes[1].legend(fontsize=14)
    axes[1].grid(True, axis='both', linestyle='--', alpha=0.5)

    fig.suptitle(f"Total {len(a_max)} signal events", fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig("./cfd_20_260325.png",dpi=300,facecolor='w')
    plt.clf()




if __name__ == "__main__":
  main()
