# proc_tools.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import poisson
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
import sys

'''
from proc_tools_TR import get_fit_results_TR, hist_tree_file_basics, plot_fit_curves, getBias
'''

root.gErrorIgnoreLevel = root.kWarning
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def round_to_sig_figs(x, sig):
  if x == 0:
    return 0
  else:
    return round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)

def get_fit_results_TR(arr_of_fits,arr_of_biases,dut_channels,mcp_channel, simplified=False):
  arr_of_ch = []
  arr_of_biases_fitted = []
  arr_of_mean = []
  arr_of_sigma = []
  arr_down_var = []
  arr_up_var = []
  arr_of_ampl = []
  arr_of_red_chi2 = []

  for channel_i, fit_func in enumerate(arr_of_fits):

    mean = fit_func[0].GetParameter(1)  # Mean of the gauss distribution
    sigma = fit_func[0].GetParameter(2) # Sigma of the gauss distribution
    var_down = fit_func[1].GetParameter(2)
    var_up = fit_func[2].GetParameter(2)
    amplitude = fit_func[0].GetParameter(0)  # Amplitude of the gauss distribution
    chi2 = fit_func[0].GetChisquare()  # Chi-squared value of the fit
    ndf = fit_func[0].GetNDF()  # Number of degrees of freedom
    arr_of_mean.append(round_to_sig_figs(mean,4))
    arr_of_sigma.append(sigma)
    arr_down_var.append(var_down)
    arr_up_var.append(var_up)
    arr_of_ampl.append(round_to_sig_figs(amplitude,3))
    arr_of_ch.append("Ch" + str(dut_channels[channel_i]))
    arr_of_biases_fitted.append(arr_of_biases[channel_i])
    if ndf == 0:
      ndf = 1
    arr_of_red_chi2.append(round_to_sig_figs((chi2/ndf),3))

  df_of_results = pd.DataFrame({
    "Channel": arr_of_ch,
    "Bias": arr_of_biases_fitted,
    "Mean": arr_of_mean,
    "Sigma": arr_of_sigma,
    "Amplitude": arr_of_ampl,
    "RChi2": arr_of_red_chi2
  })

  if mcp_channel:
    sig_dut_values = []
    sig_dut_errors = []
    mcp_tr = 0.005
    mcp_tr_err = 0.002
    print(f"[BETA ANALYSIS]: [TIME RESOLUTION] Calculating time resolution for DUT, assuming MCP time resolution {mcp_tr*1000} +/- {mcp_tr_err*1000} ps")
    for ch_ind, ch_val in enumerate(arr_of_sigma):
      sig1 = np.sqrt(ch_val**2 - mcp_tr**2)
      unc_sig1_nominal = np.abs(arr_up_var[ch_ind] - arr_down_var[ch_ind]) / 2
      sig1err = np.sqrt((ch_val*unc_sig1_nominal)**2 + (mcp_tr*mcp_tr_err)**2)/sig1
      sig_dut_values.append(round_to_sig_figs(1000*sig1,3))
      sig_dut_errors.append(round_to_sig_figs(1000*sig1err,2))
    df_of_results['Resolution'] = sig_dut_values
    df_of_results['Uncertainty'] = sig_dut_errors
  else:
    sig1 = np.sqrt(0.5*(arr_of_sigma[0]**2 + arr_of_sigma[2]**2 - arr_of_sigma[1]**2))
    sig2 = np.sqrt(0.5*(arr_of_sigma[0]**2 + arr_of_sigma[1]**2 - arr_of_sigma[2]**2))
    sig3 = np.sqrt(0.5*(arr_of_sigma[1]**2 + arr_of_sigma[2]**2 - arr_of_sigma[0]**2))
    df_of_results['Sigma_cpt'] = ["sigma_1","sigma_2","sigma_3"]
    df_of_results['Sigma_value'] = [sig1,sig2,sig3]

  if simplified:
    return df_of_results[["Bias","Sigma_cpt","Sigma_value"]]

  else:
    return df_of_results

def hist_tree_file_timeres(tree,file,var,ch,nBins,xLower,xUpper,biasVal,cut_cond):
  thisHist = root.TH1F("CH "+str(ch)+" "+biasVal, var+";tn-tn+1 / ns ;Events", nBins, xLower, xUpper)
  tree.Draw(var+">>CH "+str(ch)+" "+biasVal,cut_cond)
  thisHist.SetLineWidth(2)
  thisHist.SetLineColor(ch+1)
  return thisHist

def plot_fit_curves(xLower,xUpper,fit_type,hist_to_fit,channel_index,biasVal):
  thisFit = TF1(fit_type+"_hist"+biasVal+" CH "+str(channel_index+1), fit_type, xLower, xUpper)
  hist_to_fit.Fit(thisFit, "Q")
  thisFit.SetLineWidth(3)
  thisFit.SetLineColor(channel_index+1)
  #thisFit.SetLineStyle(2)
  return thisFit
