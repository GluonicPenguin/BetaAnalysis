import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.optimize import minimize
from scipy.stats import poisson, median_abs_deviation
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
from scipy.optimize import curve_fit

from firstPass_ProcTools import getBias
from secondPass_ProcTools import binned_fit_gaussian, gaussian

def round_to_sig_figs(x, sig):
  if x == 0:
    return 0
  else:
    return round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)

def plot_gaussian(var, file, file_index, tree, channel_array, nBins, SNDisc_signal_events_bool, savename):

  arr_of_ch = []
  arr_of_biases = []
  arr_of_mean = []
  arr_of_sigma = []
  arr_of_sse = []
  arr_of_rchi2 = []

  SNDisc_Index = 0
  data_filtered_signal_dut_set = []
  data_filtered_signal_mcp = []
  pmax_unselected_mcp = []
  var_dict = {"timeres":"tn-tn+1 / ns"}
  for ch_ind, ch_val in enumerate(channel_array):
    data_filtered_signal_dut = []
    sensorType, AtQfactor, mcp_selection = ch_val
    if sensorType == 1:
      bias_of_channel = getBias(str(file), ch_ind)
      arr_of_biases.append(bias_of_channel)
      for i, entry in enumerate(tree):
        if SNDisc_signal_events_bool[SNDisc_Index][i]:
          sig_event = entry.cfd[ch_ind][2]
          data_filtered_signal_dut.append(sig_event)
      SNDisc_Index += 1
      data_filtered_signal_dut_set.append(data_filtered_signal_dut)
    elif sensorType == 2:
      for i, entry in enumerate(tree):
        if SNDisc_signal_events_bool[SNDisc_Index][i]:
          pmax_unselected_mcp_event = entry.pmax[ch_ind]
          sig_event = entry.cfd[ch_ind][2]
          data_filtered_signal_mcp.append(sig_event)
          pmax_unselected_mcp.append(pmax_unselected_mcp_event)
      SNDisc_Index += 1
      mcp_specs = 
    else:
      continue


  if mcp_exists:
    




    for j in range(len(channel_array)):
      if (channel_array[j][0] == 1):
        bias = getBias(str(file), j)
        arr_of_biases.append(bias)
        duts_to_analyse.append([["cfd["+str(j)+"][2]-cfd[","cfd["+str(j)+"][0]-cfd[","cfd["+str(j)+"][4]-cfd["], result[j], j])
        channel_of_dut.append(j)
      elif (channel_array[j][0] == 2):
        mcp_channel = [[str(j)+"][2]",str(j)+"][0]",str(j)+"][4]"], result[j]]
        mcp_exists = True

    duts_vars_cuts = []
    if (mcp_exists == True) & (mcp_tr != (0, 0)):
      for dut in duts_to_analyse:
        vars_cuts_defined = ([x + y for x, y in zip(dut[0], mcp_channel[0])], dut[1] + " && " + mcp_channel[1], dut[2])
        duts_vars_cuts.append(vars_cuts_defined)
    if (mcp_exists == False) | (mcp_tr == (0, 0)):
      for dut in duts_to_analyse:
        vars_cuts_defined = ([x + y for x, y in zip(dut[0], mcp_channel[0])], dut[1] + " && " + mcp_channel[1], dut[2])
        duts_vars_cuts.append(vars_cuts_defined)
      adapted_dut_channel = [second_plane[4:9] for second_plane in duts_to_analyse[1][0]]
      dual_plane_var = [x + y for x, y in zip(duts_to_analyse[0][0], adapted_dut_channel)]
      dual_plane_duts = [dual_plane_var, duts_to_analyse[0][1]+" && "+duts_to_analyse[1][1], int(mcp_channel[0][1][0])]
      duts_vars_cuts.append(dual_plane_duts)
      arr_of_biases.append("MCP")
      channel_of_dut.append(int(mcp_channel[0][1][0]))

    hists_to_plot = []
    for j in range(len(duts_vars_cuts)):
      hist_down_up_dev = []
      for dut_var_ind, dut_var in enumerate(duts_vars_cuts[j][0]):
        thisHist = hist_tree_file_timeres(tree, file, dut_var, duts_vars_cuts[j][2], self.nBins, -1*self.xUpper, self.xUpper, arr_of_biases[j], duts_vars_cuts[j][1])
        if dut_var_ind == 0:
          hists_to_plot.append(thisHist)
        hist_down_up_dev.append(thisHist)
      arr_of_hists.append(hist_down_up_dev)
      arr_of_biases.append(bias)


































    plt.figure(figsize=(10, 6))
    data_var = np.array(data_filtered_signal)

    popt, sse, rchi2 = binned_fit_gaussian(data_var, nBins)

    arr_of_ch.append("Ch"+str(ch_ind))
    arr_of_biases.append(bias_of_channel)
    arr_of_mean.append(popt[1])
    arr_of_sigma.append(popt[2])
    arr_of_sse.append(sse)
    arr_of_rchi2.append(rchi2)

    fig = go.Figure()
    fig.update_layout(
      xaxis_title=var_dict[var],
      yaxis_title='Probability Density',
      title='Gaussian Fit',
    )

    fig.add_trace(
      go.Histogram(
        x=data_var,
        xbins=dict(start=0, end=2*np.mean(data_var), size=2*np.mean(data_var)/nBins),
        name='Histogram',
        histnorm='probability density',
        #nbinsx=nBins,
        opacity=0.6,
        marker=dict(color='green', line=dict(color='black', width=1)),
      )
    )

    x_axis = np.linspace(0, 2*np.mean(data_var), 999)
    fig.add_trace(
      go.Scatter(
        x=x_axis,
        y=gaussian(x_axis, *popt),
        name=f'Gaussian Fit<br>μ={popt[1]:.2e}<br>σ={popt[2]:.2e}',
        mode='lines',
        line=dict(width=3,dash='dash',color='black'),
      )
    )

    if not os.path.exists(var):
      os.makedirs(var)
    fig.write_image(var+"/"+savename+"_Ch"+str(ch_ind)+".png")
    print("[BETA ANALYSIS]: [GAUSSIAN PLOTTER] Saved file "+var+"/"+savename+"_Ch"+str(ch_ind)+".png")

  df_of_results = pd.DataFrame({
    "Channel": arr_of_ch,
    "Bias": arr_of_biases,
    "Mean": arr_of_mean,
    "Sigma": arr_of_sigma,
    "SSE score": arr_of_sse,
    "Red. Chi2": arr_of_rchi2,
  })
  return df_of_results
