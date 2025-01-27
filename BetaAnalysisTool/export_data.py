# export_data.py

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

from proc_tools import getBias

def direct_to_table(name_and_df_couples, channel_configs, output_savename):

  number_of_duts = sum(1 for element in channel_configs if element[0] == 1)
  number_of_bias_pts = int(len(name_and_df_couples[0][1]) / number_of_duts)
  pmax_low = []
  pmax_high = []
  nmax_low = []
  tmax_low = []
  tmax_high = []

  pmax_low_mcp = []
  pmax_high_mcp = []
  nmax_low_mcp = []
  tmax_low_mcp = []
  tmax_high_mcp = []

  mcp_channel = False
  for i, (T, _, (A, B, C, D, E)) in enumerate(channel_configs):
    if T == 1 or T == 2:
      if (A == 0.0) or (A == []):
        pmax_low_col_ch_i = np.full(number_of_bias_pts, 0.0, dtype=float)
      else:
        pmax_low_col_ch_i = A
      pmax_high_col_ch_i = np.full(number_of_bias_pts, B, dtype=float)
      nmax_low_col_ch_i = np.full(number_of_bias_pts, C, dtype=float)
      tmax_low_col_ch_i = np.full(number_of_bias_pts, D, dtype=float)
      tmax_high_col_ch_i = np.full(number_of_bias_pts, E, dtype=float)
      if T == 1:
        pmax_low.append(pmax_low_col_ch_i)
        pmax_high.append(pmax_high_col_ch_i)
        nmax_low.append(nmax_low_col_ch_i)
        tmax_low.append(tmax_low_col_ch_i)
        tmax_high.append(tmax_high_col_ch_i)
      if T == 2:
        mcp_channel = True
        pmax_low_mcp = pmax_low_col_ch_i
        pmax_high_mcp = pmax_high_col_ch_i
        nmax_low_mcp = nmax_low_col_ch_i
        tmax_low_mcp = tmax_low_col_ch_i
        tmax_high_mcp = tmax_high_col_ch_i

  dfs_to_concat = []
  first_df_found = False
  for index, (var, df) in enumerate(name_and_df_couples):
    if var == "amplitude":
      df_ampl = df[['Channel','Bias','Amplitude MPV']]
      first_df_found = True
      df_ampl = df_ampl.rename(columns={'Amplitude MPV':'Amplitude / mV'})
      dfs_to_concat.append(df_ampl)
    elif var == "risetime":
      if first_df_found == False:
        first_df_found = True
        df_rt = df[['Channel','Bias','Mean']]
      else:
        df_rt = df[['Mean']]*1000
      df_rt = df_rt.rename(columns={'Mean':'Rise time / ps'})
      dfs_to_concat.append(df_rt)
    elif var == "charge":
      if first_df_found == False:
        first_df_found = True
        df_charge = df[['Channel','Bias','Charge MPV','Frac above 1p5']]
      else:
        df_charge = df[['Charge MPV','Frac above 1p5']]
      df_charge = df_charge.rename(columns={'Charge MPV':'Charge / fC'})
      dfs_to_concat.append(df_charge)
    elif var == "rms":
      if first_df_found == False:
        first_df_found = True
        df_rms = df[['Channel','Bias','Mean']]
      else:
        df_rms = df[['Mean']]
      df_rms = df_rms.rename(columns={'Mean':'RMS Noise / mV'})
      dfs_to_concat.append(df_rms)
    elif var == "timeres":
      if first_df_found == False:
        first_df_found = True
        df_tr = df[['Channel','Bias','Resolution','Uncertainty']]
      else:
        df_tr = df[['Resolution','Uncertainty']]
      df_tr = df_tr.rename(columns={'Resolution':'Time Resolution / ps', 'Uncertainty':'Resolution Unc / ps'})
      dfs_to_concat.append(df_tr)
  
  dfs_comb = pd.concat(dfs_to_concat, axis=1)
  dfs_comb['PMAX low / mV'] = np.ravel(pmax_low)
  dfs_comb['PMAX high / mV'] = np.ravel(pmax_high)
  dfs_comb['NMAX low / mV'] = np.ravel(nmax_low)
  dfs_comb['TMAX low / mV'] = np.ravel(tmax_low)
  dfs_comb['TMAX high / mV'] = np.ravel(tmax_high)

  if mcp_channel == True:
    dfs_comb['MCP PMAX low / mV'] = np.tile(pmax_low_mcp, number_of_duts)
    dfs_comb['MCP PMAX high / mV'] = np.tile(pmax_high_mcp, number_of_duts)
    dfs_comb['MCP NMAX low / mV'] = np.tile(nmax_low_mcp, number_of_duts)
    dfs_comb['MCP TMAX low / mV'] = np.tile(tmax_low_mcp, number_of_duts)
    dfs_comb['MCP TMAX high / mV'] = np.tile(tmax_high_mcp, number_of_duts)

  print(f"[BETA ANALYSIS] : [DATA COLLATOR] Writing data to {output_savename}.csv.")
  dfs_comb.to_csv(output_savename+'.csv', index=False)
