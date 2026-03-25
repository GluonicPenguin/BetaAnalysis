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

from proc_tools import getBias, landau_tr_quad_fit

def convert_and_save_csv(data, savename):
  flat_dict = {}
  for df in data:
        # Identify variable name automatically
        var_cols = [c for c in df.columns if c not in ["BIAS"]]
        # There should be exactly two: X and EVENTS
        x_cols = [c for c in var_cols if not c.endswith("_EVENTS")]
        if not x_cols:
            continue
        var = x_cols[0]
        y_col = f"{var}_EVENTS"

        # Only include X once per variable (first row is representative)
        if var not in flat_dict:
            flat_dict[var] = list(df.iloc[0][var])

        # Add all bias-dependent EVENTS columns
        for _, row in df.iterrows():
            bias = row["BIAS"]
            col_name = f"{y_col}_{bias}"
            flat_dict[col_name] = list(row[y_col])

  flat_df = pd.DataFrame(flat_dict)
  flat_df.to_csv(savename, index=False)

def direct_to_table(name_and_df_couples, channel_configs, output_savename, thickness_info):

  number_of_duts = sum(1 for element in channel_configs if element[0] == 1)
  number_of_bias_pts = int(len(name_and_df_couples[0][1]) / number_of_duts)
  thickness_info = list(map(int, thickness_info))
  # thickness less 2 um to get active thickness vs nominal thickness of substrate
  thickness_col = np.repeat(np.array(thickness_info) - 2, number_of_bias_pts)

  pmax_low = []
  pmax_high = []

  pmax_low_mcp = []
  pmax_high_mcp = []

  mcp_channel = False
  for i, (T, _, (A, B)) in enumerate(channel_configs):
    if T == 1 or T == 2:
      if (A == 0.0) or (A == []):
        pmax_low_col_ch_i = np.full(number_of_bias_pts, 0.0, dtype=float)
      else:
        pmax_low_col_ch_i = A
      pmax_high_col_ch_i = np.full(number_of_bias_pts, B, dtype=float)
      if T == 1:
        pmax_low.append(pmax_low_col_ch_i)
        pmax_high.append(pmax_high_col_ch_i)
      if T == 2:
        mcp_channel = True
        pmax_low_mcp = pmax_low_col_ch_i
        pmax_high_mcp = pmax_high_col_ch_i

  dfs_to_concat = []
  first_df_found = False
  for index, (var, df) in enumerate(name_and_df_couples):
    if var == "amplitude":
      df_ampl = df[['Channel','Bias','Amplitude','Amplitude Unc','Landau width','Gaussian sigma','LTF','LTF Unc','LTF from area','LTFmax','LTFmax Unc','Landau Frac','Landau Frac Unc']]
      first_df_found = True
      #df_ampl.loc[:, 'Amplitude'] = df_ampl['Amplitude'].round(1)
      df_ampl = df_ampl.rename(columns={'Amplitude':'Amplitude / mV','Amplitude Unc': 'A_Unc','Landau width':'A_Landau','Gaussian sigma':'A_Gaus','LTF':'A_LTF','LTF Unc':'A_LTF_unc',
                                        'LTF from area':'A_LTF_area','LTFmax':'A_LTF_max','LTFmax Unc':'A_LTF_max_unc','Landau Frac':'A_xiompv','Landau Frac Unc':'A_xiompv_unc'})
      dfs_to_concat.append(df_ampl)
    elif var == "charge":
      if first_df_found == False:
        first_df_found = True
        df_charge = df[['Channel','Bias','Charge','Charge Unc','Landau width','Gaussian sigma','LTF','LTF Unc','LTF from area','LTFmax','LTFmax Unc','Landau Frac','Landau Frac Unc']]
      else:
        df_charge = df[['Charge','Charge Unc','Landau width','Gaussian sigma','LTF','LTF Unc','LTFmax','LTFmax Unc','Landau Frac','Landau Frac Unc']]
      #df_charge.loc[:, 'Charge'] = df_charge['Charge'].round(1)
      #df_charge.loc[:, 'Landau width'] = df_charge['Landau width'].round(3)
      #df_charge.loc[:, 'Gaussian sigma'] = df_charge['Gaussian sigma'].round(3)
      #df_charge.loc[:, 'LTF'] = df_charge['LTF'].round(3)
      #df_charge.loc[:, 'LTFmax'] = df_charge['LTFmax'].round(3)
      df_charge = df_charge.rename(columns={'Charge':'Charge / fC','Charge Unc':'Q_Unc','Landau width':'Q_Landau','Gaussian sigma':'Q_Gaus','LTF':'Q_LTF','LTF Unc':'Q_LTF_unc',
                                            'LTF from area':'Q_LTF_area','LTFmax':'Q_LTF_max','LTFmax Unc':'Q_LTF_max_unc','Landau Frac':'Q_xiompv','Landau Frac Unc':'Q_xiompv_unc'})
      #df_charge['Gain'] = 100*(df_charge['Charge / fC'] / thickness_col).round(2)
      dfs_to_concat.append(df_charge)
    elif var == "gain":
      if first_df_found == False:
        first_df_found = True
        df_gain = df[['Channel','Bias','Gain','Gain Unc','Landau width','Gaussian sigma','LTF','LTF Unc','LTF from area','LTFmax','LTFmax Unc','Landau Frac','Landau Frac Unc']]
      else:
        df_gain = df[['Gain','Gain Unc','Landau width','Gaussian sigma','LTF','LTF Unc','LTF from area','LTFmax','LTFmax Unc','Landau Frac','Landau Frac Unc']]
      #df_gain.loc[:, 'Gain'] = df_gain['Gain'].round(1)
      #df_gain.loc[:, 'Landau width'] = df_gain['Landau width'].round(3)
      #df_gain.loc[:, 'Gaussian sigma'] = df_gain['Gaussian sigma'].round(3)
      #df_gain.loc[:, 'LTF'] = df_gain['LTF'].round(3)
      #df_gain.loc[:, 'LTFmax'] = df_gain['LTFmax'].round(3)
      df_gain = df_gain.rename(columns={'Gain Unc':'G_Unc','Landau width':'G_Landau','Gaussian sigma':'G_Gaus','LTF':'G_LTF','LTF Unc':'G_LTF_unc','LTF from area':'G_LTF_area',
                                        'LTFmax':'G_LTF_max','LTFmax Unc':'G_LTF_max_unc','Landau Frac':'G_xiompv','Landau Frac Unc':'G_xiompv_unc'})
      dfs_to_concat.append(df_gain)
  
  dfs_comb = pd.concat(dfs_to_concat, axis=1)
  dfs_comb.loc[:, 'Bias'] = dfs_comb['Bias'].str[:-1].astype(int)
  #thickness_col = thickness_col[-len(dfs_comb['Bias']):]
  dfs_comb['Thickness / um'] = thickness_col
  dfs_comb['E field / kV/cm'] = 10*(dfs_comb['Bias'] / dfs_comb['Thickness / um'])
  dfs_comb.loc[:, 'E field / kV/cm'] = dfs_comb['E field / kV/cm'] // 1
  dfs_comb = dfs_comb.rename(columns={'NEvents':'N.Ev. [DUT]'})

  columns = dfs_comb.columns.tolist()
  for col_to_move in ['Thickness / um','E field / kV/cm']:
    columns.remove(col_to_move)
  columns[2:2] = ['Thickness / um','E field / kV/cm']
  dfs_comb = dfs_comb[columns]

  dfs_comb['Bias'] = pd.to_numeric(dfs_comb['Bias'], errors='coerce')
  dfs_comb = dfs_comb.sort_values(by=['Channel','Bias'])

  dfs_comb['PMAX low / mV'] = np.ravel(pmax_low)
  dfs_comb['PMAX high / mV'] = np.ravel(pmax_high)
  if mcp_channel == True:
    dfs_comb['MCP PMAX low / mV'] = np.tile(pmax_low_mcp, number_of_duts)
    dfs_comb['MCP PMAX high / mV'] = np.tile(pmax_high_mcp, number_of_duts)

  print(f"[BETA ANALYSIS] : [DATA COLLATOR] Writing data to {output_savename}.csv.")
  dfs_comb.to_csv(output_savename+'.csv', index=False)
