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

def direct_to_table(name_and_df_couples, channel_configs, output_savename, thickness_info, atq_info):

  number_of_duts = sum(1 for element in channel_configs if element[0] == 1)
  number_of_bias_pts = int(len(name_and_df_couples[0][1]) / number_of_duts)
  thickness_info = list(map(int, thickness_info))
  # thickness less 2 um to get active thickness vs nominal thickness of substrate
  thickness_col = np.repeat(np.array(thickness_info), number_of_bias_pts)
  atq_col = np.repeat(np.array(atq_info), number_of_bias_pts)

  area_low = []
  area_high = []
  pmax_low = []
  pmax_high = []
  tmax_low = []
  tmax_high = []
  nmax_low = []

  area_low_mcp = []
  area_high_mcp = []
  pmax_low_mcp = []
  pmax_high_mcp = []
  tmax_low_mcp = []
  tmax_high_mcp = []
  nmax_low_mcp = []

  mcp_channel = False
  for i, (T, _, (A, B, C, D, E, F, G)) in enumerate(channel_configs):
    if T == 1 or T == 2:
      if (A == 0.0) or (A == []):
        area_low_col_ch_i = np.full(number_of_bias_pts, 0.0, dtype=float)
      else:
        area_low_col_ch_i = A
      area_high_col_ch_i = np.full(number_of_bias_pts, B, dtype=float)
      if (C == 0.0) or (C == []):
        pmax_low_col_ch_i = np.full(number_of_bias_pts, 0.0, dtype=float)
      else:
        pmax_low_col_ch_i = C
      pmax_high_col_ch_i = np.full(number_of_bias_pts, D, dtype=float)
      if (E == 0.0) or (E == []):
        tmax_low_col_ch_i = np.full(number_of_bias_pts, -50.0, dtype=float)
      else:
        tmax_low_col_ch_i = E
      if (F == 0.0) or (F == []):
        tmax_high_col_ch_i = np.full(number_of_bias_pts, 50.0, dtype=float)
      else:
        tmax_high_col_ch_i = F
      if G == 0.0:
        nmax_low_col_ch_i = np.full(number_of_bias_pts, -100.0, dtype=float)
      else:
        nmax_low_col_ch_i = np.full(number_of_bias_pts, G, dtype=float)
      if T == 1:
        area_low.append(area_low_col_ch_i)
        area_high.append(area_high_col_ch_i)
        pmax_low.append(pmax_low_col_ch_i)
        pmax_high.append(pmax_high_col_ch_i)
        tmax_low.append(tmax_low_col_ch_i)
        tmax_high.append(tmax_high_col_ch_i)
        nmax_low.append(nmax_low_col_ch_i)
      if T == 2:
        mcp_channel = True
        area_low_mcp = area_low_col_ch_i
        area_high_mcp = area_high_col_ch_i
        pmax_low_mcp = pmax_low_col_ch_i
        pmax_high_mcp = pmax_high_col_ch_i
        tmax_low_mcp = tmax_low_col_ch_i
        tmax_high_mcp = tmax_high_col_ch_i
        nmax_low_mcp = nmax_low_col_ch_i

  dfs_to_concat = []
  first_df_found = False
  for index, (var, df) in enumerate(name_and_df_couples):
    if var == "amplitude":
      df_ampl = df[['Channel','Bias','Amplitude MPV']]
      first_df_found = True
      df_ampl.loc[:, 'Amplitude MPV'] = df_ampl['Amplitude MPV'].round(1)
      df_ampl = df_ampl.rename(columns={'Amplitude MPV':'Amplitude / mV'})
      dfs_to_concat.append(df_ampl)
    elif var == "risetime": # risetime between 10% and 90%
      if first_df_found == False:
        first_df_found = True
        df_rt = df[['Channel','Bias','Mean','Sigma','NEvents']]
      else:
        df_rt = df[['Mean','Sigma','NEvents']]
      df_rt.loc[:, 'Mean'] = (1000*df_rt['Mean']).round(0)
      df_rt.loc[:, 'Sigma'] = (1000*df_rt['Sigma']).round(0)
      df_rt = df_rt.rename(columns={'Mean':'Rise time / ps','Sigma':'Rise time Unc / ps'})
      dfs_to_concat.append(df_rt)
    elif var == "area_fitted":
      if first_df_found == False:
        first_df_found = True
        df_area_fitted = df[['Channel','Bias','Area MPV','Landau width','Gaussian sigma','Frac above 1p5 MPV']]
      else:
        df_area_fitted = df[['Area MPV','Landau width','Gaussian sigma','Frac above 1p5 MPV']]
      df_area_fitted.loc[:, 'Area MPV'] = df_area_fitted['Area MPV'].round(3)
      df_area_fitted.loc[:, 'Landau width'] = df_area_fitted['Landau width'].round(3)
      df_area_fitted.loc[:, 'Gaussian sigma'] = df_area_fitted['Gaussian sigma'].round(3)
      df_area_fitted.loc[:, 'Frac above 1p5 MPV'] = df_area_fitted['Frac above 1p5 MPV'].round(3)
      df_area_fitted = df_area_fitted.rename(columns={'Area MPV':'Area / pWb','Landau width':'Landau Cpt Charge','Gaussian sigma':'Gaussian Cpt Charge','Frac above 1p5 MPV':'Frac Charge >1.5xMPV'})
      df_area_fitted['Charge / fC'] = (df_area_fitted['Area / pWb']/atq_col).round(3)
      df_area_fitted['Gain'] = 100*(df_area_fitted['Charge / fC'] / thickness_col).round(3)
      dfs_to_concat.append(df_area_fitted)
    elif var == "rms":
      if first_df_found == False:
        first_df_found = True
        df_rms = df[['Channel','Bias','Mean','Sigma','NEvents']]
      else:
        df_rms = df[['Mean','Sigma']]
      df_rms.loc[:, 'Sigma'] = df_rms['Sigma'].round(2)
      df_rms = df_rms.rename(columns={'Mean':'RMS Noise / mV', 'Sigma':'RMS Unc / mV'})
      dfs_to_concat.append(df_rms)
    
    elif var == "timeres":
      if first_df_found == False:
        first_df_found = True
        df_tr = df[['Channel','Bias','Resolution @ 30%','Uncertainty @ 30%','Resolution @ 50%','Uncertainty @ 50%','NEvents_TR']]
      else:
        df_tr = df[['Resolution @ 30%','Uncertainty @ 30%','Resolution @ 50%','Uncertainty @ 50%','NEvents_TR']]
      df_tr = df_tr.rename(columns={'Resolution @ 30%':'TR @ 30% / ps', 'Uncertainty @ 30%':'TR Unc @ 30% / ps', 'Resolution @ 50%':'TR @ 50% / ps', 'Uncertainty @ 50%':'TR Unc @ 50% / ps', 'NEvents_TR':'N.Ev. [DUT:MCP]'})
      dfs_to_concat.append(df_tr)
  
  dfs_comb = pd.concat(dfs_to_concat, axis=1)
  dfs_comb.loc[:, 'Bias'] = dfs_comb['Bias'].str[:-1].astype(int)
  dfs_comb['Thickness / um'] = thickness_col
  dfs_comb['E field / V/cm'] = 10000*(dfs_comb['Bias'] / dfs_comb['Thickness / um'])
  dfs_comb.loc[:, 'E field / V/cm'] = dfs_comb['E field / V/cm'] // 1
  dfs_comb = dfs_comb.rename(columns={'NEvents':'N.Ev. [DUT]'})
  if ('Rise time / ps' in dfs_comb.columns) and ('Amplitude / mV' in dfs_comb.columns) and ('RMS Noise / mV' in dfs_comb.columns):
    dfs_comb['Jitter / ps'] = dfs_comb['RMS Noise / mV'] / (dfs_comb['Amplitude / mV'] / dfs_comb['Rise time / ps'])
    dfs_comb.loc[:, 'Jitter / ps'] = dfs_comb['Jitter / ps'].round(1)
    unc_cpt_rms = dfs_comb['RMS Unc / mV'] / dfs_comb['RMS Noise / mV']
    unc_cpt_risetime = dfs_comb['Rise time Unc / ps'] / dfs_comb['Rise time / ps']
    unc_cpt_ampl = 0 # idk the unc for a Langaus fit
    dfs_comb['Jitter Unc / ps'] = dfs_comb['Jitter / ps'] * np.sqrt(unc_cpt_rms**2 + unc_cpt_risetime**2 + unc_cpt_ampl**2)
    dfs_comb.loc[:, 'Jitter Unc / ps'] = dfs_comb['Jitter Unc / ps'].round(1)
    if 'TR @ 30% / ps' in dfs_comb.columns:
      # Fit between charge and time res calculation
      dfs_comb['Landau TR Cpt / ps'], dfs_comb['Landau TR Unc / ps'] = landau_tr_quad_fit(dfs_comb['Charge / fC'], dfs_comb['TR @ 30% / ps'], dfs_comb['TR Unc @ 30% / ps'])
      # Direct quad difference calculation
      #dfs_comb['Landau TR Cpt / ps'] = np.sqrt(dfs_comb['TR @ 30% / ps']**2 - dfs_comb['Jitter[20%:80%] / ps']**2)
      #unc_cpt_jit = dfs_comb['Jitter[20%:80%] / ps']*dfs_comb['Jitter[20%:80%] Unc / ps']
      #unc_cpt_tr = dfs_comb['TR @ 30% / ps']*dfs_comb['TR Unc @ 30% / ps']
      #dfs_comb['Landau TR Unc / ps'] = np.sqrt(unc_cpt_jit**2 + unc_cpt_tr**2) / dfs_comb['Landau TR Cpt / ps']
      dfs_comb.loc[:, 'Landau TR Cpt / ps'] = dfs_comb['Landau TR Cpt / ps'].round(1)
      dfs_comb.loc[:, 'Landau TR Unc / ps'] = dfs_comb['Landau TR Unc / ps'].round(1)
      dfs_comb['WF6 Param / ps/um'] = dfs_comb['Landau TR Cpt / ps'] / dfs_comb['Thickness / um']
      dfs_comb['WF6 Param Unc / ps/um'] = dfs_comb['WF6 Param / ps/um'] * dfs_comb['Landau TR Unc / ps'] / dfs_comb['Landau TR Cpt / ps']
      dfs_comb.loc[:, 'WF6 Param / ps/um'] = dfs_comb['WF6 Param / ps/um'].round(2)
      dfs_comb.loc[:, 'WF6 Param Unc / ps/um'] = dfs_comb['WF6 Param Unc / ps/um'].round(2)

  columns = dfs_comb.columns.tolist()
  for col_to_move in ['Thickness / um','E field / V/cm']:
    columns.remove(col_to_move)
  columns[2:2] = ['Thickness / um','E field / V/cm']
  dfs_comb = dfs_comb[columns]

  dfs_comb['Bias'] = pd.to_numeric(dfs_comb['Bias'], errors='coerce')
  dfs_comb = dfs_comb.sort_values(by=['Channel','Bias'])

  dfs_comb['AREA low / mV'] = np.ravel(area_low)
  dfs_comb['AREA high / mV'] = np.ravel(area_high)
  dfs_comb['PMAX low / mV'] = np.ravel(pmax_low)
  dfs_comb['PMAX high / mV'] = np.ravel(pmax_high)
  dfs_comb['TMAX low / ns'] = np.ravel(tmax_low)
  dfs_comb['TMAX high / ns'] = np.ravel(tmax_high)
  dfs_comb['NMAX low / ns'] = np.ravel(nmax_low)

  if mcp_channel == True:
    dfs_comb['MCP AREA low / mV'] = np.tile(area_low_mcp, number_of_duts)
    dfs_comb['MCP AREA high / mV'] = np.tile(area_high_mcp, number_of_duts)
    dfs_comb['MCP PMAX low / mV'] = np.tile(pmax_low_mcp, number_of_duts)
    dfs_comb['MCP PMAX high / mV'] = np.tile(pmax_high_mcp, number_of_duts)
    dfs_comb['MCP TMAX low / ns'] = np.tile(tmax_low_mcp, number_of_duts)
    dfs_comb['MCP TMAX high / ns'] = np.tile(tmax_high_mcp, number_of_duts)
    dfs_comb['MCP NMAX low / ns'] = np.tile(nmax_low_mcp, number_of_duts)


  print(f"[BETA ANALYSIS] : [DATA COLLATOR] Writing data to {output_savename}.csv.")
  dfs_comb.to_csv(output_savename+'.csv', index=False)
