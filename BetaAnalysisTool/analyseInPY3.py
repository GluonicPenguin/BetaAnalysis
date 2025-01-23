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

from classPlotter import plotVar
from classTRPlotter import plotTRVar
from cardReader import read_text_card
from langaus import plot_langaus

var_dict = {"tmax":"t_{max} / 10 ns" , "pmax":"p_max / mV" , "negpmax":"-p_max / mV", "charge":"Q / fC", "area_new":"Area / pWb" , "rms":"RMS / mV"}

def main():
  parser = argparse.ArgumentParser(description='Read a text card containing information and location of ROOT analysis files and plot distributions of corresponding variables.')
  parser.add_argument('config', type=str, help='Path to the configuration text card (e.g., config.txt)')
  args = parser.parse_args()

  print(f"[BETA ANALYSIS] : [CARD READER] Reading text card {args.config}.")
  config = read_text_card(args.config)
  output_name = os.path.splitext(os.path.basename(args.config))[0]

  file_list = config.get('files', [])
  channels = config.get('channels', [[0, 1]] * 8)

  print(f"[BETA ANALYSIS] : [FILE READER] Reading files {file_list}.")

  if config.get('pmax', False):
    pmax_params = config.get('pmax_params', None)
  if config.get('negpmax', False):
    negpmax_params = config.get('negpmax_params', None)
  if config.get('charge', False):
    charge_params = config.get('charge_params', None)
  if config.get('rms', False):
    rms_params = config.get('rms_params', None)
  if config.get('timeres', False):
    timeres_params = config.get('timeres_params', None)

  filtered_channels = [ch for ch in config['channels'] if ch[0] != 0] # drop channels that are not specified in the textCard
  modified_channels = [[ch[0], ch[1], (ch[2][0], 1000 if ch[2][1] == 0 else ch[2][1], -100 if ch[2][2] >= 0 else ch[2][2], -50 if ch[2][3] == 0 else ch[2][3], 50 if ch[2][4] == 0 else ch[2][4])] for ch in filtered_channels] # set +ve negpmax lower bounds to -100 mV if they are +ve or 0 in the textCard (and tmax cuts on full range if specified as 0,0)
  config['channels'] = modified_channels

  channel_mapping = {
    1: "DUT",
    2: "MCP",
    3: "Reference sensor",
    0: "Unknown"
  }

  board_mapping = {
    4.7: "on SC board",
    5: "on Mignone board",
    1: "unmounted"
  }

  file_array = []
  tree_array = []
  output_name_array = []

  for pattern in file_list:
    root_files = glob.glob(pattern)
    output_name_const = output_name
    for root_file in root_files:
      try:
        theFile = root.TFile(root_file)
        file_array.append(theFile)
        tree_array.append(theFile.Get("Analysis"))
        output_name_w_bias = ""
        for ch_ind, ch_val in enumerate(config['channels']):
          if ch_val[0] == 1:
            bias_after_channel = re.search(rf"Ch{ch_ind}-(\d+)V_", pattern)
            output_name_w_bias = output_name_w_bias + f"_Ch{ch_ind}-" + bias_after_channel.group(1) +"V"
        output_name_const = "hist_" + output_name_const + output_name_w_bias
        output_name_array.append(output_name_const)
      except Exception as e:
        print(f"Error reading {root_file}: {e}")

  if len(file_array) == 0:
    print(f"[BETA ANALYSIS] : [FILE READER] No files found.")
    sys.exit(0)
  else:
    print(f"[BETA ANALYSIS] : [FILE READER] Total {len(file_array)} input ROOT files read.")

  plot_variables = [var for var, flag in config.items() if var in ['tmax', 'pmax', 'negpmax', 'charge', 'rms', 'timeres', 'discretisation', 'waveform'] and flag]  

  if plot_variables:
    sentence = "will plot " + ", ".join(plot_variables)
  else:
    sentence = "will perform analysis without plotting."

  print(f"[BETA ANALYSIS] : [CARD READER] Analyser " + sentence + " distributions for the following setup:")
  for i, ch in enumerate(config['channels']):
    print(f"        CH {i} : {channel_mapping.get(ch[0])} {board_mapping.get(ch[1])}")

  if config.get('tmax', False) == True:
    for file_ind, file_real in enumerate(file_array):
      plot_tmax = plotVar("tmax", 1000, -2, 2, False, output_name_array[file_ind]+"_tmax.png", fit=None)
      plot_tmax.run(file_real, tree_array[file_ind], config['channels'])
  if config.get('pmax', False) == True:
    for file_ind, file_real in enumerate(file_array):
      plot_pmax = plotVar("pmax", pmax_params[0], pmax_params[1], pmax_params[2], True, output_name_array[file_ind]+"_pmax.png", fit=None)
      plot_pmax.run(file_real, tree_array[file_ind], config['channels'])
  if config.get('negpmax', False) == True:
    for file_ind, file_real in enumerate(file_array):
      plot_negpmax = plotVar("negpmax", negpmax_params[0], negpmax_params[1], negpmax_params[2], True, output_name_array[file_ind]+"_negpmax.png", fit=None)
      plot_negpmax.run(file_real, tree_array[file_ind], config['channels'])
  if config.get('charge', False) == True:
    charge_dfs = []
    for file_ind, file_real in enumerate(file_array):
      df_data = plot_langaus(file_real, tree_array[file_ind], config['channels'], charge_params[0], charge_params[1], charge_params[2], output_name_array[file_ind]+"_charge")
      charge_dfs.append(df_data)
    charge_data = pd.concat(charge_dfs, ignore_index=True)
    print(charge_data.sort_values(by='Channel'))
  if config.get('rms', False) == True:
    rms_dfs = []
    for file_ind, file_real in enumerate(file_array):
      plot_rms = plotVar("rms", rms_params[0], rms_params[1], rms_params[2], True, output_name_array[file_ind]+"_rms.png", fit="gaus")
      df_data = plot_rms.run(file_real, tree_array[file_ind], config['channels'])
      rms_dfs.append(df_data)
    rms_data = pd.concat(rms_dfs, ignore_index=True)
    print(rms_data.sort_values(by='Channel'))
  if config.get('timeres', False) == True:
    time_res_dfs = []
    for file_ind, file_real in enumerate(file_array):
      plot_timeres = plotTRVar("timeres", timeres_params[0], timeres_params[1], timeres_params[2], True, output_name_array[file_ind]+"_timeres.png")
      df_data = plot_timeres.run(file_real, tree_array[file_ind], config['channels'])
      time_res_dfs.append(df_data)
    time_res_data = pd.concat(time_res_dfs, ignore_index=True)
    print(time_res_data.sort_values(by='Channel'))

  #if args.doDiscretisation: risingEdgeDiscretisation.run(file_array,tree_array,args.ch-1,total_number_channels)
  #if args.doWaveform: plot_waveform.run(file_array,tree_array,args.ch-1,total_number_channels)
  #if args.csvOut: makeCSV.run(file_array,tree_array,total_number_channels)

if __name__ == "__main__":
    main()
