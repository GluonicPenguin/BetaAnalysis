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
from cardReader import read_text_card
from langaus import plot_langaus
from export_data import direct_to_table, convert_and_save_csv

def main():
  parser = argparse.ArgumentParser(description='Read a text card containing information and location of ROOT analysis files and plot distributions of corresponding variables.')
  parser.add_argument('config', type=str, help='Path to the configuration text card (e.g., config.txt)')
  args = parser.parse_args()

  print(f"[BETA ANALYSIS] : [CARD READER] Reading text card {args.config}.")
  config, thicknesses, mcp_specs, safemode = read_text_card(args.config)
  thicknesses = [thickness for thickness in thicknesses if thickness != "nDUT"]
  output_name = os.path.splitext(os.path.basename(args.config))[0]

  file_list = config.get('files', [])
  channels = config.get('channels', [[0, 1]] * 8)

  print(f"[BETA ANALYSIS] : [FILE READER] Reading files {file_list}.")

  if (config.get('pmax', False)) or (config.get('amplitude', False)):
    pmax_params = config.get('pmax_params', None)
  if config.get('charge', False):
    charge_params = config.get('charge_params', None)

  while config['channels'] and config['channels'][-1][0] == 0:
    config['channels'].pop()

  for ch in config['channels']:
    if ch[0] == 0: ch[1] = 0

  channel_mapping = {
    1: "DUT",
    2: "MCP",
    3: "Reference sensor",
    0: "Unknown or unused channel"
  }

  board_mapping = {
    4.7: "on SC board",
    5: "on Mignone board",
    1: "unmounted",
    0: ""
  }

  if safemode:
    root.gROOT.SetBatch(True)

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

  plot_variables = [var for var, flag in config.items() if var in ['pmax', 'amplitude', 'charge', 'gain'] and flag]

  if plot_variables:
    sentence = "will plot " + ", ".join(plot_variables)
  else:
    sentence = "will perform analysis without plotting."

  print(f"[BETA ANALYSIS] : [CARD READER] Analyser " + sentence + " distributions for the following setup:")
  for i, ch in enumerate(config['channels']):
    print(f"        CH {i} : {channel_mapping.get(ch[0])} {board_mapping.get(ch[1])}")

  if not safemode:
    print("\n\n\n\n    ***************************************************************************************************************************************************************************\n")
    print("    [EPILEPSY WARNING] : This plotter can cause rapid imagery to appear on the screen that may trigger seizures or other symptoms in individuals with photosensitive epilepsy.")
    print("     If you experience dizziness, altered vision, muscle twitching, disorientation, or any other unusual symptoms, immediately stop the programme and seek medical attention.\n")
    print("    ***************************************************************************************************************************************************************************\n\n\n\n")
    input("Press any key to continue")

  data_out = []
  data_langaus_out = []
  if config.get('pmax', False) == True:
    print(f"[BETA ANALYSIS]: [PLOTTER] Plotting PMAX distribution (note that for PMAX no selections are applied to the phase space)")
    for file_ind, file_real in enumerate(file_array):
      plot_pmax = plotVar("pmax", pmax_params[0], pmax_params[1], pmax_params[2], True, output_name_array[file_ind]+"_pmax.png", fit=None)
      plot_pmax.run(file_real, file_ind, tree_array[file_ind], config['channels'])
  if config.get('amplitude', False) == True:
    amplitude_dfs = []
    ampl_langaus_dfs = []
    for file_ind, file_real in enumerate(file_array):
      df_data, df_langaus_data = plot_langaus('amplitude', file_real, file_ind, tree_array[file_ind], config['channels'], pmax_params[0], pmax_params[1], pmax_params[2], output_name_array[file_ind]+"_amplitude", int(thicknesses[0]))
      amplitude_dfs.append(df_data)
      ampl_langaus_dfs.append(df_langaus_data)
    amplitude_data = pd.concat(amplitude_dfs, ignore_index=True)
    ampl_langaus_dfs = pd.concat(ampl_langaus_dfs, ignore_index=True)
    print(amplitude_data.sort_values(by=['Channel','Bias']))
    data_out.append(('amplitude', amplitude_data.sort_values(by=['Channel','Bias'])))
    data_langaus_out.append(ampl_langaus_dfs)
  if config.get('charge', False) == True:
    charge_dfs = []
    charge_langaus_dfs = []
    for file_ind, file_real in enumerate(file_array):
      df_data, df_langaus_data = plot_langaus('charge', file_real, file_ind, tree_array[file_ind], config['channels'], charge_params[0], charge_params[1], charge_params[2], output_name_array[file_ind]+"_charge", int(thicknesses[0]))
      charge_dfs.append(df_data)
      charge_langaus_dfs.append(df_langaus_data)
    charge_data = pd.concat(charge_dfs, ignore_index=True)
    charge_langaus_dfs = pd.concat(charge_langaus_dfs, ignore_index=True)
    print(charge_data.sort_values(by=['Channel','Bias']))
    data_out.append(('charge', charge_data.sort_values(by=['Channel','Bias'])))
    data_langaus_out.append(charge_langaus_dfs)
  if config.get('gain', False) == True:
    gain_dfs = []
    gain_langaus_dfs = []
    for file_ind, file_real in enumerate(file_array):
      df_data, df_langaus_data = plot_langaus('gain', file_real, file_ind, tree_array[file_ind], config['channels'], charge_params[0], charge_params[1], charge_params[2], output_name_array[file_ind]+"_gain", int(thicknesses[0]))
      gain_dfs.append(df_data)
      gain_langaus_dfs.append(df_langaus_data)
    gain_data = pd.concat(gain_dfs, ignore_index=True)
    gain_langaus_dfs = pd.concat(gain_langaus_dfs, ignore_index=True)
    print(gain_data.sort_values(by=['Channel','Bias']))
    data_out.append(('gain', gain_data.sort_values(by=['Channel','Bias'])))
    data_langaus_out.append(gain_langaus_dfs)

  convert_and_save_csv(data_langaus_out, 'data_langaus_'+output_name+'.csv')
  if len(data_out) > 1:
    direct_to_table(data_out, config['channels'], output_name, thicknesses)
    

if __name__ == "__main__":
    main()
