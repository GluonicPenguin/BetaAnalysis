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

from classPlotter import plotVar
from cardReader import read_text_card

var_dict = {"tmax":"t_{max} / 10 ns" , "pmax":"p_max / mV" , "negpmax":"-p_max / mV", "charge":"Q / fC", "area_new":"Area / pWb" , "rms":"RMS / mV"}

def main():
  parser = argparse.ArgumentParser(description='Read .root files and plot distributions of corresponding variables.')
  parser.add_argument('config', type=str, help='Path to the configuration text card (e.g., config.txt)')
  args = parser.parse_args()

  config = read_text_card(args.config)

  file_list = config.get('files', [])
  channels = config.get('channels', [0] * 8)

  print(f"[BETA ANALYSIS] : Reading files {file_list}")


  for i, (channel_type, channel_value) in enumerate(channels):
    print(f"[BETA ANALYSIS] : Channel {i + 1}: Type {channel_type}, Value {channel_value}")
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

  file_array = []
  tree_array = []

  for pattern in file_list:
    root_files = glob.glob(pattern)
    for root_file in root_files:
      try:
        theFile = root.TFile(root_file)
        file_array.append(theFile)
        tree_array.append(theFile.Get("Analysis"))
        print(f"Successfully read {root_file}")
      except Exception as e:
        print(f"Error reading {root_file}: {e}")

  if len(file_array) == 1:
    print(f"Single input ROOT file read.")
  else:
    print(f"Total {len(file_array)} input ROOT files read.")

  if args.ch == 0:
    tree_with_channels = theFile.Get("Analysis")
    branch_with_number_channels = tree_with_channels.GetBranch("t")
    outer_vectors = root.std.vector('std::vector<double>')()
    tree_with_channels.SetBranchAddress("t", outer_vectors)
    outer_vector_count = 0
    for i in range(tree_with_channels.GetEntries()):
      tree_with_channels.GetEntry(i)
      outer_vector_count += len(outer_vectors)
    total_number_channels = int(outer_vector_count / branch_with_number_channels.GetEntries())
    print(f"Total {total_number_channels} channels in the file.")
  else:
    print(f"Analysing specifically CH {args.ch}.")
    total_number_channels = 1


  if config.get('tmax', False) == True:
    plot_tmax = plotVar("tmax", 1000, -2, 2, False, "tmax.png", cut_cond=args.cutCond, fit=None)
    plot_tmax.run(file_array,tree_array,args.ch-1,total_number_channels)
  if config.get('pmax', False) == True:
    plot_pmax = plotVar("pmax", pmax_params[0], pmax_params[1], pmax_params[2], True, "pmax.png", cut_cond=args.cutCond, fit=None)
    plot_pmax.run(file_array,tree_array,args.ch-1,total_number_channels)
  if config.get('negpmax', False) == True:
    plot_negpmax = plotVar("negpmax", negpmax_params[0], negpmax_params[1], negpmax_params[2], True, "negpmax.png", cut_cond=args.cutCond, fit=None)
    plot_negpmax.run(file_array,tree_array,args.ch-1,total_number_channels)
  if args.rms:
    plot_rms = plotVar("rms", rms_params[0], rms_params[1], rms_params[2], True, "negpmax.png", cut_cond=args.cutCond, fit="gaus")
    plot_rms.run(file_array,tree_array,args.ch-1,total_number_channels)


  if args.doChargeDist: plot_charge_dist.run(file_array,tree_array,args.ch-1,total_number_channels)
  if args.doRMSNoise: plot_rms.run(file_array,tree_array,args.ch-1,total_number_channels)
  if args.doTimeRes: plot_time_res.run(file_array,tree_array,args.ch-1,total_number_channels)
  if args.doDiscretisation: risingEdgeDiscretisation.run(file_array,tree_array,args.ch-1,total_number_channels)
  if args.doWaveform: plot_waveform.run(file_array,tree_array,args.ch-1,total_number_channels)
  if args.csvOut: makeCSV.run(file_array,tree_array,total_number_channels)

if __name__ == "__main__":
    main()
