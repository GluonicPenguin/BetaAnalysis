# classPlotter.py

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

from proc_tools import get_fit_results, hist_tree_file_basics, plot_fit_curves, getBias

class plotVar:
  def __init__(self, var, nBins, xLower, xUpper, log_scale, save_name, fit=None):

    self.var = var
    self.nBins = nBins
    self.xLower = xLower
    self.xUpper = xUpper
    self.log_scale = log_scale
    self.save_name = save_name
    self.fit = fit

  def run(self, files, trees, channel_array):

    arr_of_hists = []
    arr_of_biases = []

    root.gErrorIgnoreLevel = root.kWarning
    total_number_channels = len(channel_array)

    result = []

    print(channel_array)

    for i, (_, _, (A, B, C, D, E)) in enumerate(channel_array):
      condition = f"event>-1 && pmax[{i}] > {A} && pmax[{i}] < {B} && negpmax[{i}] > {C} && tmax[{i}] > {D} && tmax[{i}] < {E}"
      result.append(condition)

    for i in range(len(trees)):
      for j in range(len(channel_array)):
        bias = getBias(files[i])
        if (channel_array[j][0] == 1) or (channel_array[j][0] == 2):
          thisHist = hist_tree_file_basics(trees[i], files[i], self.var, j, self.nBins, self.xLower, self.xUpper, bias, result[j], j, 0)
        else:
          thisHist = None
        arr_of_hists.append(thisHist)
        arr_of_biases.append(bias)

    if (self.var == "pmax") or (self.var == "negpmax") or (self.var == "tmax"):
      print(f"[BETA ANALYSIS]: Plotting {self.var} (note that for {self.var} no selections are applied to the phase space)")
    else:
      print(f"[BETA ANALYSIS]: Plotting {self.var} with specified selections on the phase space")

    c1 = root.TCanvas("c1", f"Distribution {self.var}", 800, 600)
    if self.log_scale:
      c1.SetLogy()

    valid_hists = [hist for hist in arr_of_hists if hist is not None]

    max_y = max(hist.GetMaximum() for hist in valid_hists) * 1.05
    valid_hists[0].GetYaxis().SetRangeUser(1 if self.log_scale else 0, max_y)
    valid_hists[0].SetTitle(f"Distribution {self.var}")
    valid_hists[0].Draw()
    if len(valid_hists) > 1:
      for hist_to_draw in valid_hists[1:]:
        hist_to_draw.Draw("SAME")

    print(self.fit)
    if self.fit:
      arr_of_fits = []
      for channel_i in range(len(self.fit)):
        if (channel_array[j][0] == 1):
          print(f"[BETA ANALYSIS]: Performing {self.fit} fit to channel {channel_i}")
          thisFit = plot_fit_curves(self.xLower, self.xUpper, self.fit[channel_i], arr_of_hists[channel_i], channel_i, arr_of_biases[channel_i])
          arr_of_fits.append(thisFit)
          thisFit.Draw("SAME")
        else:
          arr_of_fits.append(0)

    legend = root.TLegend(0.7, 0.7, 0.9, 0.9)
    for i in range(len(valid_hists)):
      legend.AddEntry(valid_hists[i], arr_of_biases[i] + " CH " + str(i+1), "l")

    legend.Draw()
    c1.SaveAs(self.save_name)
    print(f"[BETA ANALYSIS]: Saved {self.var} as "+self.save_name)

    if self.fit:
      fit_results = get_fit_results(arr_of_fits,arr_of_biases)
      print(fit_results)
