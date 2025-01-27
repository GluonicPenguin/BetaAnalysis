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

from proc_tools_TR import get_fit_results_TR, hist_tree_file_timeres, plot_fit_curves, getBias

class plotTRVar:
  def __init__(self, var, nBins, xLower, xUpper, log_scale, save_name):

    self.var = var
    self.nBins = nBins
    self.xLower = xLower
    self.xUpper = xUpper
    self.log_scale = log_scale
    self.save_name = save_name

  def run(self, file, tree, channel_array):

    arr_of_hists = []
    arr_of_biases = []

    root.gErrorIgnoreLevel = root.kWarning

    result = []
    for i, (_, _, (A, B, C, D, E)) in enumerate(channel_array):
      condition = f"pmax[{i}] > {A} && pmax[{i}] < {B} && negpmax[{i}] > {C} && tmax[{i}] > {D} && tmax[{i}] < {E}"
      result.append(condition)

    duts_to_analyse = []
    mcp_exists = False
    for j in range(len(channel_array)):
      if (channel_array[j][0] == 1):
        duts_to_analyse.append([["cfd["+str(j)+"][2]-cfd[","cfd["+str(j)+"][0]-cfd[","cfd["+str(j)+"][4]-cfd["], result[j], j])
      elif (channel_array[j][0] == 2):
        mcp_channel = [[str(j)+"][2]",str(j)+"][0]",str(j)+"][4]"], result[j]]
        mcp_exists = True

    duts_vars_cuts = []
    if mcp_exists == True:
      for dut in duts_to_analyse:
        vars_cuts_defined = ([x + y for x, y in zip(dut[0], mcp_channel[0])], dut[1] + " && " + mcp_channel[1], dut[2])
        duts_vars_cuts.append(vars_cuts_defined)
    if mcp_exists == False:
      print("NEED THE THREE CHANNEL DUT SETUP")
      sys.exit(0)

    for j in range(len(duts_vars_cuts)):
      bias = getBias(file)
      hist_down_up_dev = []
      for dut_var in duts_vars_cuts[j][0]:
        thisHist = hist_tree_file_timeres(tree, file, dut_var, duts_vars_cuts[j][2], self.nBins, self.xLower, self.xUpper, bias, duts_vars_cuts[j][1])
        hist_down_up_dev.append(thisHist)
      arr_of_hists.append(hist_down_up_dev)
      arr_of_biases.append(bias)

    c1 = root.TCanvas("c1", f"Distribution time resolution", 800, 600)
    if self.log_scale:
      c1.SetLogy()


    max_y = max(hist.GetMaximum() for hist in arr_of_hists[:][0]) * 1.05
    arr_of_hists[0][0].GetYaxis().SetRangeUser(1 if self.log_scale else 0, max_y)
    arr_of_hists[0][0].SetTitle(f"Distribution time resolution")
    arr_of_hists[0][0].Draw()
    if len(arr_of_hists) > 1:
      for hist_to_draw in arr_of_hists[1:][0]:
        hist_to_draw.Draw("SAME")

    arr_of_fits = []
    for i, thisHist in enumerate(arr_of_hists):
      fit_down_up_dev = []
      for j, toa_thresh_hist in enumerate(thisHist):
        thisFit = plot_fit_curves(self.xLower, self.xUpper, "gaus", toa_thresh_hist, i, arr_of_biases[i])
        if j == 0:
          thisFit.Draw("SAME")
        fit_down_up_dev.append(thisFit)
      arr_of_fits.append(fit_down_up_dev)

    legend = root.TLegend(0.7, 0.7, 0.9, 0.9)
    for i in range(len(arr_of_hists)):
      legend.AddEntry(arr_of_hists[i][0], arr_of_biases[i] + " CH " + str(i+1), "l")

    legend.Draw()
    if not os.path.exists("timeres"):
      os.makedirs("timeres")
    c1.SaveAs("timeres/"+self.save_name)
    print(f"[BETA ANALYSIS]: [TIME RESOLUTION] Saved time resolution as "+self.save_name)

    fit_results = get_fit_results_TR(arr_of_fits,arr_of_biases,mcp_exists)
    return fit_results
