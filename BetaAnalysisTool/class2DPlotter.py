# class2DPlotter.py

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

from proc_tools_2D import hist_tree_file_2D, getBias

class plotHeatMap:
  def __init__(self, var_x, var_y, params_x, params_y, save_name):

    self.var_x = var_x
    self.var_y = var_y
    self.params_x = params_x
    self.params_y = params_y
    self.save_name = save_name

  def run(self, file, file_index, tree, channel_array):

    root.gErrorIgnoreLevel = root.kWarning

    result = []
    for i, (_, _, (_, _, _, _, E, F, _)) in enumerate(channel_array):
      if (E == 0.0) or (E == []):
        E_string = "0.0"
      else:
        E_string = f"{E[file_index]}"
      if (F == 0.0) or (F == []):
        F_string = "0.0"
      else:
        F_string = f"{F[file_index]}"

      condition = f"tmax[{i}] > "+E_string+f" && tmax[{i}] < "+F_string
      result.append(condition)

    channel_of_dut = []
    for j in range(len(channel_array)):
      bias = getBias(str(file), j)
      if (channel_array[j][0] == 1):
        #if (channel_array[j][0] == 1): channel_of_dut.append(j)
        this2DHist = hist_tree_file_2D(tree, file, self.var_x, self.var_y, j, self.params_x, self.params_y, self.save_name, result[j], bias)
        #thisHist = hist_tree_file_basics(tree, file, self.var, j, self.nBins, self.xLower, self.xUpper, bias, result[j], j)
      else:
        thisHist = None
        num_ev = 0
      #arr_of_hists.append(thisHist)

