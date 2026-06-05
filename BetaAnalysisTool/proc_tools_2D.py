# proc_tools.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit
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

'''
from proc_tools import get_fit_results, hist_tree_file_basics, plot_fit_curves, getBias
'''

root.gErrorIgnoreLevel = root.kWarning
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def round_to_sig_figs(x, sig):
  if x == 0:
    return 0
  else:
    return round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)

def getBias(filename, chnum):
  pattern_ch = f"Ch{chnum}"
  ch_match = re.search(pattern_ch, filename)
  if ch_match:
    start_index = ch_match.end()
    substring_to_search = filename[start_index:]
        
    pattern_bias = r"-(\d{2,4}V)"
    bias_match = re.search(pattern_bias, substring_to_search)
    if bias_match:
      return bias_match.group(1)
    else:
      print("[GetBias] : BIAS NOT FOUND")
      return None
  else:
    return None

def hist_tree_file_2D(tree,file,var_x,var_y,ch,params_x,params_y,save_name,tmax_cut_cond,bias):

  root.gStyle.SetOptStat(0)
  var_dict = {"tmax":"t_{max} / ns" , "pmax":"p_{max} / mV" , "area_new":"Area / pWb"}

  c1 = root.TCanvas(f"c1_{ch}", "c1", 800, 700)
  hname1 = "CH "+str(ch)+" "+str(bias)
  h1 = root.TH2F(hname1,var_x+"["+str(ch)+"] vs "+var_y+"["+str(ch)+"];"+var_dict[var_x]+";"+var_dict[var_y], 
                 int(10*(params_x[2]-params_x[1])), params_x[1], params_x[2], int(10*(params_y[2]-params_y[1])), params_y[1], params_y[2])
  draw_expr_1 = var_x+"["+str(ch)+"]:"+var_y+"["+str(ch)+"]>>"+hname1
  tree.Draw(draw_expr_1, tmax_cut_cond, "COLZ")

  os.makedirs("area_v_pmax", exist_ok=True)
  c1.SaveAs("area_v_pmax/"+save_name+"_Ch"+str(ch)+".png")
  outfile = root.TFile(f"area_v_pmax/{save_name}_Ch{ch}.root", "RECREATE")
  h1.Write()
  outfile.Close()
  print("[BETA ANALYSIS]: [LANGAUS PLOTTER] Saved file area_v_pmax/"+save_name+"_Ch"+str(ch)+".png")

