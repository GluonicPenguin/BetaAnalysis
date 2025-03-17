import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, poisson, median_abs_deviation
from scipy.optimize import curve_fit, minimize
from scipy.special import erf
import pandas as pd
from landaupy import langauss

import ROOT as root
from ROOT import TF1
import argparse
import glob
import re
import os
import csv
import sys

from firstPass_ProcTools import getBias

def binned_fit_langauss(samples, bins, min_x_val, max_x_val, channel, nan='remove'):
  if nan == 'remove':
    samples = samples[~np.isnan(samples)]

  hist, bin_edges = np.histogram(samples, bins, range=(min_x_val,max_x_val), density=True)
  bin_centres = bin_edges[:-1] + np.diff(bin_edges) / 2
  hist = np.insert(hist, 0, sum(samples < bin_edges[0]))
  bin_centres = np.insert(bin_centres, 0, bin_centres[0] - np.diff(bin_edges)[0])
  hist = np.append(hist, sum(samples > bin_edges[-1]))
  bin_centres = np.append(bin_centres, bin_centres[-1] + np.diff(bin_edges)[0])

  hist = hist[1:-1]
  bin_centres = bin_centres[1:-1]
  landau_x_mpv_guess = bin_centres[np.argmax(hist)]
  landau_xi_guess = median_abs_deviation(samples) / 5
  gauss_sigma_guess = landau_xi_guess / 10

  popt, pcov = curve_fit(
    lambda x, mpv, xi, sigma: langauss.pdf(x, mpv, xi, sigma),
    xdata=bin_centres,
    ydata=hist,
    p0=[landau_x_mpv_guess, landau_xi_guess, gauss_sigma_guess],
  )
  return popt, pcov, hist, bin_centres

def gaussian(x, A, mu, sigma):
  return A * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

def SNDisc_extract_signal(file, file_index, tree, channel_array, nBins, savename):

  png_files = glob.glob("SNDisc_performance/*.png")
  for png_file in png_files:
    os.remove(png_file)
  max_pmax = nBins
  arr_signal_events = []

  for ch_ind, ch_val in enumerate(channel_array):
    pmax_list = []
    width_list = []
    sensorType, AtQfactor, ansatz_pmax = ch_val
    if sensorType == 1:
      bias_of_channel = getBias(str(file), ch_ind)
      for entry in tree:
        pmax_sig = entry.pmax[ch_ind]
        width_sig = entry.width[ch_ind][2]
        pmax_list.append(pmax_sig) # to fit Gaus noise and Langaus signal
        width_list.append(width_sig) # to fit Gaus noise and Langaus signal on pmax/width@30%
    else:
      continue

    pmax = np.array(pmax_list)
    width = np.array(width_list)

    true_pmax_sig = pmax[pmax >= ansatz_pmax[file_index]]
    true_pmax_noise = pmax[pmax < ansatz_pmax[file_index]]
    true_width_sig = width[np.where(pmax >= ansatz_pmax[file_index])[0]]
    true_width_noise = width[np.where(pmax < ansatz_pmax[file_index])[0]]

    num_events = len(pmax)
    sig_events = np.count_nonzero(pmax >= ansatz_pmax[file_index])
    labels = np.concatenate([np.ones(sig_events), np.zeros(num_events - sig_events)])  # 1 = signal, 0 = noise

    # Convert to PyTorch tensors
    max_amplitudes = torch.tensor(pmax, dtype=torch.float32)
    max_pow = torch.tensor(pmax / width, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.float32)

    # Define a differentiable signal probability model
    class SignalProbabilityModel(nn.Module):
      def __init__(self):
        super().__init__()
        self.amp_mu = nn.Parameter(torch.tensor(2.0*ansatz_pmax[file_index]))   # Langaus centre
        self.amp_sigma = nn.Parameter(torch.tensor(3.0)) # Langaus width
        self.pow_mu = nn.Parameter(torch.tensor(2.0*ansatz_pmax[file_index]))  # Langaus centre
        self.pow_sigma = nn.Parameter(torch.tensor(3.0)) # Langaus width

      def forward(self, amplitudes, pow):
        amp_prob = torch.sigmoid((amplitudes - self.amp_mu) / self.amp_sigma)
        pow_prob = torch.sigmoid((pow - self.pow_mu) / self.pow_sigma)
        return amp_prob * pow_prob  # Combined probability

    num_epochs = 10000
    precision = 0.002
    plot_every = precision*1
    dec_p = len(str(precision)) - 2
    arr_prob_threshold = np.round(np.arange(0.04,0.05,precision), dec_p)

    data_list = []
    model = SignalProbabilityModel()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(num_epochs):
      optimizer.zero_grad()
      scores = model(max_amplitudes, max_pow)
      loss = -torch.mean(labels * torch.log(scores + 1e-6) + (1 - labels) * torch.log(1 - scores + 1e-6))  
      loss.backward()
      optimizer.step()
      if epoch % 2000 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")
        #for name, param in model.named_parameters():
        #  print(name, param.grad)

    scores = model(max_amplitudes, max_pow).detach().numpy()
    df = pd.DataFrame(scores)
    df.to_csv(savename + "SNDisc_Ch"+str(ch_ind)+".csv", index=False, header=False)

    arr_of_MPV = []
    arr_of_width = []
    arr_of_sigma = []
    arr_mpv_frac = []
    arr_qmax_frac = []
    arr_of_sse = []
    arr_of_rchi2 = []
    plt.figure(figsize=(10, 6))

    scores[max_amplitudes.numpy() > max_pmax] = 0

    for prob_threshold in arr_prob_threshold:
      selected_events = scores > prob_threshold
      
      # Stopping condition for probability threshold that there are fewer selected events by the NN than from the linear cut in PMAX
      if len(selected_events) < sig_events:
        continue
      filtered_amplitudes = max_amplitudes[selected_events].numpy()
      data_var = filtered_amplitudes

      histo, bins, _ = plt.hist(data_var, bins=max_pmax, range=(0, max_pmax), color='blue', edgecolor='black', alpha=0.6, density=True)
      print(f"Number of filtered events: {len(data_var)}")

      bin_centres = bins[:-1] + np.diff(bins) / 2
      popt, pcov, fitted_hist, bin_centres = binned_fit_langauss(data_var, nBins, 0, max_pmax, ch_ind)
      arr_of_MPV.append(popt[0])
      arr_of_width.append(popt[1])
      arr_of_sigma.append(popt[2])

      mpv, xi, sigma = popt
      mu_lang, sigma_lang, k_lang = popt
      y_fit = langauss.pdf(bin_centres, mpv, xi, sigma)

      residuals = histo - y_fit
      sse = np.sum(residuals**2)
      norm_sse = sse / len(data_var)
      arr_of_sse.append(sse)
      sigma = np.sqrt(histo)
      sigma[sigma == 0] = 1
      chi2 = np.sum((residuals / sigma) ** 2)

      nu = len(histo) - len(popt)
      chi2_red = chi2 / nu
      arr_of_rchi2.append(chi2_red)
      rchi2 = chi2_red

      bin_heights, bin_edges = np.histogram(max_amplitudes.numpy(), bins=max_pmax, range=(0, max_pmax), density=True)
      bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
      popt_gaussian, _ = curve_fit(gaussian, bin_centres, bin_heights, p0=[1-(sig_events / num_events), 0.5*ansatz_pmax[file_index], 0.5*ansatz_pmax[file_index]])
      A_gauss, mu_gauss, sigma_gauss = popt_gaussian
      filtered_heights, filtered_edges = np.histogram(filtered_amplitudes, bins=bin_edges, density=True)
      filtered_centres = (filtered_edges[:-1] + filtered_edges[1:]) / 2

      if np.isclose(prob_threshold / plot_every, np.round(prob_threshold / plot_every)) == True:
        signal_event_count = len(filtered_amplitudes)
        total_event_count = len(max_amplitudes)
        scale_factor = signal_event_count / total_event_count
        langaus_scaled = langauss.pdf(bin_centres, *popt) * scale_factor

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].hist(max_amplitudes.numpy(), bins=max_pmax, range=(0, max_pmax), alpha=0.7, label="Signal + Noise", color='lightgray', density=True)
        gaus_label = f"Gauss (Noise) Fit\nμ = {A_gauss:.2f}, σ = {mu_gauss:.2f}, k = {sigma_gauss:.2f}"
        axes[0].plot(bin_centres, gaussian(bin_centres, *popt_gaussian), color='orange', linestyle='-', label=gaus_label, linewidth=2)
        langaus_label1 = f"Rescaled Langaus (Signal) Fit\nμ = {mu_lang:.2f}, k = {k_lang:.2f}, σ = {sigma_lang:.2f}"
        axes[0].plot(bin_centres, langaus_scaled, 'g-', label=langaus_label1, linewidth=2)
        axes[0].set_xlabel("Max Amplitude")
        axes[0].set_ylabel("Normalised Counts")
        axes[0].set_title("Raw Signal + Noise Distribution")
        axes[0].set_yscale('log')
        axes[0].set_ylim(1/num_events, 1)
        axes[0].legend()

        axes[1].hist(filtered_amplitudes, bins=max_pmax, range=(0, max_pmax), alpha=0.4, label="Filtered Signal", color='g', density=True)
        langaus_label = f"Langaus (Signal) Fit\nμ = {mu_lang:.2f}, σ = {sigma_lang:.2f}, k = {k_lang:.2f}\nRed. $\chi^{2}$ = {rchi2:.2e}"
        axes[1].plot(filtered_centres, langauss.pdf(filtered_centres, *popt), 'k--', label=langaus_label, linewidth=2)
        axes[1].set_xlabel("Max Amplitude")
        axes[1].set_ylabel("Normalised Counts")
        axes[1].set_title("Filtered Signal Distribution")
        axes[1].legend()
        axes[1].annotate(str(format(prob_threshold, "."+str(dec_p)+"f")), xy=(0.6,0.5), xycoords='axes fraction', fontsize=25, fontweight='bold')
        plt.tight_layout()
        plt.savefig("SNDisc_performance/pmax_Ch"+str(ch_ind)+"_"+str(num_epochs)+"_"+str(format(prob_threshold, "."+str(dec_p)+"f"))[2:]+".png",facecolor='w')

      modchi2 = rchi2*pow(len(filtered_amplitudes),-1.4)
      num_ev_above_1p0 = len(filtered_amplitudes[filtered_amplitudes > mu_lang])
      num_ev_above_1p5 = len(filtered_amplitudes[filtered_amplitudes > 1.5*mu_lang])
      frac_ev_above_1p5 = num_ev_above_1p5 / num_ev_above_1p0
      data_list.append([ch_ind, bias_of_channel, num_epochs, prob_threshold, len(filtered_amplitudes), mu_lang.round(2), k_lang.round(3), sigma_lang.round(3), round(frac_ev_above_1p5,3), sse.round(6), norm_sse.round(10), chi2.round(4), rchi2.round(6), modchi2])

    column_headings = ["Channel","Bias","Number of epochs","Probability threshold","Signal event count","Amplitude MPV","Landau width","Gaussian sigma","Frac above 1p5 MPV","SSE score","SSE / No. events","Chi2","Red. Chi2","Mod. Chi2"]
    df = pd.DataFrame(data_list, columns=column_headings)
    min_sse_prob = df.loc[df["SSE score"].idxmin(), "Probability threshold"]
    min_chi2_prob = df.loc[df["Red. Chi2"].idxmin(), "Probability threshold"]
    smallest_prob = min(min_sse_prob, min_chi2_prob)
    optimal_selected_events = scores > smallest_prob
    arr_signal_events.append(optimal_selected_events)
    
    #filtered_amplitudes = max_amplitudes[optimal_selected_events].numpy()
    #df.to_csv("disc_analysis.csv", index=False)
    amplitude_df = df.loc[df["Probability threshold"] == smallest_prob]
    amplitude_df = amplitude_df[["Channel","Bias","Amplitude MPV","Landau width","Gaussian sigma","Frac above 1p5 MPV","SSE score","Red. Chi2"]]

    colours = ["r","orange","yellow","lime","green","blue","purple","magenta"]
    markers = ["o","v","s","^","D","p","d","h"]

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))

    for i, y_column in enumerate(['Signal event count', 'Amplitude MPV', 'SSE score', 'Red. Chi2', 'Mod. Chi2']):
      ax = axes[i]
      for j, label in enumerate(df['Number of epochs'].unique()):
        subset = df[df['Number of epochs'] == label]
        ax.plot(subset['Probability threshold'], subset[y_column], label=f'{label} ({y_column})', color=colours[j], marker=markers[j], markersize=8, markeredgecolor='black', markeredgewidth=1)

      ax.set_xlabel('Probability threshold')
      ax.set_ylabel(y_column)
      if i >= 2:
        ax.set_yscale("log")
      ax.legend()

    plt.tight_layout()
    plt.savefig(savename+"_Ch"+str(ch_ind)+"_pdperformanceplots.png",facecolor='w')

    df.to_csv(savename + "diagnostics_SNDisc_Ch"+str(ch_ind)+".csv", index=False)

    # NEED TO DO AMPLITUDE FIT AND RETURN VALUES AS BEFORE

  return arr_signal_events, amplitude_df
