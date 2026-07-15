import numpy as np
import ROOT
import plotly.graph_objects as go
import pandas as pd
import math
import os

from proc_tools import getBias

ROOT.EnableImplicitMT()

##########################################################################
# Compile CERN Landau-Gaussian convolution once
##########################################################################

ROOT.gInterpreter.Declare(r"""

Double_t langaufun(Double_t *x, Double_t *par)
{
   Double_t invsq2pi = 0.3989422804014;
   Double_t mpshift  = -0.22278298;

   Double_t np = 100.0;
   Double_t sc = 5.0;

   Double_t sum = 0.0;

   Double_t xx;
   Double_t xlow,xupp;
   Double_t step;
   Double_t i;

   Double_t mpc;
   Double_t fland;

   mpc = par[1] - mpshift*par[0];

   xlow = x[0]-sc*par[3];
   xupp = x[0]+sc*par[3];

   step = (xupp-xlow)/np;

   for(i=1.0;i<=np/2;i++){

      xx=xlow+(i-.5)*step;
      fland=TMath::Landau(xx,mpc,par[0])/par[0];
      sum += fland*TMath::Gaus(x[0],xx,par[3]);

      xx=xupp-(i-.5)*step;
      fland=TMath::Landau(xx,mpc,par[0])/par[0];
      sum += fland*TMath::Gaus(x[0],xx,par[3]);

   }

   return par[2]*step*sum*invsq2pi/par[3];
}

""")

def round_to_sig_figs(x, sig):
    if x == 0:
        return 0
    return round(x, sig - int(math.floor(math.log10(abs(x)))) - 1)


def fmt_val_unc(val, unc, sig_figs_val=3):
    if val == 0:
        decimals = 0
    else:
        order = int(np.floor(np.log10(abs(val))))
        decimals = max(sig_figs_val - 1 - order, 0)
    return f"{round(val, decimals):.{decimals}f}" + " ± " + f"{round(unc, decimals):.{decimals}f}"


def build_histogram(data, nbins, xmin, xmax, ch_ind):
    h = ROOT.TH1D("hist_Ch{ch_ind}", "", nbins, xmin, xmax)
    for value in data:
        h.Fill(float(value))
    return h


def fit_langau_root(hist, xmin, xmax, ch_ind):
    mpv_guess = hist.GetBinCenter(hist.GetMaximumBin())
    width_guess = hist.GetRMS() / 5.
    sigma_guess = width_guess
    area_guess = hist.Integral()
    fit = ROOT.TF1("langau_Ch{ch_ind}", "langaufun", xmin, xmax, 4)
    fit.SetParNames("Width", "MP", "Area", "GSigma")
    fit.SetParameters(width_guess, mpv_guess, area_guess, sigma_guess)
    hist.Fit(fit, "RQ0")
    return fit


def extract_fit_parameters(fit):
    return {
        "mpv": fit.GetMaximumX(),
        "landau_width": fit.GetParameter(0),
        "gauss_sigma": fit.GetParameter(3),
        "area": fit.GetParameter(2),
        "fit": fit
    }


def evaluate_fit(fit, xmin, xmax, npoints=1000):
    x = np.linspace(xmin, xmax, npoints)
    y = np.array([fit.Eval(xx) for xx in x])
    return x, y


def evaluate_landau(fit, xmin, xmax, npoints=1000):
    landau = ROOT.TF1("landau", "[2]*TMath::Landau(x,[1],[0])", xmin, xmax)
    landau.SetParameters(fit.GetParameter(0), fit.GetParameter(1), fit.GetParameter(2))

    x = np.linspace(xmin, xmax, npoints)
    y = np.array([landau.Eval(xx) for xx in x])

    return x, y


def bootstrap_langau(hist, fit, n_boot=500):
    rng = ROOT.TRandom3(0)

    bestpars = [fit.GetParameter(i) for i in range(4)]

    mpvs = []
    widths = []
    sigmas = []

    for i in range(n_boot):
        hboot = hist.Clone(f"hboot_{i}")

        for b in range(1, hist.GetNbinsX() + 1):
            c = hist.GetBinContent(b)
            hboot.SetBinContent(b, rng.Poisson(c))

        fboot = fit.Clone(f"fit_{i}")
        fboot.SetParameters(*bestpars)

        hboot.Fit(fboot, "RQ0")

        mpvs.append(fboot.GetMaximumX())
        widths.append(fboot.GetParameter(0))
        sigmas.append(fboot.GetParameter(3))

    return {
        "mpvs": np.asarray(mpvs),
        "widths": np.asarray(widths),
        "sigmas": np.asarray(sigmas)
    }

def bootstrap_summary(samples):
    return {
        "mean":np.mean(samples),
        "std":np.std(samples),
        "median":np.median(samples),
    }

def compute_ltf_from_data(data, mpv):
    n1 = np.sum(data > mpv)
    n2 = np.sum(data > 1.5*mpv)
    if n1 == 0: return 0.
    return n2/n1

def plot_langaus(var, file, file_index, tree, channel_array, nBins, xLower, xUpper, savename, thickness, n_bootstrap=500):

  arr_of_ch = []
  arr_of_biases = []
  arr_of_MPV = []
  arr_of_MPV_unc = []
  arr_of_width = []
  arr_of_sigma = []
  arr_of_ltf = []
  arr_of_ltf_unc = []
  arr_ratio = []
  arr_ratio_unc = []
  arr_of_sse = []
  arr_of_rchi2 = []

  x_axis_add = []
  y_fit_counts_add = []

  dict_of_vars = {"amplitude": "Amplitude / mV", "area_fitted": "Area / pWb", "charge": "Charge / fC", "gain": "Gain / fC", "dvdt": "dV/dt / mV/ps", "dvdt_2080": "dV/dt[20%:80%] / mV/ps"}
  for ch_ind, ch_val in enumerate(channel_array):
    area_list = []
    pmax_list = []
    negpmax_list = []

    sensorType, AtQfactor, (A, B, C, D, E, F, G) = ch_val
    if AtQfactor == 0:
      AtQfactor = 1
    if sensorType == 1:
      if (A == 0.0) or (A == []):
        A = 0.0
      else:
        A = A[file_index]
      if B == 0: B = 1600
      if (C == 0.0) or (C == []):
        C = 0.0
      else:
        C = C[file_index]
      if D == 0: D = 1600
      if (E == 0.0) or (E == []):
        E = -50
      else:
        E = E[file_index]
      if (F == 0.0) or (F == []):
        F = 50
      else:
        F = F[file_index]
      if G == 0:
        G = -100
      bias_of_channel = getBias(str(file), ch_ind)
      for entry in tree:
        area_sig = entry.area_new[ch_ind]
        pmax_sig = entry.pmax[ch_ind]
        tmax_sig = entry.tmax[ch_ind]
        negpmax_sig = entry.negpmax[ch_ind]
        if (area_sig < A) or (area_sig > B) or (pmax_sig < C) or (pmax_sig > D) or (tmax_sig < E) or (tmax_sig > F) or (negpmax_sig < G):
          continue
        else:
          pmax_list.append(pmax_sig)
          area_list.append(area_sig)
          negpmax_list.append(negpmax_sig)
    else:
      continue

    if var == "area_fitted":
      area = np.array(area_list)
      data_var = area[(area>=xLower) & (area<=xUpper)]
    else:
      pmax = np.array(pmax_list)
      data_var = pmax[(pmax>=xLower) & (pmax<=xUpper)]

    hist = build_histogram(data_var,nBins,xLower,xUpper, ch_ind)
    counts=np.array([hist.GetBinContent(i+1) for i in range(nBins)])
    centres=np.array([hist.GetBinCenter(i+1) for i in range(nBins)])

    fit = fit_langau_root(hist,xLower,xUpper,ch_ind)
    pars = extract_fit_parameters(fit)

    toys = bootstrap_langau(hist, fit, n_bootstrap)
    mpv_stats = bootstrap_summary(toys["mpvs"])
    width_stats = bootstrap_summary(toys["widths"])
    sigma_stats = bootstrap_summary(toys["sigmas"])

    arr_of_ch.append(f"Ch{ch_ind}")
    bias=getBias(str(file), ch_ind)
    arr_of_biases.append(bias)

    arr_of_MPV.append(pars["mpv"])
    arr_of_MPV_unc.append(mpv_stats["std"])
    arr_of_width.append(pars["landau_width"])
    arr_of_sigma.append(pars["gauss_sigma"])
    arr_ratio.append(pars["landau_width"] / pars["mpv"])
    arr_ratio_unc.append(arr_ratio*np.sqrt((mpv_stats["std"]/pars["mpv"])**2 + (width_stats["std"]/pars["landau_width"])**2))

    ltf = compute_ltf_from_data(data_var, pars["mpv"])
    arr_of_ltf.append(ltf)
    toy_ltf=[]

    for mpv in toys["mpvs"]:
        toy_ltf.append(compute_ltf_from_data(data_var, mpv))

    toy_ltf=np.asarray(toy_ltf)
    arr_of_ltf_unc.append(np.std(toy_ltf))

    y_fit=np.array([fit.Eval(x) for x in centres])

    residuals=counts-y_fit

    sigma = np.sqrt(np.maximum(counts, 1))

    chi2 = np.sum((residuals / sigma) ** 2)
    nu = nBins - 4
    arr_rchi2.append(chi2 / nu)
    arr_sse.append(np.sum(residuals ** 2))
    x_axis, y_fit_curve = evaluate_fit(fit, xLower, xUpper)
    _, y_landau = evaluate_landau(fit, xLower, xUpper)
    x_axis_add.append(x_axis)
    y_fit_counts_add.append(y_fit_curve)

    legend_fontsize=42
    bias_of_channel_spaced = bias_of_channel[:-1] + " " + bias_of_channel[-1]

    fig = go.Figure()
    fig.update_layout(
      width=1600,
      height=1000,
      font=dict(
        family="Arial",
        size=legend_fontsize,
        color="black"
      ),
      legend=dict(font=dict(size=legend_fontsize+4),
        x=1,
        y=1,
        xanchor='right',
        yanchor='top',
        bgcolor='white',
        bordercolor='lightgray',
        borderwidth=2,
        itemwidth=80
        ),
      xaxis_title=dict_of_vars[var],
      yaxis_title='Number of events',
    )
    if (var == "charge") & (int(bias_of_channel.rstrip("V")) >= 300):
      fig.add_trace(
        go.Scatter(
          x=centres,
          y=counts,
          mode='markers',
          name=f"{thickness} μm, V<sub>bias</sub> = {bias_of_channel_spaced}",
          marker=dict(color='black', size=15),
          error_y=dict(type='data', array=np.sqrt(counts))
        )
      )
    else:
      fig.add_trace(
        go.Scatter(
          x=bin_centres,
          y=counts,
          mode='markers',
          name=f"{thickness} μm, V<sub>bias</sub> = {bias_of_channel_spaced}",
          marker=dict(color='black', size=15),
          error_y=dict(type='data', array=np.sqrt(histo))
        )
      )
    x_axis = np.linspace(xLower, xUpper, 999)
    fig.add_trace(
    go.Scatter(
        x=x_axis,
        y=y_fit_curve,
        name=(
             f"<i>Q</i>~<b>θ</b>(<i>Q</i><sub>MPV</sub> = {fmt_val_unc(fmt_val_unc(pars["mpv"], mpv_stats["std"]))},<br>"
             f"          ξ = {fmt_val_unc(fmt_val_unc(pars["landau_width"], width_stats["std"]))},<br>"
             f"          σ = {fmt_val_unc(fmt_val_unc(pars["gauss_sigma"], sigma_stats["std"]))})"
            ),
        mode='lines',
        line=dict(color='red', width=10)
        )
    )
    fig.add_trace(
      go.Scatter(
        x=x_axis,
        y=y_landau,
        mode='lines',
        name=f'  Landau contribution',
        line=dict(color='blue', width=10, dash='dash')
      )
    )

    fig.update_layout(
      plot_bgcolor="white",
      paper_bgcolor="white",
      xaxis=dict(
        showgrid=True,
        gridcolor="lightgray",
        zeroline=False
      ),
      yaxis=dict(
        showgrid=True,
        gridcolor="lightgray",
        zeroline=False,
        range=[0, 1.2*max(counts)]
      )
    )

    if not os.path.exists(var):
      os.makedirs(var)
    fig.write_image(var+"/GS_STUDIES_"+savename+"_Ch"+str(ch_ind)+".png")
    print("[BETA ANALYSIS]: [LANGAUS PLOTTER] Saved file "+var+"/"+savename+"_Ch"+str(ch_ind)+".png")
    
  if (var != "dvdt") & (var != "dvdt_2080"): var = var.capitalize()
  if var == "Area_fitted": var = "Area" 

  df_of_langaus = pd.DataFrame({"BIAS": arr_of_biases,var: x_axis_add,var+"_EVENTS": y_fit_counts_add})
  var = var.capitalize()
  df_of_results = pd.DataFrame({
    "Channel": arr_of_ch,
    "Bias": arr_of_biases,
    var: arr_of_MPV,
    var+" Unc": arr_of_MPV_unc,
    "Landau width": arr_of_width,
    "Gaussian sigma": arr_of_sigma,
    "LTF": arr_of_ltf,
    "LTF Unc": arr_of_ltf_unc,
    #"LTF from area": arr_mpv_frac_from_area,
    #"LTFmax": arr_maxbin_frac,
    #"LTFmax Unc": arr_maxbin_frac_unc,
    "Landau Frac": arr_ratio,
    "Landau Frac Unc": arr_ratio_unc,
    "SSE score": arr_of_sse,
    "Red. Chi2": arr_of_rchi2,
  })
  return df_of_results, df_of_langaus
