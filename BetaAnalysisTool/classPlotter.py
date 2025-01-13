# classPlotter.py

import ROOT as root

from proc_tools import get_fit_results, hist_tree_file_basics, plot_fit_curves, getBias

class plotVar:
  def __init__(self, var, nBins, xLower, xUpper, log_scale, save_name, cut_cond="event>-1", fit=None):

    self.var = var
    self.nBins = nBins
    self.xLower = xLower
    self.xUpper = xUpper
    self.log_scale = log_scale
    self.save_name = save_name
    self.cut_cond = cut_cond
    self.fit = fit

  def run(self, files, trees, ch, total_number_channels):
    print(f"[BETA ANALYSIS]: Reading {self.var}")
    arr_of_hists = []
    arr_of_biases = []

    if total_number_channels == 1:
      for i in range(len(trees)):
        bias = getBias(files[i])
        thisHist = hist_tree_file_basics(trees[i], files[i], self.var, i, self.nBins, self.xLower, self.xUpper, bias, cut_cond, ch, 0)
        arr_of_hists.append(thisHist)
        arr_of_biases.append(bias)
      else:
        for i in range(len(trees)):
          for j in range(total_number_channels):
            bias = getBias(files[i])
            thisHist = hist_tree_file_basics(trees[i], files[i], self.var, j, self.nBins, self.xLower, self.xUpper, bias, cut_cond, j, 0)
            arr_of_hists.append(thisHist)
            arr_of_biases.append(bias)

    print(f"[BETA ANALYSIS]: Plotting {self.var}")

    c1 = root.TCanvas("c1", f"Distribution {self.var}", 800, 600)
    if self.log_scale:
      c1.SetLogy()

    max_y = max(hist.GetMaximum() for hist in arr_of_hists) * 1.05
    arr_of_hists[0].GetYaxis().SetRangeUser(1 if self.log_scale else 0, max_y)
    arr_of_hists[0].SetTitle(f"Distribution {self.var}")
    arr_of_hists[0].Draw()
    if len(arr_of_hists) > 1:
      for hist_to_draw in arr_of_hists[1:]:
        hist_to_draw.Draw("SAME")

    if self.fit:
      arr_of_fits = []
      for channel_i in range(len(self.fit)):
        if self.fit[channel_i] != 0:
          print(f"[BETA ANALYSIS]: Performing {self.fit} fit to channel {channel_i}")
          thisFit = plot_fit_curves(self.xLower, self.xUpper, self.fit[channel_i], arr_of_hists[channel_i], channel_i, arr_of_biases[channel_i])
          arr_of_fits.append(thisFit)
          thisFit.Draw("SAME")
        else:
          arr_of_fits.append(0)

    legend = root.TLegend(0.7, 0.7, 0.9, 0.9)
    for i in range(len(arr_of_hists)):
      if ch == -1: ch_num = i + 1
      if ch != -1: ch_num = ch + 1
      legend.AddEntry(arr_of_hists[i], arr_of_biases[i] + " CH " + str(ch_num), "l")

    legend.Draw()
    c1.SaveAs(self.save_name)
    print(f"[BETA ANALYSIS]: Saved {self.var} as "+save_name)

    if self.fit:
      fit_results = get_fit_results(arr_of_fits,arr_of_biases)
      print(fit_results)
