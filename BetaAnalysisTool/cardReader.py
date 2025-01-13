# cardReader.py

import re

def read_text_card(file_path):
  config = {}

  channel_mapping = {
      "DUT": 1,
      "MCP": 2,
      "REF": 3
  }
  channel_area_to_charge_mapping = {
        "SC": 4.7,
        "Mig": 5
  }
  channels = [[0, 1]] * 8

  plot_flags = {
    "tmax": False,
    "pmax": False,
    "negpmax": False,
    "charge": False,
    "rms": False,
    "timeres": False,
    "discretisation": False,
    "waveform": False,
    "all_with_no_plots": False,
  }

  plot_params = {
    "pmax_params": None,
    "negpmax_params": None,
    "charge_params": None,
    "rms_params": None,
    "timeres_params": None
  }

  with open(file_path, 'r') as f:
    for line in f:
      line = line.strip()
      if not line or line.startswith('#'):
        continue

      # Match lines in the format: key = value
      match = re.match(r'^(\w+)\s*=\s*(.+)$', line)
      if match:
        key, value = match.groups()
        value = value.strip().strip('"').strip("'")

        if key == "files":
          value = [item.strip() for item in value.split(',')]

        elif key.startswith("CH_") and key[3:].isdigit():
          index = int(key[3:]) - 1  # Convert to 0-based index

           parts = [part.strip() for part in value.split(',')]
           type_str = parts[0]
           additional_str = parts[1] if len(parts) > 1 else ""

           channel_type = channel_type_mapping.get(type_str.upper(), 0)
           channel_value = channel_value_mapping.get(additional_str, 1)

           channels[index] = [channel_type, channel_value, None]
        elif key.startswith("CH") and key.endswith("_cut"):
           channel_index = int(key[2]) - 1
           lower_bound, upper_bound, additional_condition = map(int, value.split(","))
           channels[channel_index][2] = (lower_bound, upper_bound, additional_condition)

        elif key in plot_flags:
          plot_flags[key] = value.lower() == "true"
        elif key.endswith("_nB_xL_xU"):
          param_key = key.split("_nB_xL_xU")[0]
          if plot_flags.get(param_key, False):  # Check if this plot is enabled
            nBins, xLower, xUpper = map(float, value.split(","))
            plot_params[param_key] = (nBins, xLower, xUpper)
        else:
          config[key] = value

  config['channels'] = channels
  config.update(plot_flags)
  config.update(plot_params)
  return config
