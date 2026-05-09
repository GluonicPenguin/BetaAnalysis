# cardReader.py

import re

def read_text_card(file_path):
  config = {}

  channel_type_mapping = {
      "DUT": 1,
      "MCP": 2,
      "REF": 3
  }
  channel_area_to_area_fitted_mapping = {
        "SC": 4.7,
        "Mig": 5,
        "None": 1
  }
  channels = [[0, 1]] * 8

  plot_flags = {
    "tmax": False,
    "area_new": False,
    "pmax": False,
    "amplitude": False,
    "risetime": False,
    "area_fitted": False,
    "rms": False,
    "timeres": False,
  }

  plot_params = {
    "tmax_params": None,
    "area_params": None,
    "pmax_params": None,
    "risetime_params": None,
    "area_fitted_params": None,
    "rms_params": None,
    "timeres_params": None
  }

  MCP_specs = None

  with open(file_path, 'r') as f:
    current_key = None  # Track the current key being processed
    current_value = []  # Collect multi-line values
    thickness_info = []
    for line in f:
      line = line.strip()
      if not line or line.startswith('#'):  # Skip empty lines and comments
        continue

      match = re.match(r'^(\w+)\s*=\s*(.+)$', line)
      if match:
        if current_key and current_value:
          if current_key == "files":
            config[current_key] = "".join(current_value).strip('",').split(',')
          else:
            config[current_key] = "".join(current_value).strip()
          current_key = None
          current_value = []

        key, value = match.groups()
        value = value.strip().strip('"').strip("'")

        if key == "files":  # Handle multi-line `files`
          current_key = key
          current_value.append(value)

        elif key == "run_safe_mode":
          safemode = value.lower() == "true"

        elif key.startswith("CH_") and key[3:].isdigit():  # Handle CH_ keys
          index = int(key[3:]) - 1  # Convert to 0-based index
          parts = [part.strip() for part in value.split(',')]
          type_str = parts[0]
          additional_str = parts[1] if len(parts) > 1 else ""
          thickness_str = parts[2] if (len(parts) > 2) & (type_str.upper() != "MCP") else "nDUT"

          channel_type = channel_type_mapping.get(type_str.upper(), 0)
          channel_value = channel_area_to_area_fitted_mapping.get(additional_str, 1)
          thickness_info.append(thickness_str)

          if (type_str.upper() == "MCP"):
            if (parts[2] == 0) & (parts[3] == 0):
              MCP_specs = (0, 0)
            else:
              MCP_specs = (float(parts[2]), float(parts[3]))

          channels[index] = [channel_type, channel_value, None]

        elif key.startswith("CH") and key.endswith("_cut"):
          channel_index = int(key[2]) - 1
          match = re.match(r"^\s*(\[\s*(?:-?\d+(?:\.\d+)?\s*,\s*)*-?\d+(?:\.\d+)?\s*\]|\[\s*\]|0)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(\[\s*(?:-?\d+(?:\.\d+)?\s*,\s*)*-?\d+(?:\.\d+)?\s*\]|\[\s*\]|0)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(\[\s*(?:-?\d+(?:\.\d+)?\s*,\s*)*-?\d+(?:\.\d+)?\s*\]|\[\s*\]|0)\s*,\s*(\[\s*(?:-?\d+(?:\.\d+)?\s*,\s*)*-?\d+(?:\.\d+)?\s*\]|\[\s*\]|0)\s*$", value)

          if not match:
            raise ValueError(f"Invalid format for {key}: Must be '[...],0,[...],0,[...],[...]' where arrays are '[x,y,...]', '[]', or '0'.")

          raw_lower_bound = match.group(1).strip()
          upper_bound = float(match.group(2).strip())
          raw_plow = match.group(3).strip()
          phigh = float(match.group(4).strip())
          raw_tlow = match.group(5).strip()
          raw_thigh = match.group(6).strip()

          def parse_array(raw_value, field_name):
            if raw_value == "[]" or raw_value == "0":
              return []
            arr = list(map(float, raw_value.strip("[]").split(",")))
            if len(arr) != len(config.get('files', [])):
              raise ValueError(f"Invalid length for {field_name} in {key}: Must match the number of files ({len(config['files'])}).")
            return arr

          lower_bound = parse_array(raw_lower_bound, "lower_bound")
          plow = parse_array(raw_plow, "plow")
          tlow = parse_array(raw_tlow, "tlow")
          thigh = parse_array(raw_thigh, "thigh")

          channels[channel_index][2] = (lower_bound, upper_bound, plow, phigh, tlow, thigh)

        elif key in plot_flags:  # Handle plot flags
          plot_flags[key] = value.lower() == "true"
        elif key.endswith("_nB_xL_xU"):  # Handle plot parameters
          param_key = key.split("_nB_xL_xU")[0]
          if param_key == "area":
            temp_param_key = "area_new"
          else:
            temp_param_key = param_key
          if plot_flags.get(temp_param_key, False) == True:  # Check if this plot is enabled
            nBins, xLower, xUpper = map(float, value.split(","))
            plot_params[param_key+"_params"] = (int(nBins), xLower, xUpper)
          elif (plot_flags.get("amplitude", False)) and (param_key == "pmax"):
            nBins, xLower, xUpper = map(float, value.split(","))
            plot_params["pmax_params"] = (int(nBins), xLower, xUpper)
        else:  # Handle generic key-value pairs
          config[key] = value
      elif current_key:  # Handle continuation lines
        current_value.append(line.strip())

    if current_key and current_value:
        if current_key == "files":
          config[current_key] = "".join(current_value).strip('",').split(',')
        else:
          config[current_key] = "".join(current_value).strip()

  config['channels'] = channels
  config.update(plot_flags)
  config.update(plot_params)
  return config, thickness_info, MCP_specs, safemode
