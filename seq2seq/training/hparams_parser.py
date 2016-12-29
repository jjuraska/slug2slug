"""Utility to pass parameter strings of the form "p1=aa,p2=3,p4=True"
"""


class HParamsParser(object):
  """Pases a comma-separated string of hyperaprameters
  """

  def __init__(self, default_params):
    self.default_params = default_params

  def parse(self, params_str, with_defaults=True):
    """Passes a parameter strng and returns a dictionary of parsed values
    merged with default parameters.

    Args:
      params_str: A string of parameyters, e.g. "p1=aa,p2=3,p4=True"
      with_defaults: If true, merged the parsed values with default values.

    Returns:
      A dictionary of parameter values. These values are merged with the
      default values.
    """
    final_params = {}
    if with_defaults:
      final_params.update(self.default_params.copy())

    # Split paraeters
    params = params_str.split(",")
    params = [param.split("=") for param in params]
    params = dict([(k.strip(), v.strip()) for k, v in params])

    # Cast parameters to expected type
    for key, value in params.items():
      value_type = type(self.default_params[key])
      # To support casting strings like "128.00" to int we
      # need to cast to float first.
      if value_type == int:
        value = int(float(value))
      else:
        value = value_type(value)
      final_params.update({key: value})

    return final_params