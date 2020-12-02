"""
@Filename: main.py
@Author: Yize Wang
@Date: 2020/11/30
@Description: 
"""

import os
import sys
import json

from settings import *
from error_codes import *
from pykernel import init_cuda
from parse_arg import parse_arg
from parse_settings import parse_settings
from opt_wt import OptWT, run_wind_turbine

if __name__ == "__main__":
    # initialize color settings
    cl.init(autoreset=True)

    # parse case name and project name from the input argument
    case_name, proj_name = parse_arg(sys.argv)
    print(LOG + "Case: {}\nPath: {}\n".format(case_name, proj_name) + RESET + "\tOutputs can be found under this path")

    # parse case settings stored in the .ini file
    settings = parse_settings(os.path.join(proj_name, case_name), default_settings)
    print(LOG + "Settings: \n" + RESET + json.dumps(settings, indent=4))
    # append project folder in settings
    settings["proj_name"] = proj_name

    # initialize cuda environment
    kernels = init_cuda(settings)

    # optimize the hybrid energy system accordingly
    if bool(settings["wind_turbine"]["pre_layouts"]):
        if bool(int(settings["wave_energy_converter"]["converter"])):
            print(LOG + "\nOptimize wave energy converters solely" + RESET)
        else:
            print(TIP + "No jobs to do: pre_layouts and converter can not be defined simultaneously" + RESET)
            exit(no_jobs_todo)
    else:
        if bool(int(settings["wave_energy_converter"]["converter"])):
            print(LOG + "\nOptimize wind turbines and converters" + RESET)
        else:
            # if no pre_layouts of wind turbines and no converters are defined, optimize wind turbines solely
            print(LOG + "\nOptimize wind turbines solely" + RESET)

            # instantiate a problem instance
            opt_wt = OptWT(settings, kernels)
            # run the genetic algorithm
            run_wind_turbine(opt_wt)
