"""
@Filename: main.py
@Author: Yize Wang
@Date: 2020/11/30
@Description: 
"""

import os
import sys

from settings import *
from error_codes import *
from parse_arg import parse_arg
from parse_settings import parse_settings

if __name__ == "__main__":
    # initialize color settings
    cl.init(autoreset=True)

    # parse case name and project name from the input argument
    case_name, proj_name = parse_arg(sys.argv)
    print(LOG + "Case: {}\nPath: {}\n".format(case_name, proj_name) + RESET + "\tOutputs can be found under this path")

    # parse case settings stored in the .ini file
    settings = parse_settings(os.path.join(proj_name, case_name))
    print(LOG + "Settings: \n" + RESET + "".join(["\t{}: {}\n".format(key, settings[key]) for key in settings.keys()]))
