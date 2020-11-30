"""
@Filename: main.py
@Author: Yize Wang
@Date: 2020/11/30
@Description: 
"""

import os
import sys
import colorama as cl

from error_codes import *
from parse_arg import parse_arg

if __name__ == "__main__":
    # initialize color settings
    cl.init(autoreset=True)

    # parse case name and project name from the input argument
    case_name, proj_name = parse_arg(sys.argv)
    print(cl.Fore.BLUE + "Case: {}\nPath: {}\n".format(case_name, proj_name) +
          cl.Fore.RESET + "All outputs can be found in the presented path")

    # parse case settings stored in the .ini file
