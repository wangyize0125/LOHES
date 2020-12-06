"""
@Filename: pykernel.py
@Author: Yize Wang
@Date: 2020/12/1
@Description: initialize cuda environment
"""

import pycuda.autoinit
from pycuda.compiler import SourceModule
from mako.template import Template

from settings import *


def init_cuda(settings):
    # read the kernels file
    kernels = open("./kernels.cu", "r")
    kernels = Template("".join(kernels.readlines()))
    kernels = kernels.render(
        num_inds=int(settings["global"]["num_individual"]),
        num_turbs=int(settings["wind_turbine"]["num_turbine"]),
        num_converters=int(settings["wave_energy_converter"]["num_converter"]),
    )

    # compile the kernel
    module = SourceModule(kernels, no_extern_c=True)

    # log a debug information
    print(LOG + "CUDA environment initialized" + RESET)

    return module
