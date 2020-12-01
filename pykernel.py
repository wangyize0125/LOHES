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


def init_cuda():
    # read the kernels file
    kernels = open("./kernels.cu", "r")
    kernels = Template("".join(kernels.readlines()))
    kernels = kernels.render()

    # compile the kernel
    module = SourceModule(kernels, no_extern_c=True)

    # log a debug information
    print(LOG + "CUDA environment initialized" + RESET)

    return module
