"""
@Filename: settings.py
@Author: Yize Wang
@Date: 2020/11/30
@Description: global settings for all codes
"""

# output in color
import colorama as cl

ERROR = cl.Fore.RED
LOG = cl.Fore.BLUE
TIP = cl.Fore.YELLOW
RESET = cl.Fore.RESET

# default values for case settings
default_settings = {
    # for global settings
    "global": {
        "length_x": 20000.0,
        "length_y": 20000.0,
        "num_individual": 600,
        "max_generation": 1000,
        "mut_factor": 0.2,
        "mut_prob": 0.2,
        "cor_factor": 0.9,
        "log_trace": 1,
        "log_by_print": 0,
    },

    # for wind turbine layout optimization
    "wind_turbine": {
        "num_turbine": 16,
        "sx_turbine": 1000.0,
        "sy_turbine": 1000.0,
        # I hope these codes can be utilized to optimize the layouts of wave energy converters for existing
        # offshore wind turbine farm. Hence, if predefined layouts are given, only converters will be optimized
        # by default, no predefined layouts are given
        "pre_layouts": None,
        "wind_distribution": "wind_distribution.txt",
        "fineness_direc": 24,
        "fineness_vel": 10,
        "max_wv": 25,
        "start_vel": 3.0,
        "cut_vel": 25.0,
        "turbulence_intensity": 0.8,
        "rotor_diameter": 90,
        "cp": 0.4,
        "plot_wind": 12,
        "ignore_direction": 1,
        "ignore_velocity": 1,
        "ignore_direc_vel": 1,
        "test": None,
    },

    # for wave energy converter layout optimization
    "wave_energy_converter": {
        "num_converter": 20,
        "sx_converter": 100.0,
        "sy_converter": 100.0,
        "converter": True,
        "wave_distribution": None,
    },
}
