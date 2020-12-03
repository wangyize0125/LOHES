"""
@Filename: opt_wt.py
@Author: Yize Wang
@Date: 2020/12/1
@Description: optimize wind turbines solely
"""

import os
import time
import numpy as np
import geatpy as ga
from tqdm import tqdm
import matplotlib.pyplot as plt
from pycuda import driver as drv

from settings import *
from error_codes import *


class OptWT(ga.Problem):
    """
        used to optimize the wind turbines solely
    """

    def __init__(self, settings, kernels):
        self.name = "OptWT"
        # dimension of the target output: maximum energy output
        self.M = 1
        # 1: min, -1: max
        self.maxormins = [-1]
        # dimensions of the variables: x and y of wind turbines
        self.Dim = int(settings["wind_turbine"]["num_turbine"]) * 2
        # types of variables, 0: continuous, 1: discrete
        self.varTypes = [0] * self.Dim
        # lower limits of variables
        self.lb = [0] * self.Dim
        # upper limits of variables
        self.ub = [float(settings["global"]["length_x"])] * (self.Dim // 2) +\
                  [float(settings["global"]["length_y"])] * (self.Dim // 2)
        # if the borders are included
        self.lbin = [1] * self.Dim
        self.ubin = [1] * self.Dim
        self.settings = settings
        # spacing in x and y directions
        self.sx = float(settings["wind_turbine"]["sx_turbine"])
        self.sy = float(settings["wind_turbine"]["sy_turbine"])
        # wind distribution
        self.direcs, self.vels, self.wind_dist = self.handle_wind_distribution(
            os.path.join(self.settings["proj_name"], self.settings["wind_turbine"]["wind_distribution"])
        )
        # kernels
        self.kernels = kernels
        # total number of generations
        self.total_gen = int(settings["global"]["max_generation"])

        # display the info of this problem for debug
        self.display_info()

        ga.Problem.__init__(self, self.name, self.M, self.maxormins, self.Dim,
                            self.varTypes, self.lb, self.ub, self.lbin, self.ubin)

        # progress bar
        self.pbar = tqdm(total=self.total_gen)

    def display_info(self):
        print("""
            A wind turbine optimization problem is set up:
                Individual dimension: {},
                Length of the region in x and y: {}, {},
                Spacing in x and y: {}, {},
                Total generation: {}
        """.format(self.Dim, self.ub[0], self.ub[-1], self.sx, self.sy, self.total_gen) + RESET)

    def handle_wind_distribution(self, filename):
        """
        :param filename:
        :return: wind direction and wind velocity distribution
        """

        # open file
        file = open(filename, "r")
        # step over the first line
        file.readline()

        # wind distribution array with row represents velocities and column represents directions
        wind_distribution = np.zeros((int(self.settings["wind_turbine"]["fineness_direc"]),
                                      int(self.settings["wind_turbine"]["fineness_vel"])))
        line_number = 0  # record how many lines the file has in total
        step_direc = 360 / int(self.settings["wind_turbine"]["fineness_direc"])
        # directions used for loop when calculating
        direcs = np.linspace(0, 360, int(self.settings["wind_turbine"]["fineness_direc"]), endpoint=False)
        # maximum wind velocity is 25 m s-1 in this study. To alter it later, use a parameter here
        step_vel = float(self.settings["wind_turbine"]["max_wv"]) / int(self.settings["wind_turbine"]["fineness_vel"])
        # velocities used for loop when calculating
        vels = np.linspace(0, float(self.settings["wind_turbine"]["max_wv"]),
                           int(self.settings["wind_turbine"]["fineness_vel"]), endpoint=False) + step_vel / 2
        for line in file.readlines():
            data = line.split()
            # compute row and column index
            column_id = int(float(data[0]) // step_vel)
            row_id = int(float(data[1]) // step_direc)
            # plus this data into distribution
            wind_distribution[row_id, column_id] += 1

            line_number += 1
        file.close()
        # calculate probability
        wind_distribution /= line_number

        # from the simulation results, it was found that the optimization results approximately
        # face the major wind direction. Hence, I add an option to disable wind direction
        if bool(int(self.settings["wind_turbine"]["ignore_direction"])):
            direcs = np.array([0])
            wind_distribution = np.sum(wind_distribution, axis=0).reshape((1, -1))

        return direcs, vels, wind_distribution

    def aimFunc(self, pop):
        # update progress bar
        self.pbar.update(1)

        # pick out individuals
        inds = pop.Phen

        # sort the turbines according to the x coordinate
        # these codes are transplanted to GPU kernels

        # calculate the constraint values first, no needs of ordering
        pop.CV = self.cal_cv(inds).reshape((-1, 1))

        # objective values
        pop.ObjV = np.zeros((inds.shape[0], 1))
        # iterate each wave period and height combination
        for j in range(self.vels.size):
            for i in range(self.direcs.size):
                if self.wind_dist[i][j] <= 1E-10:
                    # no wind
                    continue
                else:
                    # calculate the energy output corresponding this wind, needs ordering and rotating
                    pop.ObjV = self.predict_energy(inds, pop.ObjV, self.direcs[i], self.vels[j], self.wind_dist[i][j])
        pop.ObjV = pop.ObjV.reshape((-1, 1))

    def cal_cv(self, inputs):
        """
            calculate the constraint values for each individual
        """

        rows, cols = inputs.shape

        # prepare data
        k_layouts = np.float32(inputs).flatten()
        k_cvs = np.float32(np.zeros(rows))
        k_sx = np.float32(self.sx)
        k_sy = np.float32(self.sy)

        # pick out the function
        func = self.kernels.get_function("cal_cv_turb")
        func(drv.In(k_layouts), drv.Out(k_cvs), drv.In(k_sx), drv.In(k_sy),
             grid=(int(rows // 10 + 1), 1, 1), block=(10, 1, 1))

        return k_cvs

    def predict_energy(self, inputs, objv, direc, vel, prob):
        """
            predict the energy output of the wind turbines
        """

        rows, cols = inputs.shape

        # prepare data
        k_layouts = np.float32(inputs).flatten()
        k_energys = np.float32(objv).flatten()
        k_direc = np.float32(direc).flatten()
        k_vel = np.float32(vel).flatten()
        k_start_vel = np.float32(self.settings["wind_turbine"]["start_vel"]).flatten()
        k_cut_vel = np.float32(self.settings["wind_turbine"]["cut_vel"]).flatten()
        k_turb_int = np.float32(self.settings["wind_turbine"]["turbulence_intensity"]).flatten()
        k_prob = np.float32(prob).flatten()
        k_ct = np.float32(self.settings["wind_turbine"]["ct"]).flatten()
        k_rad = np.float32(self.settings["wind_turbine"]["rotor_diameter"]).flatten()
        k_cp = np.float32(self.settings["wind_turbine"]["cp"]).flatten()

        # predict energy
        func = self.kernels.get_function("pre_energy_turb")
        func(drv.In(k_layouts), drv.InOut(k_energys), drv.In(k_direc), drv.In(k_vel), drv.In(k_start_vel),
             drv.In(k_cut_vel), drv.In(k_turb_int), drv.In(k_prob), drv.In(k_ct), drv.In(k_rad), drv.In(k_cp),
             grid=(int(rows // 10 + 1), 1, 1), block=(10, 1, 1))

        return k_energys

    def plot_field(self, ind):
        # generate coordinates in x and y
        x_array = np.linspace(0, self.ub[0], 100, endpoint=True)
        y_array = np.linspace(0, self.ub[-1], 100, endpoint=True)
        x_array, y_array = np.meshgrid(x_array, y_array)

        # here, we need to sort the turbines owing to that we will use predict_energy later to
        # predict the wind velocity and turbulence intensity ahead the wind turbines, in which
        # wind turbines will be sorted. If we use unsorted turbines continuously, they cant
        # correspond each other one-by-one. So, we sort the decides first:
        x_coordinate, y_coordinate = np.array(ind[:self.Dim // 2]), np.array(ind[self.Dim // 2:])
        sort_indices = np.argsort(x_coordinate)
        x_coordinate = x_coordinate[sort_indices].flatten()
        y_coordinate = y_coordinate[sort_indices].flatten()
        ind = np.hstack((x_coordinate, y_coordinate))

        # first, call predict energy function
        # then, the first rows of wind_vels and turb_ints on GPU
        # maintain those of this ind, we can step in
        self.predict_energy(np.array(ind).reshape((1, -1)), 0, 0, float(self.settings["wind_turbine"]["plot_wind"]), 1)

        # prepare data
        k_input = np.float32(ind).flatten()
        k_x = np.float32(x_array).flatten()
        k_y = np.float32(y_array).flatten()
        k_winds = np.float32(np.zeros_like(k_x)).flatten()
        k_turbs = np.float32(np.zeros_like(k_x)).flatten()
        k_num = np.int32(k_x.size).flatten()
        k_plot_wind = np.float32(float(self.settings["wind_turbine"]["plot_wind"])).flatten()
        k_plot_turb = np.float32(float(self.settings["wind_turbine"]["turbulence_intensity"])).flatten()
        k_ct = np.float32(self.settings["wind_turbine"]["ct"]).flatten()
        k_rad = np.float32(self.settings["wind_turbine"]["rotor_diameter"]).flatten()

        func = self.kernels.get_function("plot_field_turb")
        func(drv.In(k_input), drv.In(k_x), drv.In(k_y), drv.Out(k_winds), drv.Out(k_turbs), drv.In(k_num),
             drv.In(k_plot_wind), drv.In(k_plot_turb), drv.In(k_ct), drv.In(k_rad),
             grid=(int(k_x.size // 20 + 1), 1, 1), block=(20, 1, 1))

        # reshape winds and turbs
        wind_vels = k_winds.reshape(x_array.shape)
        turb_ints = k_turbs.reshape(x_array.shape)

        return x_array, y_array, wind_vels, turb_ints


def run_wind_turbine(problem):
    """
        optimize the wind turbines solely
    """

    # settings for genetic algorithm, most parameters can be specified through the .ini file
    Encoding = "RI"     # encoding of the individuals, RI means use real numbers to encode
    NIND = int(problem.settings["global"]["num_individual"])    # number of individuals
    Field = ga.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
    population = ga.Population(Encoding, Field, NIND)   # generate the population instance
    myAlgo = ga.soea_DE_best_1_L_templet(problem, population)   # initialize the problem
    myAlgo.MAXGEN = int(problem.settings["global"]["max_generation"])   # number of generations
    myAlgo.mutOper.F = float(problem.settings["global"]["mut_factor"])  # mutation factor
    myAlgo.recOper.XOVR = float(problem.settings["global"]["cor_factor"])   # crossover factor
    myAlgo.drawing = 0  # 0: no drawing; 1: drawing at the end of process; 2: animated drawing
    myAlgo.logTras = int(problem.settings["global"]["log_trace"])   # record log every * steps
    myAlgo.verbose = bool(int(problem.settings["global"]["log_by_print"]))  # whether print log

    # display running information for debug
    print(TIP + """
        Genetic algorithm will start with:
            Number of individuals: {},
            Number of generations: {},
            Mutation factor: {},
            Crossover factor: {}
    """.format(NIND, myAlgo.MAXGEN, myAlgo.mutOper.F, myAlgo.recOper.XOVR) + RESET)

    # record computational time
    start_t = time.time()
    # run genetic algorithm
    [best_ind, population] = myAlgo.run()
    # delete progress bar in problem
    problem.pbar.close()
    # time interval
    end_t = time.time()
    interval_t = end_t - start_t

    # judge whether genetic algorithm failed
    if myAlgo.log["gen"] == 0:
        print(ERROR + "Genetic algorithm failed!!!" + RESET)
        print(TIP + "You can re-execute these codes by amplifying your target wave farm." + RESET)
        exit(ga_failed)
    # otherwise, genetic algorithm succeed
    # output: optimization results of wind turbines
    best_ind.save(os.path.join(problem.settings["proj_name"], "record_array"))
    file = open(os.path.join(problem.settings["proj_name"], "record_array", "record.txt"), "w")
    file.write("# Used time: {} s\n".format(interval_t))
    file.write("# average_objv, maximum_objv\n")
    if myAlgo.log:
        file.write(
            "\n".join(
                ["{:.6f}, {:.6f}".format(myAlgo.log["f_avg"][gen], myAlgo.log["f_max"][gen]) for gen in range(len(myAlgo.log["gen"]))]
            )
        )
    file.close()

    # output: plot the objv curves
    fig = plt.figure(figsize=(10, 10))
    plt.plot(myAlgo.log["f_max"], label="max")
    plt.plot(myAlgo.log["f_avg"], label="avg")
    plt.legend()
    plt.savefig(os.path.join(problem.settings["proj_name"], "record_array", "objvs.png"))

    # output: plot the wind turbine array figure
    fig = plt.figure(figsize=(16, 8))
    file = open(os.path.join(problem.settings["proj_name"], "record_array", "Phen.csv"))
    # read out the best individual
    var_trace = [float(item) for item in file.readline().split(",")]
    # plot the best individual in tow sub plots
    for sub in range(1, 3):
        plt.subplot(1, 2, sub)
        for i in range(len(var_trace) // 2):
            # plt.scatter([var_trace[i]], [var_trace[i + len(var_trace) // 2]], s=100, c="k", zorder=2)
            plt.plot(
                [var_trace[i], var_trace[i]],
                [var_trace[i + len(var_trace) // 2] - float(problem.settings["wind_turbine"]["rotor_diameter"]),
                 var_trace[i + len(var_trace) // 2] + float(problem.settings["wind_turbine"]["rotor_diameter"])],
                "k-", linewidth=5, zorder=2
            )

        # tune the sub plots limits
        plt.xlim((0, problem.ub[0]))
        plt.ylim((0, problem.ub[-1]))
    # compute flow fields of the best individual
    x_array, y_array, wind_vels, turb_ints = problem.plot_field(var_trace)
    # plot wind velocity and turbulence intensity
    plt.subplot(1, 2, 1)
    plt.contourf(x_array, y_array, wind_vels, zorder=1, cmap="bwr")
    plt.subplot(1, 2, 2)
    plt.contourf(x_array, y_array, turb_ints, zorder=1, cmap="bwr")
    # output flow fields into a file
    x_array = x_array.flatten()
    y_array = y_array.flatten()
    wind_vels = wind_vels.flatten()
    turb_ints = turb_ints.flatten()
    file = open(os.path.join(problem.settings["proj_name"], "record_array", "flow_fields.txt"), "w")
    file.write("# x, y, wind_velocity, turb_intensity\n")
    for i in range(x_array.size):
        file.write("{}, {}, {}, {}\n".format(x_array[i], y_array[i], wind_vels[i], turb_ints[i]))
    file.close()
    plt.savefig(os.path.join(problem.settings["proj_name"], "record_array", "array.png"))
