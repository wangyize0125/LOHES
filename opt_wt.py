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
import matplotlib.pyplot as plt
from pycuda import driver as drv

from settings import *


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

        return direcs, vels, wind_distribution

    def aimFunc(self, pop):
        # pick out individuals
        inds = pop.Phen

        # sort the turbines according to the x coordinate
        x_coordinate, y_coordinate = inds[:, :self.Dim // 2], inds[:, self.Dim // 2:]
        sort_indices = np.argsort(x_coordinate, axis=1)
        x_coordinate = np.take_along_axis(x_coordinate, sort_indices, axis=1)
        y_coordinate = np.take_along_axis(y_coordinate, sort_indices, axis=1)
        inds = np.hstack((x_coordinate, y_coordinate))

        # calculate the constraint values first
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
                    # calculate the energy output corresponding this wind
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
        k_num = np.int32(rows)
        k_nd = np.int32(cols // 2)

        # pick out the function
        func = self.kernels.get_function("cal_cv")
        func(drv.In(k_layouts), drv.Out(k_cvs), drv.In(k_sx), drv.In(k_sy), drv.In(k_num), drv.In(k_nd),
             grid=(10, 1, 1), block=(int(rows // 10 + 1), 1, 1))

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
        k_prob = np.float32(prob).flatten()
        k_num = np.int32(rows).flatten()
        k_nd = np.int32(cols // 2).flatten()

        # predict energy
        func = self.kernels.get_function("pre_energy")
        func(drv.In(k_layouts), drv.InOut(k_energys), drv.In(k_direc), drv.In(k_vel), drv.In(k_start_vel),
             drv.In(k_cut_vel), drv.In(k_prob), drv.In(k_num), drv.In(k_nd),
             grid=(10, 1, 1), block=(int(rows // 10 + 1), 1, 1))

        return k_energys


def run_wind_turbine(problem):
    """
        optimize the wind turbines solely
    """

    # settings for genetic algorithm, most parameters can be specified through the .ini file
    Encoding = "RI"
    NIND = int(problem.settings["global"]["num_individual"])
    Field = ga.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
    population = ga.Population(Encoding, Field, NIND)
    myAlgo = ga.soea_DE_best_1_L_templet(problem, population)
    myAlgo.MAXGEN = int(problem.settings["global"]["max_generation"])
    myAlgo.mutOper.F = float(problem.settings["global"]["mut_factor"])
    myAlgo.recOper.XOVR = float(problem.settings["global"]["cor_factor"])
    myAlgo.drawing = 0

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
    [best_ind, population] = myAlgo.run()
    # time interval
    end_t = time.time()
    interval_t = end_t - start_t

    # output: optimization results of wind turbines
    best_ind.save(os.path.join(problem.settings["proj_name"], "record_array"))
    file = open(os.path.join(problem.settings["proj_name"], "record_array", "record.txt"), "w")
    file.write("# Best target output: {}\n".format(best_ind.ObjV[0][0]))
    file.write("# Used time: {} s\n".format(interval_t))
    file.close()

    # output: plot the wind turbine array figure
    fig = plt.figure(figsize=(10, 10))
    file = open(os.path.join(problem.settings["proj_name"], "record_array", "Phen.csv"))
    var_trace = [float(item) for item in file.readline().split(",")]
    for i in range(len(var_trace) // 2):
        plt.scatter([var_trace[i]], [var_trace[i + len(var_trace) // 2]], s=30)
    plt.savefig(os.path.join(problem.settings["proj_name"], "record_array", "record_array.png"))

    # # output: plot the objv curves
    # fig = plt.figure(figsize=(10, 10))
    # file = open(os.path.join(problem.settings["proj_name"], "record_array.txt"), "r")
    # file.readline()
    # file.readline()
    # file.readline()
    # file.readline()
    # file.readline()
    # data = np.array([[float(item) for item in line.split()] for line in file.readlines()])
    # plt.plot(data[:, 0], label="min")
    # plt.plot(data[:, 1], label="max")
    # plt.legend()
    # plt.savefig(os.path.join(problem.settings["proj_name"], "record_array_objv.png"))
