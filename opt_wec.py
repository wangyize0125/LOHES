"""
@Filename: opt_wec.py
@Author: Yize Wang
@Date: 2020/12/3
@Description: optimize wave energy converter solely with predefined wind turbines
"""

import os
import time
import numpy as np
import geatpy as ga
from tqdm import tqdm
import matplotlib.pyplot as plt

from settings import *
from error_codes import *


class OptWEC(ga.Problem):
    """
        used to optimized wave energy converters solely
    """

    def __init__(self, settings, kernels):
        self.name = "OptWEC"
        # dimension of the target output: maximum energy output and minimum wave loads
        self.M = 2
        # 1: min, -1: max
        self.maxormins = [-1, 1]
        # dimensions of the variables: x and y coordinates
        self.Dim = int(settings["wave_energy_converter"]["num_converter"]) * 2
        # types of variables, 0: continuous, 1: discrete
        self.varTypes = [0] * self.Dim
        # lower limits of variables
        self.lb = [0] * self.Dim
        # upper limits of variables
        self.ub = [float(settings["global"]["length_x"])] * (self.Dim // 2) + \
                  [float(settings["global"]["length_y"])] * (self.Dim // 2)
        # if the borders are included
        self.lbin = [1] * self.Dim
        self.ubin = [1] * self.Dim
        self.settings = settings
        # spacing in x and y directions
        self.sx = float(settings["wave_energy_converter"]["sx_converter"])
        self.sy = float(settings["wave_energy_converter"]["sy_converter"])
        # kernels
        self.kernels = kernels
        # total number of generations
        self.total_gen = int(settings["global"]["max_generation"])
        # load wave distribution
        self.wave_dist = self.load_wave_dist()

        # display the info of this problem for debug
        self.display_info()

        ga.Problem.__init__(self, self.name, self.M, self.maxormins, self.Dim,
                            self.varTypes, self.lb, self.ub, self.lbin, self.ubin)

        self.pbar = tqdm(total=self.total_gen)

    def display_info(self):
        print("""
            A wave energy converter optimization problem is set up:
                Individual dimension: {},
                Length of the region in x and y: {}, {},
                Spacing in x and y: {}, {},
                Total generation: {}
        """.format(self.Dim, self.ub[0], self.ub[-1], self.sx, self.sy, self.total_gen) + RESET)

    def load_wave_dist(self):
        # open the wave distribution file
        file = open(
            os.path.join(
                self.settings["proj_name"],
                self.settings["wave_energy_converter"]["wave_distribution"]
            )
        )

        # parse the wave distribution file
        periods = np.array([float(item) for item in file.readline().split(":")[1].split(",")])
        heights = np.array([float(item) for item in file.readline().split(":")[1].split(",")])
        depth = float(file.readline().split(":")[1])
        file.readline()
        file.readline()
        file.readline()
        probs = np.array([[float(item) for item in line.split(",")] for line in file.readlines()])
        probs = probs / np.sum(probs)

        # calculate the accumulated wave period and height
        acc_T = np.sum(np.sum(probs, axis=0) * periods)
        acc_H = np.sum(np.sum(probs, axis=1) * heights[::-1])
        print(LOG + "Acc_T: {}\nAcc_H: {}\n".format(acc_T, acc_H) + RESET)

        # use a dictionary to store the wave distribution
        wave_dist = {"Ts": periods, "Hs": heights, "Ps": probs, "num_T": periods.size, "num_H": heights.size, "d": 0.35}

        return wave_dist

    def aimFunc(self, pop):
        # update progress bar
        self.pbar.update(1)

        # pick out individuals
        inds = pop.Phen

        # sort the wave energy converters according to the x coordinate
        # these codes are transplanted to GPU kernels

        # calculate the constraint values first, no needs of ordering
        pop.CV = self.cal_cv(inds).reshape((-1, 1))

        # objective values
        pop.ObjV = np.zeros((inds.shape[0], 1))
        # iterate each wave period and height combination
        for j in range(self.wave_dist["num_T"]):
            for i in range(self.wave_dist["num_H"]):
                if self.wave_dist["Ps"][i][j] <= 1E-10:
                    # no wave
                    continue
                else:
                    # calculate the energy output corresponding this wave period and height
                    pop.ObjV = self.predict_energy(inds)
        pop.ObjV = pop.ObjV.reshape((-1, 2))

    def cal_cv(self, inputs):
        return np.zeros((inputs.shape[0], 1)) - 1

    def predict_energy(self, inputs):
        return np.ones(inputs.shape[0] * 2)


def run_wave_energy_converter(problem):
    """
        optimize the wave energy converters solely
    """

    # settings for genetic algorithm, most parameters can be specified through the .ini file
    Encoding = "RI"     # encoding of the individuals, RI means use real number to encode
    NIND = int(problem.settings["global"]["num_individual"])    # number of individuals
    Field = ga.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
    population = ga.Population(Encoding, Field, NIND)   # generate the population instance
    myAlgo = ga.moea_NSGA2_templet(problem, population)     # initialize the problem
    myAlgo.MAXGEN = int(problem.settings["global"]["max_generation"])  # number of generations
    myAlgo.mutOper.F = float(problem.settings["global"]["mut_factor"])  # mutation factor
    myAlgo.mutOper.Pm = float(problem.settings["global"]["mut_prob"])  # mutation probability
    myAlgo.recOper.XOVR = float(problem.settings["global"]["cor_factor"])  # crossover factor
    myAlgo.drawing = 0  # 0: no drawing; 1: drawing at the end of process; 2: animated drawing
    myAlgo.logTras = int(problem.settings["global"]["log_trace"])  # record log every * steps
    myAlgo.verbose = bool(int(problem.settings["global"]["log_by_print"]))  # whether print log

    # display running information for debug
    print(TIP + """
        Genetic algorithm will start with:
            Number of individuals: {},
            Number of generations: {},
            Mutation factor: {},
            Mutation Probability: {},
            Crossover factor: {}
    """.format(NIND, myAlgo.MAXGEN, myAlgo.mutOper.F, myAlgo.mutOper.Pm, myAlgo.recOper.XOVR) + RESET)

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
    if len(myAlgo.log["gen"]) == 0 or best_ind.sizes == 0:
        print(ERROR + "Genetic algorithm failed!!!" + RESET)
        print(TIP + "You can re-execute these codes by amplifying your target wave farm." + RESET)
        exit(ga_failed)

    print(TIP + "Genetic algorithm finds {} solutions".format(best_ind.sizes) + RESET)

    # otherwise, genetic algorithm succeed
    # output: optimization results of wave energy converters
    best_ind.save(os.path.join(problem.settings["proj_name"], "record_array"))
    file = open(os.path.join(problem.settings["proj_name"], "record_array", "record.txt"), "w")
    file.write("# Used time: {} s\n".format(interval_t))
    file.write("# HV\n")
    if myAlgo.log:
        file.write(
            "\n".join(
                ["{:.6f}".format(myAlgo.log["hv"][gen]) for gen in range(len(myAlgo.log["gen"]))]
            )
        )
    file.close()

    # output: plot the HV curves
    fig = plt.figure(figsize=(10, 10))
    plt.plot(myAlgo.log["hv"])
    plt.savefig(os.path.join(problem.settings["proj_name"], "record_array", "hvs.png"))

    # output: plot the wave energy converter array figure
    # find out the best wave energy converter array layouts first
    # owing to that I want to use WEC to reduce the wave loads
    # hence, I found out the layout with the smallest wave loads
    file = open(os.path.join(problem.settings["proj_name"], "record_array", "ObjV.csv"))
    # read out all of the objective values
    objvs = np.array([[float(item) for item in line.split(",")] for line in file.readlines()])
    # find out the smallest objective value of wave loads
    index = np.argmin(objvs[:, 1])
    # find out the maximum energy output
    index = np.argmax(objvs[:, 0][np.argwhere(objvs[:, 1] / objvs[index, 1] <= 1.05)])
    # then, index indicates where the best individual exists
    fig = plt.figure(figsize=(10, 10))
    file = open(os.path.join(problem.settings["proj_name"], "record_array", "Phen.csv"))
    # skip the former individuals and read out the best one
    for i in range(index):
        file.readline()
    # read out the best individual
    var_trace = [float(item) for item in file.readline().split(",")]
    for i in range(len(var_trace) // 2):
        # plt.scatter([var_trace[i]], [var_trace[i + len(var_trace) // 2]], s=100, c="k", zorder=2)
        plt.plot(
            [var_trace[i], var_trace[i]],
            [var_trace[i + len(var_trace) // 2] - 12.5, var_trace[i + len(var_trace) // 2] + 12.5],
            "k-", linewidth=5, zorder=2
        )

        # tune the plot limits
        plt.xlim((0, problem.ub[0]))
        plt.ylim((0, problem.ub[-1]))
    plt.savefig(os.path.join(problem.settings["proj_name"], "record_array", "array.png"))
