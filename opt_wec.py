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
from pycuda import driver as drv

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
        self.settings = settings
        # spacing in x and y directions
        self.sx = float(settings["wave_energy_converter"]["sx_converter"])
        self.sy = float(settings["wave_energy_converter"]["sy_converter"])
        # load pre_layouts of the wind turbines
        self.pre_layouts, self.corrections = self.load_pre_layouts()
        # upper limits of variables
        self.ub = [float(settings["global"]["length_x"])] * (self.Dim // 2) + \
                  [float(settings["global"]["length_y"])] * (self.Dim // 2)
        # if the borders are included
        self.lbin = [1] * self.Dim
        self.ubin = [1] * self.Dim
        # kernels
        self.kernels = kernels
        # total number of generations
        self.total_gen = int(settings["global"]["max_generation"])
        # load wave distribution
        self.wave_dist = self.load_wave_dist()
        # load wake and energy output model
        self.wake_model, self.energy_model = self.parse_model()
        # new feature: use global memory to store the layouts of the wave energy converter
        # to save the time for transferring data between host and device
        self.g_layouts = drv.mem_alloc(int(self.settings["global"]["num_individual"]) * self.Dim * 4)
        # wave heights for the wind turbines
        self.g_wave_heights = None

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

    def parse_model(self):
        """
            load wake model of wave energy converters
        """

        file = open("./func_and_paras.txt", "r")
        models = []

        for line in file.readlines():
            if line[0] in ("#", "\n"):
                # comment line
                continue
            else:
                models.append([float(item) for item in line.split()])
        models = np.array(models)

        # pick out wake model and energy output model
        m_wake = models[0:-1, :]
        m_energy = models[-1, :]
        file.close()

        return m_wake, m_energy

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

        if bool(int(self.settings["wave_energy_converter"]["ignore_period"])):
            print(LOG + "Ignore wave period when optimizing" + RESET)
            max_period_idx = np.argmax(np.sum(probs, axis=0))
            periods = np.array([periods[max_period_idx]])
            probs = probs[:, max_period_idx].reshape((-1, 1))
            probs /= np.sum(probs)
        if bool(int(self.settings["wave_energy_converter"]["ignore_height"])):
            print(LOG + "Ignore wave height when optimizing" + RESET)
            max_height_idx = np.argmax(np.sum(probs, axis=1))
            heights = np.array([heights[heights.size - max_height_idx - 1]])
            probs = probs[max_height_idx, :].reshape((1, -1))
            probs /= np.sum(probs)
        if bool(int(self.settings["wave_energy_converter"]["ignore_period_height"])):
            print(LOG + "Ignore wave period and height when optimizing" + RESET)
            max_height_idx, max_period_idx = np.argwhere(probs == np.max(probs))[0]
            heights = np.array([heights[heights.size - max_height_idx - 1]])
            periods = np.array([periods[max_period_idx]])
            probs = np.array([1]).reshape((1, 1))

        # use a dictionary to store the wave distribution
        wave_dist = {"Ts": periods, "Hs": heights, "Ps": probs, "num_T": periods.size, "num_H": heights.size, "d": 0.35}

        return wave_dist

    def load_pre_layouts(self):
        """
            load pre_layouts of the wind turbines
        """

        file = open(os.path.join(self.settings["proj_name"], self.settings["wind_turbine"]["pre_layouts"]), "r")

        x, y, corrections = [], [], np.zeros(2)
        for line in file.readlines():
            if line[0] in ("#", "\n"):
                # comment line
                continue
            else:
                # read x, y of the predefined wind turbines
                data = [float(item) for item in line.split(",")]
                x.append(data[0])
                y.append(data[1])
        x = np.array(x)
        y = np.array(y)

        # some corrections on length of the wave energy farm
        corrections[1] += (float(self.settings["global"]["length_y"]) - np.max(y) - np.min(y)) / 2
        y = y + corrections[1]
        if np.min(x) < 2 * self.sx:
            x = x + 2 * self.sx
            corrections[0] += 2 * self.sx
        if np.min(x) < float(self.settings["global"]["length_x"]) / 5:
            x = x + float(self.settings["global"]["length_x"]) / 5
            corrections[0] += float(self.settings["global"]["length_x"]) / 5
        if np.max(x) > float(self.settings["global"]["length_x"]):
            self.settings["global"]["length_x"] = float(self.settings["global"]["length_x"]) + 2 * self.sx
            print(LOG + "Length of the wave energy farm in x direction has been changed to {}".format(self.settings["global"]["length_x"]) + RESET)

        return np.hstack((x, y)), corrections

    def aimFunc(self, pop):
        # update progress bar
        self.pbar.update(1)

        # pick out individuals
        inds = pop.Phen.astype(np.float32)

        if self.settings["wave_energy_converter"]["test"] is not "":
            file = open(os.path.join(self.settings["proj_name"], self.settings["wave_energy_converter"]["test"]), "r")
            correct = [float(item) for item in file.readline().split(",")]
            data = []
            temp = []
            for line in file.readlines():
                if line.startswith("#"):
                    if len(temp) > 0:
                        data.append(temp)
                    temp = []
                    continue
                x_and_y = line.split(",")
                temp.insert(len(temp) // 2, float(x_and_y[0]) + correct[0])
                temp.append(float(x_and_y[1]) + correct[1])
            data.append(temp)
            data = np.array(data)
            data = np.vstack((data, np.zeros_like(data[0]) + 10 ** 10))
            inds[: data.shape[0], :] = data

        # copy the current individuals to the device
        drv.memcpy_htod(self.g_layouts, inds)

        # sort the wave energy converters according to the x coordinate
        # these codes are transplanted to GPU kernels
        self.sort_converters(inds.shape[0])

        # calculate the constraint values first, no needs of ordering
        pop.CV = self.cal_cv(inds.shape[0])

        # calculate the objective values of the individuals
        pop.ObjV = self.predict_energy(inds.shape[0])

        if self.settings["wave_energy_converter"]["test"] is not "":
            # calculate wave loads
            wave_loads = self.cal_wave_loads(data.shape[0])

            file = open(os.path.join(self.settings["proj_name"], "test_wave_energy_converter_objv.txt"), "w")
            file.write("# energy, max, min, avg, max, min, avg\n")
            for i in range(data.shape[0]):
                file.write("{}, {}, {}, {}, {}, {}, {}\n".format(pop.ObjV[i, 0], *wave_loads[i].tolist()))
            file.close()
            exit(0)

    def sort_converters(self, rows):
        """
            sort the wave energy converters on GPU
        """

        # run the kernel
        func = self.kernels.get_function("sort_converters")
        func(self.g_layouts, grid=(int(rows // 20 + 1), 1, 1), block=(20, 1, 1))

    def cal_cv(self, rows):
        """
            calculate the constrain values for each individual
        """

        # prepare data
        g_cvs = np.zeros(rows, dtype=np.float32)
        g_sx = np.float32(self.sx).flatten()
        g_sy = np.float32(self.sy).flatten()
        g_pre_layouts = np.float32(self.pre_layouts).flatten()
        g_num_pre_layouts = np.int32(self.pre_layouts.size // 2).flatten()

        # run the kernel function
        func = self.kernels.get_function("cal_cv_converter")
        func(self.g_layouts, drv.Out(g_cvs), drv.In(g_sx), drv.In(g_sy),
             drv.In(g_pre_layouts), drv.In(g_num_pre_layouts),
             grid=(int(rows // 10 + 1), 1, 1), block=(10, 1, 1))

        return g_cvs.reshape((-1, 1))

    def predict_energy(self, rows):
        """
            calculate the energy outputs of the wave energy converters
        """

        # prepare data
        g_energys = np.zeros(rows * 2, dtype=np.float32)
        g_periods = np.float32(self.wave_dist["Ts"]).flatten()
        g_num_T = np.int32(self.wave_dist["num_T"]).flatten()
        g_heights = np.float32(self.wave_dist["Hs"]).flatten()
        g_num_H = np.int32(self.wave_dist["num_H"]).flatten()
        g_probs = np.float32(self.wave_dist["Ps"]).flatten()
        g_wake_model = np.float32(self.wake_model).flatten()
        g_energy_model = np.float32(self.energy_model).flatten()
        g_pre_layouts = np.float32(self.pre_layouts).flatten()
        g_num_pre_layouts = np.int32(self.pre_layouts.size // 2).flatten()
        self.g_wave_heights = drv.mem_alloc(4 * int(self.settings["global"]["num_individual"]) * int(g_num_pre_layouts[0]))
        g_temp = drv.mem_alloc(4 * int(self.settings["global"]["num_individual"]) * int(g_num_pre_layouts[0]))

        # run the kernel
        func = self.kernels.get_function("pre_energy_converter")
        func(self.g_layouts, drv.Out(g_energys), drv.In(g_periods), drv.In(g_num_T), drv.In(g_heights),
             drv.In(g_num_H), drv.In(g_probs), drv.In(g_wake_model), drv.In(g_energy_model),
             drv.In(g_pre_layouts), drv.In(g_num_pre_layouts), self.g_wave_heights, g_temp,
             grid=(int(rows // 10 + 1), 1, 1), block=(10, 1, 1))

        return g_energys.reshape((-1, 2))

    def cal_wave_loads(self, num):
        """
            calculate wave loads of the wind turbines
        """

        g_num_pre_layouts = np.int32(self.pre_layouts.size // 2).flatten()
        g_num_test = np.int32(num).flatten()
        wave_loads = np.zeros((num, self.pre_layouts.size), dtype=np.float32).flatten()

        func = self.kernels.get_function("cal_wave_loads")
        func(self.g_wave_heights, drv.In(g_num_pre_layouts), drv.In(g_num_test), drv.Out(wave_loads),
             grid=(int(int(self.pre_layouts.size // 2 * num) // 3 + 1), 1, 1), block=(3, 1, 1))

        wave_loads = wave_loads.reshape((num * 2, -1))
        max_w, min_w, avg_w = np.max(wave_loads, axis=1), np.min(wave_loads, axis=1), np.average(wave_loads, axis=1)
        wave_loads = np.vstack((max_w, min_w, avg_w)).T
        wave_loads = np.hstack((wave_loads[: wave_loads.shape[0] // 2], wave_loads[wave_loads.shape[0] // 2:]))

        return wave_loads


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
    # HV curves are not outputted owing to the strong non-linearity
    # # output: plot the HV curves
    # fig = plt.figure(figsize=(10, 10))
    # plt.plot(myAlgo.log["hv"])
    # plt.savefig(os.path.join(problem.settings["proj_name"], "record_array", "hvs.png"))
    best_ind.save(os.path.join(problem.settings["proj_name"], "record_array"))

    # output: plot the wave energy converter array figure
    # find out the best wave energy converter array layouts first
    # owing to that I want to use WEC to reduce the wave loads
    # hence, I found out the layout with the smallest wave loads
    file = open(os.path.join(problem.settings["proj_name"], "record_array", "ObjV.csv"))
    # read out all of the objective values
    objvs = np.array([[float(item) for item in line.split(",")] for line in file.readlines()])
    # find out the smallest objective value of wave loads
    index = np.argmin(objvs[:, 1])
    # # find out the maximum energy output
    # index = np.argmax(objvs[:, 0][np.argwhere(objvs[:, 1] / objvs[index, 1] <= 1.05)])
    # then, index indicates where the best individual exists
    file = open(os.path.join(problem.settings["proj_name"], "record_array", "Phen.csv"))
    # skip the former individuals and read out the best one
    for i in range(index):
        file.readline()
    # read out the best individual
    var_trace = [float(item) for item in file.readline().split(",")]
    # change them to real coordinates
    # wave energy converter
    var_trace[:len(var_trace) // 2] -= problem.corrections[0]
    var_trace[len(var_trace) // 2:] -= problem.corrections[1]
    # wind turbines
    problem.pre_layouts[:problem.pre_layouts.size // 2] -= problem.corrections[0]
    problem.pre_layouts[problem.pre_layouts.size // 2:] -= problem.corrections[1]
    # plot wind turbines and wave energy converters
    fig = plt.figure(figsize=(10, 10))
    for i in range(len(var_trace) // 2):
        # plt.scatter([var_trace[i]], [var_trace[i + len(var_trace) // 2]], s=100, c="k", zorder=2)
        plt.plot(
            [var_trace[i], var_trace[i]],
            [var_trace[i + len(var_trace) // 2] - 12.5, var_trace[i + len(var_trace) // 2] + 12.5],
            "k-", linewidth=5, zorder=2
        )
    for i in range(problem.pre_layouts.size // 2):
        plt.plot(
            [problem.pre_layouts[i], problem.pre_layouts[i]],
            [problem.pre_layouts[i + problem.pre_layouts.size // 2] - float(problem.settings["wind_turbine"]["rotor_diameter"]) / 2,
             problem.pre_layouts[i + problem.pre_layouts.size // 2] + float(problem.settings["wind_turbine"]["rotor_diameter"]) / 2],
            "r-", linewidth=5, zorder=2
        )
    plt.savefig(os.path.join(problem.settings["proj_name"], "record_array", "array.png"))

    # output: optimization results of wave energy converters
    file = open(os.path.join(problem.settings["proj_name"], "record_array", "record.txt"), "w")
    file.write("# Used time: {} s\n".format(interval_t))
    file.write("# Corrections: {}, {}\n".format(problem.corrections[0], problem.corrections[1]))
    file.write("# Optimized wave energy converters: (m)\n")
    for i in range(len(var_trace) // 2):
        file.write("\t{}, {}\n".format(var_trace[i], var_trace[i + len(var_trace) // 2]))
    file.close()
