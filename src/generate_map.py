import json
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
import os
import time
#from numba import jit, njit
#import cupy as cp

from scipy.optimize import root
from skimage.measure import block_reduce

class CFAR(object):
    def __init__(self, train, guard, false_rate, rank=None):
        self.train = train #number of training cells
        assert self.train % 2 == 0
        self.guard = guard #number of guard cells
        assert self.guard % 2 == 0
        self.false_rate = false_rate #false alarm rate
        if rank is None: #matrix rank
            self.rank = self.train / 2
        else:
            self.rank = rank
            assert 0 <= self.rank < self.train

        self.threshold_factor_SOCA = self.calc_WGN_threshold_factor_SOCA()
        self.threshold_factor_GOCA = self.calc_WGN_threshold_factor_GOCA()

        # self.params = {
        #     "SOCA": (self.train // 2, self.guard // 2, self.threshold_factor_SOCA),
        #     "GOCA": (self.train // 2, self.guard // 2, self.threshold_factor_GOCA),
        # }

    def calc_WGN_threshold_factor_CA(self):
        return self.train * (self.false_rate ** (-1.0 / self.train) - 1)

    def calc_WGN_threshold_factor_SOCA(self):
        x0 = self.calc_WGN_threshold_factor_CA()
        for ratio in np.logspace(-2, 2, 10):
            ret = root(self.calc_WGN_pfa_SOCA, x0 * ratio)
            if ret.success:
                # print(f"Solution: {ret.x[0]}, at guess {x0 * ratio}")
                return ret.x[0]
        raise ValueError("Threshold factor of SOCA not found")

    def calc_WGN_threshold_factor_GOCA(self):
        x0 = self.calc_WGN_threshold_factor_CA()
        for ratio in np.logspace(-2, 2, 10):
            ret = root(self.calc_WGN_pfa_GOCA, x0 * ratio)
            if ret.success:
                return ret.x[0]
        raise ValueError("Threshold factor of GOCA not found")

    def calc_WGN_pfa_GOSOCA_core(self, x):
        x = float(x)
        temp = 0.0
        for k in range(int(self.train / 2)):
            l1 = math.lgamma(self.train / 2 + k)
            l2 = math.lgamma(k + 1)
            l3 = math.lgamma(self.train / 2)
            temp += math.exp(l1 - l2 - l3) * (2 + x / (self.train / 2)) ** (-k)
        return temp * (2 + x / (self.train / 2)) ** (-self.train / 2)

    def calc_WGN_pfa_SOCA(self, x):
        return self.calc_WGN_pfa_GOSOCA_core(x) - self.false_rate / 2

    def calc_WGN_pfa_GOCA(self, x):
        x = float(x)
        temp = (1.0 + x / (self.train / 2)) ** (-self.train / 2)
        return temp - self.calc_WGN_pfa_GOSOCA_core(x) - self.false_rate / 2

if __name__ == "__main__":
    ntc = np.arange(2,41,2)
    ngc = np.arange(2,21,2)
    pfa = np.arange(0.005,1,0.005)
    for train in ntc:
        for guard in ngc:
            for false_alarm_rate in pfa:
                cfar = CFAR(train, guard, false_alarm_rate)
                line = f"{train},{guard},{false_alarm_rate:.3f},{cfar.threshold_factor_SOCA},{cfar.threshold_factor_GOCA}\n"
                # print(f"({train},{guard},{false_alarm_rate}): ({cfar.threshold_factor_SOCA},{cfar.threshold_factor_GOCA})")
                with open("parameter_map1.txt", "a") as f:
                    f.write(line)
                del cfar
