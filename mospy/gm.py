# -*- coding: utf-8 -*-

import numpy as np


class Gm(object):
    """Gm calculator"""
    def __init__(self, **kwargs):
        """init"""
        allowed_kwargs = {"Id", "Vg"}
        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError('Unexpected keyword argument '
                                'passed to optimizer: ' + str(k))
        self.__dict__.update(kwargs)
        self.gm = None

    def calc(self):
        self.gm = np.diff(self.Id)
        self.Vg_diff = self.Vg[1:]
        return [self.gm, self.Vg_diff]

    def gm_max(self):
        if self.gm is None:
            self.calc()
        self.gm_max = np.max(self.gm)
        return self.gm_max

    def gm_min(self):
        if self.gm is None:
            self.calc()
        self.gm_min = np.min(self.gm)
        return self.gm_min

    def gm_argmax(self):
        if self.gm is None:
            self.calc()
        gm_argmax = self.Vg_diff[np.argmax(self.gm)]
        return gm_argmax

    def gm_decid(self, number):
        if self.gm is None:
            self.calc()
        gm_max = np.max(self.gm)
        gm_argmax = np.argmax(self.gm)
        gm_min = np.min(self.gm)
        th_arg = (gm_max+gm_min)/2
        step_num = number // 2
        range_vg = [gm_argmax-step_num, gm_argmax+step_num]
        flag_range = [True, True]
        valide_range = False
        while valide_range is False:
            for i in range_vg:
                if self.gm[i] < th_arg:
                    flag_range[i] = False
            if all(flag_range):
                if self.Id[range_vg[1]] / self.Id[range_vg[0]] > 10:
                    valide_range = True
            elif flag_range[0]:
                range_vg = [x+1 for x in range_vg]
            elif flag_range[1]:
                range_vg = [x-1 for x in range_vg]
            else:
                range_vg = []
                valide_range = True
        return range_vg

    def gm_argmaxs(self, number):
        if self.gm is None:
            self.calc()
        gm_max = np.max(self.gm)
        gm_argmax = np.argmax(self.gm)
        gm_min = np.min(self.gm)
        th_arg = (gm_max+gm_min)/2
        step_num = number // 2
        range_vg = [gm_argmax-step_num, gm_argmax+step_num]
        flag_range = [True, True]
        valide_range = False
        while valide_range is False:
            for i in range_vg:
                if self.gm[i] < th_arg:
                    flag_range[i] = False
            if all(flag_range):
                valide_range = True
            elif flag_range[0]:
                range_vg = [x+1 for x in range_vg]
            elif flag_range[1]:
                range_vg = [x-1 for x in range_vg]
            else:
                range_vg = []
                valide_range = True
        return range_vg


def gm_max(Vg, Id):
    return Gm(Id=Id, Vg=Vg).gm_max()


def gm_argmax(Vg, Id):
    return Gm(Id=Id, Vg=Vg).gm_argmax()


def gm_argmaxs(Vg, Id, num=5):
    return Gm(Id=Id, Vg=Vg).gm_argmaxs(num)


def gm_decid(Vg, Id, num=5):
    return Gm(Id=Id, Vg=Vg).gm_decid(num)
