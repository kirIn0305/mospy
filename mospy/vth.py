# -*- coding: utf-8 -*-

import logging
logger = logging.getLogger(__name__)
from abc import ABCMeta, abstractmethod
from mospy import gm
from scipy import stats


class Vth(object, metaclass=ABCMeta):
    """Abstract Vth base class."""
    def __init__(self, **kwargs):
        allowed_kwargs = {"Id", "Vg", "np"}
        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError('Unexpected keyword argument '
                                'passed to optimizer: ' + str(k))
        self.__dict__.update(kwargs)

    @abstractmethod
    def calc(self):
        raise NotImplementedError()

class VthC(Vth):
    """ Constant Current Vth calculater """
    def __init__(self, ith, **kwargs):
        super(VthC, self).__init__(**kwargs)
        self.ith = ith

    def calc(self):
        raise NotImplementedError()

class VthEx(Vth):
    """ Linear extrapolation from saturation region """
    def __init__(self,**kwargs):
        super(VthEx, self).__init__(**kwargs)

    def calc(self):
        raise NotImplementedError()


class VthOn(Vth):
    """ Von calculater """
    def __init__(self, **kwargs):
        super(VthOn, self).__init__(**kwargs)

    def calc(self):
        gm_argmaxs = gm.gm_argmaxs(self.Vg, self.Id)
        Id_tmp = self.Id[gm_argmaxs[0]:gm_argmaxs[1]]
        Vg_tmp = self.Vg[gm_argmaxs[0]:gm_argmaxs[1]]
        slope, intercept, r_value, p_value, std_err = stats.linregress(Vg_tmp, Id_tmp)
        self.von = -intercept/slope
        return self.von


def vthon(Vg, Id):
    vth = VthOn(Vg=Vg, Id=Id, np="n")
    return vth.calc()
