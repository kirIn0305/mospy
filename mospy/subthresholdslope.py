# -*- coding: utf-8 -*-

import numpy as np
from abc import ABCMeta, abstractmethod
from mospy import gm
from scipy import stats
import logging
logger = logging.getLogger(__name__)


class SubthresholdSlope(object, metaclass=ABCMeta):
    """ Abstarct S.S. base class. """
    def __init__(self, **kwargs):
        allowed_kwargs = {"Id", "Vg", "np"}
        for k in kwargs:
            if k not in allowed_kwargs:
                raise TypeError('Unexpected keyword argument '
                                'passed to optimizer: ' + str(k))
        self.__dict__.update(kwargs)

    @abstractmethod
    def define_iv(self):
        """define Id-Vg data for calculation"""
        raise NotImplementedError()

    def calc(self):
        Id_tmp, Vg_tmp = self.define_iv()
        Id_tmp = [np.log10(x) for x in Id_tmp]
        slope, intercept, r_value, p_value, std_err = stats.linregress(Vg_tmp, Id_tmp)
        return 1/slope


class SSidDec(SubthresholdSlope):
    """S.S. from Gm argmax point"""
    def __init__(self, **kwargs):
        super(SSidDec, self).__init__(**kwargs)

    def define_iv(self):
        gm_argmaxs = gm.gm_decid(self.Vg, self.Id)
        Id_tmp = self.Id[gm_argmaxs[0]:gm_argmaxs[1]]
        Vg_tmp = self.Vg[gm_argmaxs[0]:gm_argmaxs[1]]
        return [Id_tmp, Vg_tmp]
