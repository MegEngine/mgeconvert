import numpy as np

from .cnlib import cambriconLib as cnlib


class Model:
    __name = None
    __cnml_model = None

    def __init__(self, name):
        self.__name = name
        self.__cnml_model = cnlib.cnModel(self.__name)

    def dump(self, fname):
        cnlib.cnmlSaveModel(self.__cnml_model, fname)

    @property
    def name(self):
        return self.__name

    def add_fusionop(self, fusion):
        if fusion.compiled is False:
            fusion.compile()

        cnlib.cnmlAddFusionOpToModel(self.__cnml_model, fusion.op, fusion.name)
