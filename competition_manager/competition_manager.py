"""

my competition manager

"""

import os.path as osp
import pandas as pd

class IOManager:
    """
    io managing
    """

    def __init__(self, notify=None, rootdir=None, inputdir=None, pretraindir=None, outputdir=None):
        self.notify = notify
        self.rootdir = rootdir
        self.inputdir = inputdir
        self.pretraindir = pretraindir
        self.outputdir = outputdir

    def write_pickle(self, obj, name, mode="None", check_suffix=True):
        """
        write pickle
        """
        if mode == "data":
            name = osp.join(self.rootdir, self.inputdir, name)
        elif mode == "model":
            name = osp.join(self.rootdir, self.pretraindir, name)
        elif mode == "result":
            name = osp.join(self.rootdir, self.outputdir, name)
        if check_suffix and name.split('.')[-1] != "pkl":
            name+=".pkl"
        pd.to_pickle(obj, name)
        if self.notify:
            self.notify.notify(text=f"Export : {name}")

    def write_csv(self, obj, name, mode="None", index=False, header=False, check_suffix=True):
        """
        write csv
        """
        if mode == "data":
            name = osp.join(self.rootdir, self.inputdir, name)
        elif mode == "model":
            name = osp.join(self.rootdir, self.pretraindir, name)
        elif mode == "result":
            name = osp.join(self.rootdir, self.outputdir, name)
        if check_suffix and name.split('.')[-1] != "csv":
            name+=".csv"
        pd.to_csv(obj, name, index=index, header=header)
        if self.notify:
            self.notify.notify(text=f"Export : {name}")

    def load_pickle(self, name, mode="None", check_suffix=True):
        """
        load pickle
        """
        if mode == "data":
            name = osp.join(self.rootdir, self.inputdir, name)
        elif mode == "model":
            name = osp.join(self.rootdir, self.pretraindir, name)
        elif mode == "result":
            name = osp.join(self.rootdir, self.outputdir, name)
        if check_suffix and name.split('.')[-1] != "pkl":
            name+=".pkl"
        return pd.read_pickle(name)

    def load_csv(self, name, mode="None", header='infer', check_suffix=True):
        """
        csv loading
        """
        if mode == "data":
            name = osp.join(self.rootdir, self.inputdir, name)
        elif mode == "model":
            name = osp.join(self.rootdir, self.pretraindir, name)
        elif mode == "result":
            name = osp.join(self.rootdir, self.outputdir, name)
        if check_suffix and name.split('.')[-1] != "csv":
            name+=".csv"
        return pd.read_csv(name, header=header)
