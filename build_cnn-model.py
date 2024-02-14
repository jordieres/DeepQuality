#
# DeepQuality Component. Extract & normalice features either for 
# The training database or for the on-line data-base when invoked 
# with a coilID parameter.
#
# (C) UPM / JOM 2024-01-30
# v 1.1

import sys, os, datetime, pickle, time
sys.path.append('/home/jovyan/scripts/CoilDetails')
import argparse, string, psycopg2
import scipy, yaml, pdb
import pandas as pd
import numpy as np
from timeit import default_timer as timer
from datetime import datetime, timedelta, date
from sqlalchemy import create_engine
from argparse import ArgumentParser
#
from CoilDetails import  Coils, CNN_Model

#
#
# it requires the file keys.txt
#
#
# from https://stackoverflow.com/questions/6076690/verbose-level-with-argparse
#-and-multiple-v-options
class VAction(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, const=None,
                 default=None, type=None, choices=None, required=False,
                 help=None, metavar=None):
        super(VAction, self).__init__(option_strings, dest, nargs, const,
                                      default, type, choices, required,
                                      help, metavar)
        self.values = 0
    def __call__(self, parser, args, values, option_string=None):
        # print('values: {v!r}'.format(v=values))
        if values is None:
            self.values += 1
        else:
            try:
                self.values = int(values)
            except ValueError:
                self.values = values.count('v')+1
        setattr(args, self.dest, self.values)

#
# Prepare the pickle file to be used for model training.
def main():
    # Parameters
    # Path parameter establish
    #     path/data.pkl  => for information related to dosD dataframe of training DB
    #     path/vards.pkl => for info related to vard object of training DB
    #     path/dat_cid.pkl=>for info related to a single coil (Not from training DB)
    #
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", nargs='?', action=VAction,\
            dest='verbose',help="Option for detailed information")
    ap.add_argument("-p", "--path", type=str, required=False, \
            help="Path for placing the Pickle files in use")
    ap.add_argument("-m", "--modelpath", type=str, required=False, \
            help="Path for placing the hdf5 model after training")
    ap.add_argument("-n", "--nsmpl", type=int, default=25, \
            help="Number of samples for quality measurement")
    ap.add_argument("-c", "--config", type=str, default="./keys.yaml", \
            help="Config file (included absolute path)")
    #
    args = vars(ap.parse_args())
    verbose = 0
    if args['verbose']:
        verbose = args['verbose']
    path  = "./"
    if args['path']:
        path = args['path']
    mpath = None
    if args['modelpath']:
        mpath = args['modelpath']
    cnfg = None
    if args['config']:
        cnfg= args['config']
    nsmpl = args['nsmpl']
    # // Creating the set of coils from normalized data.
    coils = Coils(None,None,9)
    normf = path+'/norm_data.pkl'
    coils.load_coils_pkl(normf)
    det   = coils.extract_detcoils()
    maps  = coils.extract_mapcoils()
    nsens = coils.extract_nsens()
    cat   = [1,2]  # Category 1 => OK, Category 2 => NoK
    # // Creating model object
    mdl   = CNN_Model(det,maps,nsens,cat,nsmpl)
    mdl.save_msets(path)
    mdl.buildModel(verbose)
    mdl.save_cnnmodel(mpath,cnfg,verbose)
    mdl.save_hist_cnnmodel(path)
#
if __name__ == "__main__":
    main()