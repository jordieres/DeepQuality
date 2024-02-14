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
from CoilDetails import  DBase, Coils, CNN_Model

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
    ap.add_argument("-p", "--path", type=str, required=True, \
            help="Path to the Pickle file holding the coil info for assessment")
    ap.add_argument("-c", "--config", type=str, default="./keys.yaml", \
            help="Config file (included absolute path)")
    #
    args = vars(ap.parse_args())
    verbose = 0
    if args['verbose']:
        verbose = args['verbose']
    if args['path']:
        nomf = args['path']
    if args['config']:
        cnfg= args['config']
    # // Creating the set of coils from normalized data.
    coil  = Coils(None,None,9)
    coil.load_coils_pkl(nomf)
    det   = coil.extract_detcoils()
    maps  = coil.extract_mapcoils()
    nsens = coil.extract_nsens()
    cat   = [1,2]  # Category 1 => OK, Category 2 => NoK
    # // Creating model object
    mdl   = CNN_Model(det,maps,nsens,cat,0)
    lnmdls= mdl.list_cnnmodels(cnfg,verbose) # List of available models
    for i in lnmdls:
        mdl.eval_cnnmodel(i,cnfg,verbose)
    # Once models assessed => Update V_Coils
    cid   = coil.extract_lcoils()[0] # Obtaining the CoilID
    db    = DBase(cnfg,verbose)
    db.coil_decision_making(cid)
    #

if __name__ == "__main__":
    main()