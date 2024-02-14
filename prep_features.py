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
from CoilDetails import DBase, Coils, CoilDefs
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
    ap.add_argument("-l", "--cid", type=str, required=False, \
            help="Coil code when extracting data releted to a single unit")
    ap.add_argument("-c", "--config", type=str, default="./keys.yaml", \
            help="Config file (included absolute path)")
    #
    args = vars(ap.parse_args())
    verbose = 0
    if args['verbose']:
        verbose = args['verbose']
    path = "./"
    if args['path']:
        path = args['path']
    cnfg = None
    if args['config']:
        cnfg= args['config']
    cid  = None
    if args['cid']:
        cid = args['cid']
    #
    cdfs = CoilDefs(cnfg,path,cid,verbose)
    cdfs.set_context()
    cdfs.extract_vars()
    cdfs.save_coildefts()

    dosD = cdfs.extract_coil_dfts('data')
    varcs= cdfs.extract_coil_dfts('varcs')
    #
    normf= path+'/norm_data.pkl'
    if cid is not None:
        normf= path+'/norm_data_'+cid+'.pkl'
    #
    coils= Coils(varcs,dosD,9)
    coils.prep_coils()
    coils.norm_prep_coils(cid,normf)
    coils.save_coils_pkl(normf)
#
if __name__ == "__main__":
    main()
