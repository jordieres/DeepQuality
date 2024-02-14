#
# DeepQuality Component. When model underperforms according to mperformance table
# Select the model
# a) Find the models with more than 10 coils assessed during the last week
#    and having success ration fallen short below 75%.
# b) For each of them retrain a new model with the new data.
# c) For each of them, close those models in table models.
#
# (C) UPM / JOM 2024-02-05
# v 1.2

import sys, datetime, os
sys.path.append('/home/jovyan/scripts/CoilDetails')
import argparse, subprocess
import pandas as pd
from timeit import default_timer as timer
from datetime import datetime, timedelta
from dateutil import relativedelta
from argparse import ArgumentParser
from CoilDetails import  DBase
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
def main():
    # Parameters
    #   Hypothesis:
    #      a) The script ran from */scripts
    #      b) There are pkls directory at ../pkls/
    #      c) There are models at ../models/
    #
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", nargs='?', action=VAction,\
            dest='verbose',help="Option for detailed information")
    ap.add_argument("-c", "--config", type=str, default="./keys.yaml", \
            help="Config file (included absolute path)")
    #
    args = vars(ap.parse_args())
    verbose = 0
    if args['verbose']:
        verbose = args['verbose']
    if args['config']:
        cnfg= args['config']
    abspath = os.path.dirname(os.path.abspath(cnfg))+'/'
    # // Creating the database object for operations.
    db      = DBase(cnfg,verbose)
    # Finding models failing in last week
    cdts    = datetime.now().strftime("%Y-%m-%d")
    dtwkprv = datetime.now() + timedelta(weeks = -1)
    dtwkprvs= dtwkprv.strftime("%Y-%m-%d")
    wsql    = ' "to_ass">=\'' + dtwkprvs + '\' and "ok_percentage" < 75. ' +\
              ' and num_oper > 10 '
    mdundp  = db.findReg('"mperformance"',wsql)
    lmdlundp= mdundp['dat']['mname'].tolist()
    #
    if len(lmdlundp) > 0:
        # Prepare data for training process (once if there are models underperf.)
        cmd = '/opt/bitnami/miniconda/bin/python3'
        exe = abspath + 'prep_features.py'
        p1  = '-p'
        p2  = abspath + '../pkls/'
        p3  = '-c'
        p4  = abspath + 'keys.yaml'
        lcms= [cmd,exe,p1,p2,p3,p4]
        r1  = subprocess.Popen(lcms)
        ext = r1.wait()
        if (ext > 0 and verbose > 0):
            print("Error in Prep_Features {} [ {} ]".format(ext,lcms))
            sys.exit(2)
    # For every models underperforming ...
    for i in lmdlundp:
        # Now find models involved in the assessment of such CoilID during last week
        exe = abspath + 'build_cnn-model.py'
        p1  = '-p'
        p2  = abspath + '../pkls/'
        p3  = '-m'
        p4  = abspath + '../models/'
        p5  = '-c'
        p6  = abspath + 'keys.yaml'
        lcms= [cmd,exe,p1,p2,p3,p4,p5,p6]
        r1  = subprocess.Popen(lcms)
        ext = r1.wait()
        if (ext > 0 and verbose > 0):
            print("Error in Prep_Features {} [ {} ]".format(ext,lcms))
        else:
            # Once another model has been trained we close the underperforming one
            db.close_model(i)
#
if __name__ == "__main__":
    main()