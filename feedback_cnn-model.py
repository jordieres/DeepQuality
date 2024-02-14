#
# DeepQuality Component. When feedback from human operators is provided
# for one CoilID inside the feedback table, 
# a) Update of Label column in V_Coils for such CoilID
# b) Update model performance for the last period (~ 1week) for models
#    used to score such CoilID. Number of participations and % of success
# c) Copy GR_QPValues and V_Coils for such CoilID to T_* tables making it
#    possible to be used for training of next models
#
# (C) UPM / JOM 2024-02-01
# v 1.1

import sys, datetime
sys.path.append('/home/jovyan/scripts/CoilDetails')
import argparse
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
# Prepare the pickle file to be used for model training.
def main():
    # Parameters
    #
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--verbose", nargs='?', action=VAction,\
            dest='verbose',help="Option for detailed information")
    ap.add_argument("-c", "--config", type=str, default="./keys.yaml", \
            help="Config file (included absolute path)")
    ap.add_argument("-l", "--cid", type=str, required=True, \
            help="Coil code of the Coil having operator feedback")
    #
    args = vars(ap.parse_args())
    verbose = 0
    if args['verbose']:
        verbose = args['verbose']
    if args['config']:
        cnfg= args['config']
    if args['cid']:
        cid = args['cid']
    # // Creating the database object for operations.
    db      = DBase(cnfg,verbose)
    # Extracting the Label assigned by the operator
    fdbck   = db.findReg('"feedback"',' "COILID"=\'' + cid +'\'')
    if fdbck['dat'].shape[0] != 1:
        print("Error FeedBack_cnn-model: No one record found {} [ query: {}]".format(\
               fdbck['dat'].shape[0], fdbck['sql']))
        sys.exit(2)
    label   = fdbck['dat']["Label"].tolist()[0]
    # Now copy records from V_Coils and GR_QPValues to T_* for cid coil.
    db.move_regs_train(cid,label)
    # Now find models involved in the assessment of such CoilID during last week
    cdts    = datetime.now().strftime("%Y-%m-%d")
    dtwkprv = datetime.now() + timedelta(weeks = -1)
    dtwkprvs= dtwkprv.strftime("%Y-%m-%d")
    wsql    = ' "COILID"=\'' + cid + '\' and "ts" >= \'' + dtwkprvs + '\' '
    lrgmodls= db.findReg('"assessments"',wsql)
    lmdls   = lrgmodls['dat']['MODELNAME'].tolist()
    # For each related model review of the scoring from the last month of operation
    for i in lmdls:
        # look for assessed coils list to find assessed coils 
        # in the period for model "i"
        dtmthprv   = datetime.now() - relativedelta.relativedelta(months=1)
        dtmthprvs  = dtmthprv.strftime("%Y-%m-%d")
        wsqli      = ' "MODELNAME" = \'' + i + '\' and ts >= \'' + dtmthprvs + '\' '
        rgs_cls_mdl= db.findReg('"assessments"',wsqli)
        lclsa      = rgs_cls_mdl['dat'][['COILID','LABEL']]
        nrgs       = lclsa.shape[0] # nrgs > 0 because at least cid was assessed
        cond       = ','.join(["'"+str(x)+"'" for x in lclsa['COILID'].tolist()])
        wsqlt      = ' where "SID" in (' + cond + ') '
        dfres      = db.loadVarsBd('"T_V_Coils"','',['"SID"','"Label"'],cond=wsqlt)
        dfoper     = dfres['dat']
        dfoper.columns  = dfoper.columns.str.replace('SID','COILID')
        dfoper.columns  = dfoper.columns.str.replace('Label','LABELOP')
        dfoper.columns  = dfoper.columns.str.replace('"','')
        dfoper['COILID']=dfoper['COILID'].astype(int)
        totdfper   = pd.merge(lclsa,dfoper)
        nrgsop     = totdfper.shape[0]
        if nrgsop > 0 :
            succ   = totdfper[totdfper['LABEL']==totdfper['LABELOP']].shape[0]
            db.add_mperformance(i,cdts,dtmthprvs,nrgsop,succ*100/nrgsop,nrgs)

#
if __name__ == "__main__":
    main()