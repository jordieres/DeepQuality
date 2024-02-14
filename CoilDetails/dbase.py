#
# DeepQuality Component. Class handling the PostgresQL database 
# (C) UPM / JOM 2024-01-30
# v 1.1
#
#
# DBase Class
#
import yaml, psycopg2, datetime, pickle
import pandas as pd

class DBase:
    
    def __init__(self,cnfg,verbose=0):  #   cnfg = "./keys.txt"
        if len(cnfg) == 0:
            if verbose > 0:
                print("Error creating the DBase object. Config: {} ".format(cnfg))
            return(None)
        with open(cnfg,"r") as file_object:
            data    = yaml.load(file_object,Loader=yaml.SafeLoader)

        self.host   = data['database']['host']
        self.port   = data['database']['port']
        self.usr    = data['database']['user']
        self.pswd   = data['database']['password']
        self.dbname = data['database']['dbname']
        self.type   = data['database']['dbtype']
        self.verbose= verbose
        
        if self.type == 'postgresql':
            strg = "dbname="+self.dbname+" user="+self.usr+" password="+ \
                    self.pswd + " host="+self.host+ " port=" + str(self.port)
        else:
            strg    = ''
        try:
            self.conn = psycopg2.connect(strg)
        except (Exception, psycopg2.DatabaseError) as error:
            self.error = error
            if self.verbose > 0:
                print("Connection error: {} [string: {}]".format(self.error,strg))
        return None
    
    def loadVarsBd(self,tbl,frst,cols,cond='',ords=''):
        cur  = self.conn.cursor()
        lvr  = ','.join(cols)
        lscls= cols
        if len(frst) > 0:
            nwlst = [i.strip() for i in frst.split(',') if len(i.strip()) > 0]
            lscls = nwlst + cols
            lvr   = ','.join(lscls)
        sql  = "select "+ lvr + " from " + tbl + " " + cond + \
                " " + ords
        try:
            self.error = ''
            cur.execute(sql)
            tpl = cur.fetchall()
            # pdb.set_trace()
            if cols[0] == '*':
                lscls = [desc[0] for desc in cur.description]
            res = pd.DataFrame(tpl,columns=lscls)
        except (Exception, psycopg2.DatabaseError) as error:
            self.error = error
            if self.verbose > 0:
                print("Error LVBd: {} [ query: {}]".format(self.error, sql))
            res = pd.DataFrame()
        cur.close()
        return({'dat':res,'sql':sql, 'error':self.error})

    def findReg(self,tbl,cond): #  Wrapper for PostgresQL query 
        cur  = self.conn.cursor()
        sql  = "select * from " + tbl + " where " + cond 
        try:
            self.error = ''
            cur.execute(sql)
            tpl = cur.fetchall()
            lscls = [desc[0] for desc in cur.description]
            res = pd.DataFrame(tpl,columns=lscls)
        except (Exception, psycopg2.DatabaseError) as error:
            if self.verbose > 0:
                print("Error FindReg: {} [ query: {}]".format(self.error, sql))
            res = pd.DataFrame()
        cur.close()
        return({'dat':res,'sql':sql, 'error': self.error})

    def add_model(self,nam,ts,type):
        cur  = self.conn.cursor()
        sql  = "insert into \"models\" (name,ts,type) VALUES('"+nam+"','"+ts+ \
                       "','"+type+"');"
        try:
            self.error = ''
            cur.execute(sql)
            cur.execute("COMMIT;")
        except (Exception, psycopg2.DatabaseError) as error:
            if self.verbose > 0:
                print("Error FindReg: {} [ query: {}]".format(self.error, sql))
        cur.close()
        
    def close_model(self,nam):  # Closing validity of model
        cur  = self.conn.cursor()
        ts   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        sql  = "update \"models\" set \"valid_until\"='"+ts+ \
                        "' WHERE \"name\"='"+nam+"';"
        try:
            self.error = ''
            cur.execute(sql)
            cur.execute("COMMIT;")
        except (Exception, psycopg2.DatabaseError) as error:
            if self.verbose > 0:
                print("Error FindReg: {} [ query: {}]".format(self.error, sql))
        cur.close()

    def add_assessment(self,nam,cid,res,cnfdcy):
        cur  = self.conn.cursor()
        ts   = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
        sql  = "insert into \"assessments\" (\"COILID\",\"MODELNAME\","+\
                "\"LABEL\",\"Confidence\",\"ts\") VALUES('"+cid+"','"+\
                        nam+"',"+str(res)+","+ str(cnfdcy)+",'"+ts+"');"
        try:
            self.error = ''
            cur.execute(sql)
            cur.execute("COMMIT;")
        except (Exception, psycopg2.DatabaseError) as error:
            if self.verbose > 0:
                print("Error FindReg: {} [ query: {}]".format(self.error, sql))
        cur.close()

    def move_regs_train(self,cid,label):
        cur  = self.conn.cursor()
        sql  = "update \"V_Coils\" set \"Label\" = '"+str(label) + \
               "' where \"SID\"='"+cid+"'"
        try:
            self.error = ''
            cur.execute(sql)
        except (Exception, psycopg2.DatabaseError) as error:
            if self.verbose > 0:
                print("Error FindReg: {} [ query: {}]".format(self.error, sql))
        #
        sql0 = 'insert into "T_GR_QPValues" ("COILID","TILEID","MID","PLANT",' +\
                '"SIDE","MEAN","MAX","MIN","COUNT") select "COILID","TILEID",' +\
                '"MID","PLANT","SIDE","MEAN","MAX","MIN","COUNT" from '+\
                '"GR_QPValues" where "COILID" =\''+cid+'\';'
        try:
            self.error = ''
            cur.execute(sql0)
        except (Exception, psycopg2.DatabaseError) as error:
            if self.verbose > 0:
                print("Error FindReg: {} [ query: {}]".format(self.error, sql0))
        #
        sql1 =  'insert into "T_V_Coils" ("SID","PLANT","NAME","STARTTIME",' +\
                '"MATERIAL","LENGTH","WIDTH","THICK","WEIGHT","BASEPID",' +\
                '"CLASSLABEL","Label","ZnMin","ZnMax") select "SID","PLANT",' +\
                '"NAME","STARTTIME","MATERIAL","LENGTH","WIDTH","THICK",' +\
                '"WEIGHT","BASEPID","CLASSLABEL","Label","ZnMin","ZnMax" ' +\
                'from V_Coils where "SID"=\''+cid+'\';'
        try:
            self.error = ''
            cur.execute(sql1)
        except (Exception, psycopg2.DatabaseError) as error:
            if self.verbose > 0:
                print("Error FindReg: {} [ query: {}]".format(self.error, sql1))
        #
        sql2 =  'DELETE FROM "GR_QPValues" where "COILID"=\''+cid+'\';'
        try:
            self.error = ''
            cur.execute(sql2)
        except (Exception, psycopg2.DatabaseError) as error:
            if self.verbose > 0:
                print("Error FindReg: {} [ query: {}]".format(self.error, sql2))
        #
        sql3 =  'DELETE FROM "V_Coils" where "SID"=\''+cid+'\';'
        try:
            self.error = ''
            cur.execute(sql3)
        except (Exception, psycopg2.DatabaseError) as error:
            if self.verbose > 0:
                print("Error FindReg: {} [ query: {}]".format(self.error, sql3))
        cur.execute("COMMIT;")
        cur.close()

    def add_mperformance(self,mdl,uptoday,torigin,nrgsop,succ,nrgs):
        cur  = self.conn.cursor()
        # check if the record exists
        sql  = 'select count(*) from "mperformance" where "mname" =\''+mdl+ \
                '\' and "from_ass"=\''+torigin+'\' and "to_ass"=\'' +uptoday+\
                '\';'
        try:
            self.error = ''
            cur.execute(sql)
            tpl = cur.fetchall()
            res = pd.DataFrame(tpl)
        except (Exception, psycopg2.DatabaseError) as error:
            if self.verbose > 0:
                print("Error FindReg: {} [ query: {}]".format(self.error, sql))
        if res.loc[0,0] > 0: # Record existing => update
            sql1 = 'update "mperformance" set "ok_percentage"='+str(succ)+', '+\
                    '"num_oper"='+str(nrgsop)+', "numc_assessed"='+str(nrgs)+' '+\
                    'where "mname" =\''+mdl+ \
                '\' and "from_ass"=\''+torigin+'\' and "to_ass"=\'' +uptoday+\
                '\';'
        else:
            sql1 = 'insert into "mperformance" ("ok_percentage","num_oper", '+\
                    '"numc_assessed","mname","from_ass","to_ass") values (' +\
                    str(succ)+','+str(nrgsop)+','+str(nrgs)+',\''+mdl+'\',\''+\
                    torigin+'\',\''+uptoday+'\');'
        try:
            self.error = ''
            cur.execute(sql1)
            cur.execute("COMMIT;")
        except (Exception, psycopg2.DatabaseError) as error:
            if self.verbose > 0:
                print("Error FindReg: {} [ query: {}]".format(\
                        self.error, sql1))
        cur.close()

    def coil_decison_making(self,cid):
        cur     = self.conn.cursor()
        wsql    = ' "COILID"=\'' + cid + '\' '
        dfass   = self.findReg('"assessments"',wsql)
        dfres   = dfass['dat'][['LABEL','Confidence']].groupby('LABEL'\
                        ).sum()
        catl    = 1 
        if (dfres.loc[1,'Confidence'] < dfres.loc[2,'Confidence']):
            catl= 2
        sql     = 'update "V_Coils" set "Label"='+str(catl)+\
                        'where "SID" =\''+cid+ '\';'
        try:
            self.error = ''
            cur.execute(sql)
            cur.execute("COMMIT;")
        except (Exception, psycopg2.DatabaseError) as error:
            if self.verbose > 0:
                print("Error FindReg: {} [ query: {}]".format(\
                       self.error, sql))
        cur.close()

#
#
# Class CoilDefs handling the postgres tables (or records inside the tables)
# and place relevant data into pickle file objects.
# Then Coil class will normalize them.
#
class CoilDefs(DBase):
    def __init__(self,cnfg,path,cid,verbose):
        super().__init__(cnfg,verbose)
        self.path = path
        self.cid  = cid
        self.dosD = None
        self.varcs= None
        self.gaugs= None

    def set_context(self):
        if self.cid is None:
            self.fdat     = self.path + '/data.pkl'
            self.fvards   = self.path + 'vards.pkl'
            self.tb_coils = '"T_V_Coils"'
            self.tb_tiles = '"T_GR_QPValues"'
        else:
            self.fdat     = self.path + '/data_'+self.cid+'.pkl'
            self.fvards   = self.path + '/vards_'+self.cid+'.pkl'
            self.tb_coils = '"V_Coils"'
            self.tb_tiles = '"GR_QPValues"'        
        
    def extract_vars(self):  
        # prepare dosD and vards in pkl from T_* postgresql tables.
        # ot w/o T_* when cid is provided
        self.set_context()
        # Loading the variable list 
        colsv = ['"ID"','"PARAMETER"']
        r     = self.loadVarsBd('"Gauges"','',colsv)
        #
        self.gaugs= r['dat']
        if self.gaugs.shape[0] > 0:
            self.gaugs.set_index('"ID"',drop=True,inplace=True)
            self.gaugs[['"PARAMETER"']].sort_values(by='"ID"')
        #
        # Loading the variable list
        #
        colsv = ['*']
        cond  = ''
        if self.cid is not None:
            cond  = ' where "SID"='+"'"+self.cid+"' "
        r     = self.loadVarsBd(self.tb_coils,'',colsv,cond)
        #
        self.varcs = r['dat']
        if self.varcs.shape[0] > 0:
            self.varcs.set_index('id',drop=True,inplace=True)
        #
        cond  = ''
        if self.cid is not None:
            cond  = ' where "COILID"='+"'"+self.cid+"' "
        qd   = self.loadVarsBd(self.tb_tiles,'',colsv,cond)
        self.dosD = qd['dat']
        self.dosD = self.dosD.sort_values(['COILID', 'MID', 'TILEID']).reset_index(drop=True)
        self.dosD.reset_index(drop=True,inplace=True)
        
    def save_coildefts(self):
        GlblPrms = {'path':self.path,'cid':self.cid,'gauges':self.gaugs, \
                    'fdat':self.fdat,'varcs':self.varcs}
        with open(self.fdat, 'wb') as handle:
            pickle.dump(self.dosD, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(GlblPrms, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(self.fvards,'wb') as handle:
            pickle.dump(self.varcs, handle, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(GlblPrms, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    def load_coildefts(self,fdat,fvarcs):
        with open(fdat, 'rb') as handle:
            self.dosD = pickle.load(handle)
            GlblPrms  = pickle.load(handle)    
        self.path = GlblPrms['path']
        self.cid  = GlblPrms['cid']
        self.gaugs= GlblPrms['gauges']
        self.fdat = GlblPrms['fdat']            
        self.varcs= GlblPrms['varcs']
        with open(fvarcs, 'rb') as handle:
            self.varcs= pickle.load(handle) 

    def load_ds(self,op):
        res = {}
        if op == 'data' or op == 'all':
            with open(self.fdat, 'rb') as handle:
                self.dosD = pickle.load(handle)
                self.dosD['MID'] = self.dosD['MID'].astype(int)
                res['dat'] = self.dosD
        if op == 'varcs' or op == 'all':
            with open(self.fvards, 'rb') as handle:
                self.coils= pickle.load(handle)
                res['vard'] = self.coils
        return(res)
    
    def extract_coil_dfts(self,op):
        if op == 'data':
            return(self.dosD)
        if op == 'varcs':
            return(self.varcs)

# // End Class DBase