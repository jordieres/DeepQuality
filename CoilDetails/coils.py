#
# DeepQuality Component. Class handling the Coils management and normalization
# It will prepare the data 
# for training and assessing models
#
# (C) UPM / JOM 2024-01-30
# v 1.1
#
import scipy, pickle
import pandas as pd
import numpy as np

class Coils():
    
    # function extracting the relevant rows for the coildID
    def __init__(self,coils,dosD,nsens):
        self.coils      = coils
        self.znvals     = dosD
        self.lcoils     = []
        self.clmaps     = {}
        self.clmpswoclb = {}
        self.nsens      = nsens  # Number of sensors in a row measuring
                                 # It does impact on the number of tiles.
        self.GlblFrm    = {}     # Global Parameters
        if self.znvals is not None:
            self.lcoils = self.znvals['COILID'].unique().tolist()

    def extract_dcoil(self,id):
        res  = {}
        dc   = self.coils.loc[self.coils['SID']==id,:]
        dt   = self.znvals.loc[self.znvals['COILID']==id,:]
        if len(dc) == 0 or len(dt) == 0:
            return(None)
        lsens= dt['MID'].unique().tolist()
        num  = []
        for isns in lsens:
            res[isns] = {}
            dts = dt.loc[dt['MID']==isns,:]
            if len(dts) == 0:
                continue
            pos = dts['TILEID'].values.reshape((int(len(dts)/self.nsens),self.nsens)) #tileid
            zn  = dts['MEAN'].values.reshape((int(len(dts)/self.nsens),self.nsens)) # Zn value/tile
            res[isns]['pos'] = pd.DataFrame(pos)
            res[isns]['zn']  = pd.DataFrame(zn)
            num.append(zn.shape[0])
        res['props'] = {'ntiles':max(num),'Len': float(dc['LENGTH'].iloc[0]),
                        'Wid':float(dc['WIDTH'].iloc[0]),'Label':int(dc['Label'].iloc[0]),
                        'ZnMn':int(dc['ZnMin'].iloc[0]),'ZnMx':int(dc['ZnMax'].iloc[0]),
                        'nsns':self.nsens}
        return(res)

    def prep_coils(self):
        for ic in self.lcoils:
            self.clmaps[ic] = self.extract_dcoil(ic)

    #
    # Is there a callibration in this face as per individual sensor?
    def check_chunk(self,clmap,ims,znmn):
        callib = {}
        numchk = {}
        numtls = {}
        for js in range(self.nsens): # As per sensor in the sensors row
            callib[js] = []  # Maybe more than one
            out  = clmap[ims]['zn'][js]-znmn < 0.
            ds   = out[1:][out.diff()[1:]]  # indexing True values as below the threshold
            ds[0]= out[0]
            ds.sort_index(inplace=True)
            lidpch = ds[ds==True].index
            numchk[js] = len(lidpch)
            lfls   = ds[ds==False].index
            numtjs = 0
            for j in lidpch:
                nls = lfls[lfls > j]
                if len(nls) > 0:
                    esg = nls[0]  # End of current segment of tiles with True
                else:
                    esg = len(out) # The last one in other case.
                sqy = clmap[ims]['zn'][js][j:esg]
                if any(sqy < 5.) :  # Below this threshold is callibration.
                    clmap[ims]['zn'][js][j:esg] = znmn
                    numchk[js]= numchk[js]-1
                    callib[js].append(j)
                else:
                    numtjs = numtjs + esg-j
            numtls[js] = numtjs
        clmap['props']['callib_'+str(ims)] = callib
        clmap['props']['nchks_'+str(ims)]  = numchk
        clmap['props']['nfltls_'+str(ims)] = numtls
        return(clmap)
    
    #
    # function extracting the max range outside the limits, excluding callibrations
    def callibration(self,ic):
        clmap= self.clmaps[ic].copy()
        znmn = clmap['props']['ZnMn']
        lsms = list(clmap.keys())
        lsms.remove('props')
        for ims in lsms: # Callibration control per face
            clmap = self.check_chunk(clmap,ims,znmn)
        return(clmap)

    #
    # function extracting the max range outside the limits, excluding callibrations
    # for a particular coil.
    def saliency_range(self,ic):
        znmn = self.clmpswoclb[ic]['props']['ZnMn']
        znmx = self.clmpswoclb[ic]['props']['ZnMx']
        nss  = self.clmpswoclb[ic]['props']['nsns']
        lsms = list(self.clmpswoclb[ic].keys())
        lsms.remove('props')
        mnvt = 1000
        mxvt = -1000
        for ims in lsms: # Callibration control per face
            mnv = (self.clmpswoclb[ic][ims]['zn']-znmn).min().min()
            mxv = (self.clmpswoclb[ic][ims]['zn']-znmx).max().max()
            self.clmpswoclb[ic]['props'][str(ims)+'_mx'] = mxv
            self.clmpswoclb[ic]['props'][str(ims)+'_mn'] = mnv
            if mnvt > mnv:
                mnvt = mnv
            if mxvt < mxv:
                mxvt = mxv
        self.clmpswoclb[ic]['props']['MxVal'] = mxvt
        self.clmpswoclb[ic]['props']['MnVal'] = mnvt    

    #
    # Normalizing saliency maps
    def normalize_saliency(self,ic,Tmn,Tmx):
        clmap= self.clmpswoclb[ic]
        znmn = clmap['props']['ZnMn']
        znmx = clmap['props']['ZnMx']
        nss  = clmap['props']['nsns']
        lsms = list(clmap.keys())
        lsms.remove('props')
        for ims in lsms: # Callibration control per face
            self.clmpswoclb[ic][ims]['nzn'] = {}
            rss = clmap[ims]['zn'].copy()
            for js in range(nss):
                out  = rss[js].copy()
                # pdb.set_trace()
                idx0 = (out >= znmn) & (out <= znmx)
                idx1 = out<znmn
                idx2 = out>znmx
                out[idx0] = 0.
                out[idx1] = (out[idx1]-znmn)/abs(Tmn)
                out[idx2] = (out[idx2]-znmx)/abs(Tmx)
                rss[js]=out
            self.clmpswoclb[ic][ims]['nzn'] = pd.DataFrame(rss)

    # Embedding saliency maps
    def embed_saliency(self,ic,MaxT):
        clmap= self.clmpswoclb[ic]
        nss  = clmap['props']['nsns']
        lsms = list(clmap.keys())
        lsms.remove('props')
        xnew = np.arange(0, MaxT,1)
        for ims in lsms: # Callibration control per face
            self.clmpswoclb[ic][ims]['nzne'] = {}
            res = []
            rss = clmap[ims]['nzn'].copy()
            xold= clmap[ims]['nzn'].index.to_numpy()*(MaxT-1)/(rss.shape[0]-1)
            for js in range(nss):
                yold  = rss[js].to_numpy()
                # pdb.set_trace()
                f     = scipy.interpolate.interp1d(xold, yold)
                ynew  = f(xnew)
                res.append(ynew)
            self.clmpswoclb[ic][ims]['nzne'] = pd.DataFrame(res).T

    def norm_prep_coils(self,cid,normf):
        for ic in self.clmaps.keys():
            self.clmpswoclb[ic] = self.callibration(ic)  # Removing callibration
        TmxVal = TmnVal = numT= 0.
        for ic in self.clmpswoclb.keys():
            self.saliency_range(ic)
            TmxVal  = max(TmxVal,self.clmpswoclb[ic]['props']['MxVal'])
            TmnVal  = min(TmnVal,self.clmpswoclb[ic]['props']['MnVal'])
            numT    = max(numT,self.clmpswoclb[ic]['props']['ntiles'])
        #
        if cid is not None: # Working for a single coil => Recover the GlblFrm info
            glfname = normf.replace('_'+cid,'')
            glcoils = Coils(None,None,9)
            glcoils.load_coils_pkl(glfname)
            TmxVal  = glcoils.GlblFrm['NormMx']
            TmnVal  = glcoils.GlblFrm['NormMn']
            numT    = glcoils.GlblFrm['MaxTiles']
        
        for ic in self.clmpswoclb.keys():
            self.normalize_saliency(ic,TmnVal,TmxVal)
        #
        self.GlblFrm = {'NormMn':TmnVal,'NormMx':TmxVal,'MaxTiles':numT, \
                        'nsens':self.nsens,'lcoils':self.lcoils}
        #
        MaxT    = self.GlblFrm['MaxTiles']
        for ic in self.clmpswoclb.keys():
            self.embed_saliency(ic,MaxT)

    def save_coils_pkl(self,fich):
        if len(fich) == 0:
            if self.verbose > 0:
                print("Error saving Coils object. Fich:{}".format(fich))
            return(None)
        else:
            with open(fich, 'wb') as handle:
                pickle.dump(self.GlblFrm, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.znvals, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.coils, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.clmpswoclb, handle, protocol=pickle.HIGHEST_PROTOCOL)
                pickle.dump(self.clmaps, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_coils_pkl(self,fich):
        if len(fich) == 0:
            if self.verbose > 0:
                print("Error restoring Coils object. Fich:{}".format(fich))
            return(None)        
        else:
            with open(fich, 'rb') as handle:
                self.GlblFrm    = pickle.load(handle)
                self.znvals     = pickle.load(handle)
                self.coils      = pickle.load(handle)
                self.clmpswoclb = pickle.load(handle)
                self.clmaps     = pickle.load(handle)
                self.nsens      = self.GlblFrm['nsens']
                self.lcoils     = self.GlblFrm['lcoils']

    def extract_lcoils(self):
        return(self.lcoils)

    def extract_detcoils(self):
        return(self.coils)

    def extract_mapcoils(self):
        return(self.clmpswoclb)

    def extract_nsens(self):
        return(self.nsens)

    def extract_glblfrm(self):
        return(self.GlblFrm)

#
# // End Class Coils
#