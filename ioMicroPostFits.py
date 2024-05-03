import numpy as np
import glob,os,sys
import cv2
import matplotlib.pyplot as plt
import tifffile

from tqdm.notebook import tqdm
import pickle
from scipy.spatial.distance import cdist





def get_X(dic_drifts,dic_pts_cells,fls_fov,icell,pix=[0.200,0.1083,0.1083],nelems = 4):
    ncol = len(dic_pts_cells[0])
    Xs,Rs,hs,icols = [],[],[],[]
    R_ = 0
    for dic_drift,dic_pts_cell in zip(dic_drifts,dic_pts_cells):
        for icol in range(ncol):
            R_+=1
            if icell in dic_pts_cell[icol]:
                fits = dic_pts_cell[icol][icell]['fits']
            else:
                fits = []
            if len(fits)>0:
                X_ = fits[:nelems,1:4]
                h_ = fits[:nelems,[0,4]]
                X_ = X_-dic_drift['Ds'][0]#drift correct
                X_ = X_*pix #convert to um
                Xs.extend(X_)
                hs.extend(h_)
                Rs.extend([R_]*len(X_))
                icols.extend([icol]*len(X_))
    return np.array(Xs),np.array(hs),np.array(Rs),np.array(icols)
def get_cells(dic_pts_cells,volume_th =200000):
    cells = list(dic_pts_cells[0][0].keys())
    return np.array([icell for icell in cells if dic_pts_cells[0][0][icell]['volume']>volume_th])
def determine_number_of_chromosomes(Xs,hs,Rs,radius_chr = 1.25,nchr_=5,fr_th = 0.5,plt_val=True):
    from scipy.spatial.distance import pdist,squareform
    nRs = len(np.unique(Rs))
    mat = squareform(pdist(Xs)) #distance matrix
    mat_connection = np.exp(-mat**2/2/(radius_chr/3)**2) #gaussian connectivity matrix

    keep_index = np.arange(len(mat))
    ibests = []
    for iiter in range(nchr_):
        mat_connection_ = mat_connection[keep_index][:,keep_index]
        min_ = np.min(mat_connection_.shape)
        if min_>0:
            mat_ = mat[keep_index][:,keep_index]
            ibest = np.argmax([np.mean(row) for row in mat_connection_])
            ibests.append(keep_index[ibest])
            keep_index = keep_index[np.where(mat_[ibest]>radius_chr)[0]]

    ibests = np.array(ibests)
    #print(np.sum(mat[ifirst]<radius_chr),np.sum(mat[isecond]<radius_chr))
    cls = np.argmin(mat[ibests,:],0)
    nfs = []
    for icl,ibest in enumerate(ibests):
        keep_elems = np.where((mat[ibest]<radius_chr*1.25)&(cls==icl))[0] #which elements are closest to center chr and closeset tot that
        Rs_keep = np.unique(Rs[keep_elems])## regions we are covering
        #print(Rs_keep)
        nfr_ = len(Rs_keep)/nRs
        nfs.append(nfr_)
        if plt_val: print(nfr_)
    nfs = np.array(nfs)
    ibests = ibests[nfs>fr_th]
    if plt_val: 
        if len(ibests)>0:
            cls = np.argmin(mat[ibests,:],0)
            plt.figure()
            #plt.title("Cell "+str(icell))
            iz = 1
            plt.plot(Xs[:,iz],Xs[:,2],'.',color='gray')
            print("Final_ths:")
            for ic in range(len(ibests)):
                ibest = ibests[ic]
                keep = (cls==ic)&(mat[ibests[ic]]<radius_chr*1.25)
                Rs_keep = np.unique(Rs[keep])
                nfr_ = len(Rs_keep)/nRs
                print(nfr_)
                plt.plot(Xs[keep,iz],Xs[keep,2],'o',alpha=1)
                plt.plot(Xs[ibest,iz],Xs[ibest,2],'ko')
            #plt.plot(Xs[isecond,1],Xs[isecond,2],'ko')
            plt.axis('equal')
            plt.show()
    return Xs[ibests]


### EM functions

### Usefull functions

def nan_moving_average(a,n=3):
    a_ = np.array(a)
    if n>0: a_ = np.concatenate([a[-n:],a,a[:n]])
    ret = np.nancumsum(a_,axis=0, dtype=float)
    ret_nan = ~np.isnan(a_)
    ret_nan = np.cumsum(ret_nan,axis=0, dtype=float)
    n_=2*n+1
    ret[n_:] = ret[n_:] - ret[:-n_]
    ret_nan[n_:] = ret_nan[n_:] - ret_nan[:-n_]
    ret_ = ret[n_ - 1:] / ret_nan[n_ - 1:]
    return ret_
def moving_average(a,n=3):
    a_ = np.array(a)
    if n>0: a_ = np.concatenate([a[-n:],a,a[:n]])
    ret = np.cumsum(a_,axis=0, dtype=float)
    n_=2*n+1
    ret[n_:] = ret[n_:] - ret[:-n_]
    ret_ = ret[n_ - 1:] / n_
    return ret_
def cum_val(vals,target):
    """returns the fraction of elements with value < taget. assumes vals is sorted"""
    niter_max = 10
    niter = 0
    m,M = 0,len(vals)-1
    while True:
        mid = int((m+M)/2)
        if vals[mid]<target:
            m = mid
        else:
            M = mid
        niter+=1
        if (M-m)<2:
            break
    return mid/float(len(vals))
def flatten(l):
    return [item for sublist in l for item in sublist]

def get_Ddists_Dhs(zxys_f,hs_f,nint=5):
    h = np.ravel(hs_f)#[np.ravel(cols_f)=='750']
    h = h[(~np.isnan(h))&(~np.isinf(h))&(h>0)]
    h = np.sort(h)
    dists = []
    distsC = []
    for zxys_T in zxys_f:
        difs = zxys_T-nan_moving_average(zxys_T,nint)#np.nanmedian(zxys_T,0)
        difsC = zxys_T-np.nanmedian(zxys_T,axis=0)
        dists.extend(np.linalg.norm(difs,axis=-1))
        distsC.extend(np.linalg.norm(difsC,axis=-1))
    dists = np.array(dists)
    dists = dists[(~np.isnan(dists))&(dists!=0)]
    dists = np.sort(dists)
    
    distsC = np.array(distsC)
    distsC = distsC[(~np.isnan(distsC))&(distsC!=0)]
    distsC = np.sort(distsC)
    return h,dists,distsC
def get_maxh_estimate(pfits_cands_,Rs_u = np.arange(175)+1):
    """
    Assumes pfits_cands_ is of the form Nx5 where 1:3 - z,x,y 4-h and 5-R
    For each R, get the corresponding zxy, h, and R from the one with max h
    """
    zxys_T = []
    hs_T=[]
    hs_bk_T=[]
    if len(pfits_cands_)>0:
        Rs = pfits_cands_[:,-1]
        for R_ in Rs_u:
            pfits = pfits_cands_[Rs==R_]
            if len(pfits)==0:
                zxys_T.append([np.nan]*3)
                hs_T.append(np.nan)
                hs_bk_T.append(np.nan)
                continue
            hs = pfits[:,3]
            hs_bk = pfits[:,4]
            
            
            
            zxys = pfits[:,:3]
            imax = np.argmax(hs)
            hs_T.append(hs[imax])
            hs_bk_T.append(hs_bk[imax])
            zxys_T.append(zxys[imax])
    return zxys_T,hs_T,hs_bk_T

def get_statistical_estimate(pfits_cands_,Dhs,Ddists,DdistsC,zxys_T=None,nint=5,use_local=True,use_center=True,
                             Rs_u = np.arange(175)+1):
    if zxys_T is None:
        zxys_T,hs_T,hs_bk_T = get_maxh_estimate(pfits_cands_,Rs_u=Rs_u)
    zxys_mv = nan_moving_average(zxys_T,nint)
    zxysC = np.nanmean(zxys_T,axis=0)
    zxys_T_ = []
    hs_T=[]
    scores_T = []
    all_scores=[]
    for R_ in Rs_u:#range(len(pfits_cands_)):
        Rs = pfits_cands_[:,-1]
        pfits = pfits_cands_[Rs==R_]
        if len(pfits)==0:
            zxys_T_.append([np.nan]*3)
            hs_T.append(np.nan)
            scores_T.append(np.nan)
            continue
        hs = pfits[:,3]
        zxys_ = pfits[:,:3]
        u_i = R_-1
        dists = np.linalg.norm(zxys_-zxys_mv[u_i],axis=-1)
        distsC = np.linalg.norm(zxys_-zxysC,axis=-1)
        if use_local and use_center:
            scores = [(1-cum_val(DdistsC,dC_))*(1-cum_val(Ddists,d_))*(cum_val(Dhs,h_)) for dC_,d_,h_ in zip(distsC,dists,hs)]
        if not use_local and use_center:
            scores = [(1-cum_val(DdistsC,dC_))*(cum_val(Dhs,h_)) for dC_,d_,h_ in zip(distsC,dists,hs)]
        if use_local and not use_center:
            scores = [(1-cum_val(Ddists,d_))*(cum_val(Dhs,h_)) for dC_,d_,h_ in zip(distsC,dists,hs)]
        if not use_local and not use_center:
            scores = [cum_val(Dhs,h_) for dC_,d_,h_ in zip(distsC,dists,hs)]
        iscore = np.argmax(scores)
        all_scores.append(scores)
        scores_T.append(scores[iscore])
        zxys_T_.append(zxys_[iscore])
        hs_T.append(hs[iscore])
    zxys_T_ = np.array(zxys_T_)
    hs_T =np.array(hs_T)
    return zxys_T_,hs_T,scores_T,all_scores
def get_fov(fl):
        return int(os.path.basename(fl).split('--')[0].split('__')[-1])
class chromatin_postfits():
    def __init__(self,save_folder=r'\\BBFISH1\Raw_data_1\Glass_MERFISH\CGBB_1_25_2022_Analysis_v4'):
        self.save_folder = save_folder
        
        save_fl = save_folder+os.sep+'folders_fovs.pkl'
        if os.path.exists(save_fl):
            fovs,fls,Rfolders,Qfolders = pickle.load(open(save_fl,'rb'))
            self.fls_dics = [save_folder+os.sep+fov+'--'+os.path.basename(Rfld)+'--dic_pts_cell.pkl' 
                             for fov in fovs for Rfld in Rfolders]
            self.fls_dics_all = glob.glob(save_folder+os.sep+'*-dic_pts_cell.pkl')
            self.fls_dics = [fl for fl in self.fls_dics if fl in self.fls_dics_all]
        else:
            self.fls_dics = glob.glob(save_folder+os.sep+'*H*R*-dic_pts_cell.pkl')
        
        fls_dics = np.array(self.fls_dics)
        fovs_ = np.array([get_fov(fl) for fl in fls_dics])
        fovs,ncts = np.unique(fovs_,return_counts=True)
        dic_fls = {}
        for fov in fovs:
            dic_fls[fov]=fls_dics[fovs_==fov]
        self.dic_fls = dic_fls
        self.nHs = np.max([len(dic_fls[fov]) for fov in  dic_fls])
        self.completed_fovs = [fov for fov in  dic_fls if len(dic_fls[fov])==self.nHs]
        
        
        print("Detected fovs:",len(fovs),list(fovs))
        print("Detected complete fovs:",len(self.completed_fovs),self.completed_fovs )
        print("Detected number of hybes:",self.nHs)
        
        
    def load_fov(self,ifov,volume_th=200000,reorder=False):
        self.fov = ifov
        fls_dics = self.dic_fls[ifov]
        if reorder:
            fls_dics = np.array(fls_dics)[np.argsort([int(os.path.basename(fl).split('--H')[-1].split('_')[0])for fl in fls_dics])] 
        self.fls_fov = fls_dics
        dic_drifts = []
        dic_pts_cells = []
        #print(fls_dics)
        for fl in tqdm(fls_dics):
            dic_drift,dic_pts_cell = pickle.load(open(fl,'rb'))
            dic_pts_cells.append(dic_pts_cell)
            fl_  = fl.replace('dic_pts_cell.pkl','new_drift.pkl')
            if os.path.exists(fl_):
                dic_drift = pickle.load(open(fl_,'rb'))
            dic_drifts.append(dic_drift)

        self.dic_pts_cells = dic_pts_cells
        self.dic_drifts = dic_drifts
        self.cells = get_cells(self.dic_pts_cells,volume_th =volume_th)
        self.volume_th = volume_th

        print("Found cells: "+str(len(self.cells))) 
    def check_a_cell(self,icell_,nchr_=5,volume_th=None,pix=[0.200,0.1083,0.1083],
                     radius_chr = 1.25,enhanced_radius=1.25,fr_th=0.5,plt_val = False):
        dic_drifts,dic_pts_cells = self.dic_drifts,self.dic_pts_cells
        fls_fov = self.fls_fov  
        #if volume_th is not None:
        self.cells = get_cells(dic_pts_cells,volume_th =volume_th)
        cells = self.cells 
        icell = cells[icell_]
        Xs,hs,Rs,icols = get_X(dic_drifts,dic_pts_cells,fls_fov,icell,pix=pix,nelems = nchr_)

        print(len(Rs),len(icols))
        X_chrs = determine_number_of_chromosomes(Xs,hs,Rs,radius_chr = radius_chr,
                                                 enhanced_radius=enhanced_radius,
                                                 nchr_=nchr_,fr_th = fr_th,plt_val=plt_val)
        return X_chrs


    def get_X_cands(self,nchr_=5,volume_th=None,pix=[0.200,0.1083,0.1083],
                     radius_chr = 1.25,enhanced_radius=1.25,radius_cand =2,fr_th=0.5,nelems=50,plt_val = False):
        self.pix = pix
        
        if volume_th is None: volume_th = self.volume_th
        cells = self.cells
        fls_fov,dic_drifts,dic_pts_cells = self.fls_fov,self.dic_drifts,self.dic_pts_cells  

        X_cands = []
        icell_cands = []
        for icell in tqdm(cells):
            Xs,hs,Rs,icols = get_X(dic_drifts,dic_pts_cells,fls_fov,icell,pix=pix,nelems = nchr_)
            X_chrs = determine_number_of_chromosomes(Xs,hs,Rs,
                                        nchr_=nchr_,radius_chr = radius_chr,enhanced_radius=enhanced_radius,fr_th=fr_th
                                                     ,plt_val=False)
            if len(X_chrs)>0:

                Xs,hs,Rs,icols = get_X(dic_drifts,dic_pts_cells,fls_fov,icell,pix=pix,nelems = nelems)

                mat = cdist(X_chrs,Xs)
                nchr = len(X_chrs)
                best_asign = np.argmin(mat,axis=0)

                for ichr in range(nchr):
                    keep = (best_asign==ichr)&(mat[ichr]<radius_cand)
                    X_cands_ = np.array([Xs[keep,0],Xs[keep,1],Xs[keep,2],hs[keep,0],hs[keep,1],icols[keep],Rs[keep]]).T
                    X_cands.append(X_cands_)
                    icell_cands.append(icell)

        self.X_cands =X_cands
        self.icell_cands=icell_cands


        print("Detected number of chromosomes:" + str(len(self.icell_cands)))
        ploidy,ncells = np.unique(np.unique(self.icell_cands,return_counts=True)[-1],return_counts=True)
        for pl,nc in zip(ploidy,ncells):
            print("Number of cells with "+str(pl) +" chromosomes: "+str(nc))
    def initialize_with_max_brightness(self,nkeep = 8000,Rs_u = np.arange(177)+1):
        ### Initialize with maximum brightness########
        X_cands = self.X_cands
        self.Rs_u = Rs_u
        zxys_f,hs_f,hs_bk_f  = [],[],[]

        for pfits_cands_ in tqdm(X_cands[:nkeep]):
            zxys_T,hs_T,hs_bk_T = get_maxh_estimate(pfits_cands_,Rs_u=Rs_u)
            zxys_f.append(zxys_T)
            hs_f.append(hs_T)
            hs_bk_f.append(hs_bk_T)
        #hs_f = np.array(hs_f)# -np.array(hs_bk_f)

        self.zxys_f,self.hs_f,self.hs_bk_f = zxys_f,hs_f,hs_bk_f 

    def normalize_color_brightnesses(self):
        zxys_f,hs_f,hs_bk_f  = self.zxys_f,self.hs_f,self.hs_bk_f
        ### get dic_col
        X_cands = self.X_cands
        dic_col = {}
        for X in X_cands:
            Rs = X[:,-1]
            icols = X[:,-2]
            for R,icol in zip(Rs,icols):
                dic_col[R] = icol
        self.dic_col=dic_col

        cols = np.unique(list(dic_col.values()))

        hmed = np.nanmedian(np.array(hs_f),axis=0)
        Hths = np.array([np.nanmedian(hmed[[list(self.Rs_u).index(R) for R in dic_col if dic_col[R]==icol]]) 
                         for icol in cols])
        X_cands_ = [X.copy() for X in X_cands]
        for X in X_cands_:
            Rs = X[:,-1]
            icols = X[:,-2].astype(int)
            X[:,3]=X[:,3]/Hths[icols]
        self.X_cands_ = X_cands_
    def plot_std_col(self):
        hs_f = self.hs_f
        hs_bk_f = self.hs_bk_f
        iRs = np.arange(np.array(hs_f).shape[-1]//3)
        for icol in range(3):
            plt.figure()
            plt.title(str(icol))
            plt.plot(np.nanmedian(np.array(hs_f)[::2],axis=0)[icol::3],'o-')
            plt.plot(np.nanmedian(np.array(hs_f)[1::2],axis=0)[icol::3],'o-')
            plt.plot(np.nanmedian(np.array(hs_bk_f)[::2],axis=0)[icol::3],'o-')
            plt.plot(np.nanmedian(np.array(hs_bk_f)[1::2],axis=0)[icol::3],'o-')
            y = np.nanmedian(np.array(hs_f),axis=0)[icol::3]
            x = np.arange(len(y))
            for iR_,x_,y_ in zip(iRs,x,y):
                plt.text(x_,y_,str(iR_+1))

    def run_EM(self,nkeep = 8000,niter = 4,Rs_u = np.arange(175)+1):
        self.Rs_u = Rs_u
        #nkeep - Number of chromsomes to keep if want to check a subset of data 
        # iter = 4 #number of EM steps
        X_cands_ = self.X_cands_
        ### Initialize with maximum brightness########
        from tqdm import tqdm_notebook as tqdm
        zxys_f,hs_f,hs_bk_f  = [],[],[]

        for pfits_cands_ in tqdm(X_cands_[:nkeep]):
            zxys_T,hs_T,hs_bk_T = get_maxh_estimate(pfits_cands_,Rs_u = Rs_u)
            zxys_f.append(zxys_T)
            hs_f.append(hs_T)
            hs_bk_f.append(hs_bk_T)
        #hs_f = np.array(hs_f)# -np.array(hs_bk_f)

        ### Run to converge #########
        def refine_set(pfits_cands,zxys_f,hs_f,use_local=True,use_center=True,resample=1):
            Dhs,Ddists,DdistsC = get_Ddists_Dhs(zxys_f[::resample],hs_f[::resample],nint=5)
            zxys_f2,hs_f2,cols_f2,scores_f2,all_scores_f2  = [],[],[],[],[]
            i_ = 0
            for pfits_cands_ in tqdm(pfits_cands):
                    zxys_T,hs_T,scores_T,all_scores = get_statistical_estimate(pfits_cands_,Dhs,Ddists,DdistsC,
                                             zxys_T=zxys_f[i_],nint=5,use_local=use_local,use_center=use_center,Rs_u = Rs_u)
                    zxys_f2.append(zxys_T)
                    hs_f2.append(hs_T)
                    scores_f2.append(scores_T)
                    all_scores_f2.append(all_scores)
                    i_+=1
            return zxys_f2,hs_f2,scores_f2,all_scores_f2

        saved_zxys_f=[zxys_f[:nkeep]]
        save_hs_f=[hs_f[:nkeep]]

        for num_ref in range(niter):
            use_local = True#num_ref>=niter/2
            print('EM iteration number: ',num_ref+1)

            zxys_f,hs_f,scores_f,all_scores_f = refine_set(X_cands_[:nkeep],zxys_f[:nkeep],hs_f[:nkeep],use_local=use_local)
            saved_zxys_f.append(zxys_f)
            save_hs_f.append(hs_f)

            #check convergence
            dif = np.array(saved_zxys_f[-1])-np.array(saved_zxys_f[-2])
            nan =  np.all(np.isnan(dif),axis=-1)
            same = nan|np.all(dif==0,axis=-1)
            print("fraction the same:",np.sum(same)/float(np.prod(same.shape)))
            print("fraction nan:",np.sum(nan)/float(np.prod(nan.shape)))
        self.zxys_f,self.hs_f,self.scores_f = zxys_f,hs_f,scores_f
        self.all_scores_f = all_scores_f

    def get_scores_and_threshold(self,th_score = -6):
        scores_f,all_scores_f = self.scores_f,self.all_scores_f

        scores_all_ = [sc_ for scs in all_scores_f for sc in scs for sc_ in np.sort(sc)[:-1]]
        scores_good_ = [sc_ for scs in scores_f for sc_ in scs]
        scores_all_ = np.array(scores_all_)
        scores_all_ = scores_all_[~np.isnan(scores_all_)]
        scores_good_ = np.array(scores_good_)
        scores_good__ = scores_good_[~np.isnan(scores_good_)]

        plt.figure()
        plt.ylabel('Probability density')
        plt.xlabel('Log-score')
        plt.hist(np.log(scores_good__),density=True,bins=100,alpha=0.5,label='good spots');
        plt.hist(np.log(scores_all_),density=True,bins=100,alpha=0.5,label='background spots');
        plt.legend()



        plt.figure()
        plt.plot(np.mean(np.log(scores_f)>th_score,axis=0),'o-')
        plt.ylabel('Detection efficiency')
        plt.xlabel('Region')
        plt.figure()
        det_ef = np.mean(np.log(scores_f)>th_score,axis=1)
        plt.plot(det_ef)
        plt.title("Median detection efficiency: "+str(np.round(np.median(det_ef),2)))
        plt.ylabel('Detection efficiency')
        plt.xlabel('Chromosome')


    def plot_matrix(self,th_score=-5,lazy_color_correction = True):
        self.th_score = th_score
        Xf = np.array(self.zxys_f)
        bad = np.log(self.scores_f)<th_score
        Xf[bad] = np.nan
        if lazy_color_correction:
            ncol=3
            cm = np.nanmean(Xf[:,:,:],axis=1)[:,np.newaxis]
            for icol in range(ncol):
                Xf[:,icol::ncol,:]-=np.nanmedian(Xf[:,icol::ncol,:],axis=1)[:,np.newaxis]+cm

        from scipy.spatial.distance import pdist,squareform
        mats = np.array([squareform(pdist(X_)) for X_ in Xf])

        plt.figure(figsize=(10,10))
        keep = np.arange(mats.shape[1])#[icol::3]
        plt.imshow(np.nanmedian(mats[:,keep][:,:,keep],0),vmax=1,vmin=0.2,cmap='seismic_r')

        if False:
            from scipy.spatial.distance import pdist,squareform
            mats = np.array([squareform(pdist(X_)) for X_ in Xf])
            for icol in range(3):
                plt.figure()
                keep = np.arange(mats.shape[1])[icol::3]
                plt.imshow(np.nanmedian(mats[:,keep][:,:,keep],0),vmax=1,vmin=0.2,cmap='seismic_r')
class chromatin_postfits():
    def __init__(self,save_folder =r'X:\Glass_MERFISH\CGBB_ChrX_04_16_2023_Analysis',
                 tag = '*--H*R*_C--_Xh_RNAs.pkl',exclude_tag = None,tag_save='_C',remap=False,nHs=None):
        self.save_folder = save_folder

        save_fl = save_folder+os.sep+'folders_fovs'+tag_save+'.npy'
        if not os.path.exists(save_fl) or remap:
            self.fls_dics = glob.glob(save_folder+os.sep+tag)
            if exclude_tag is not None:
                self.fls_exclude = glob.glob(save_folder+os.sep+exclude_tag)
                self.fls_dics = np.setdiff1d(self.fls_dics,self.fls_exclude)
            np.save(save_fl,self.fls_dics)
        else:
            self.fls_dics = np.load(save_fl)
        fls_dics = np.array(self.fls_dics)
        fovs_ = np.array([get_fov(fl) for fl in fls_dics])
        fovs,ncts = np.unique(fovs_,return_counts=True)
        dic_fls = {}
        for fov in fovs:
            dic_fls[fov]=fls_dics[fovs_==fov]
        self.dic_fls = dic_fls
        print([len(dic_fls[fov]) for fov in  dic_fls])
        if nHs is None:
            self.nHs = np.max([len(dic_fls[fov]) for fov in  dic_fls])
        else:
            self.nHs=nHs
        self.completed_fovs = [fov for fov in  dic_fls if len(dic_fls[fov])>=self.nHs]


        print("Detected fovs:",len(fovs),list(fovs))
        print("Detected complete fovs:",len(self.completed_fovs),self.completed_fovs )
        print("Detected number of hybes:",self.nHs)
def get_Htag(fl): return os.path.basename(fl).split('--')[1]
def get_fullfov(fl): return os.path.basename(fl).split('--')[0]
def get_icol(fl): return int(os.path.basename(fl).split('--col')[-1].split('_')[0])
def get_R(fl): 
    try:
        icol = get_icol(fl)
        return np.array(get_Htag(fl).replace('R','').split('_')[1:],dtype=int)[icol]
    except:
        return -1

def reorder_on_R(fls):
    return np.array(fls)[np.argsort([get_R(fl) for fl in fls])]
def get_raw_fl(fl,raw_folder=r'X:\Glass_MERFISH\CGBB_ChrX_04_16_2023'):
    return raw_folder+os.sep+get_Htag(fl)+os.sep+get_fullfov(fl)+'.zarr'

from ioMicro import *
def align(fl1,fl2,szz = 50):
    imT = read_im(fl2)
    ncols,sz,sx,sy = imT.shape
    im0 = np.array(imT[-1][(sz-szz)//2:(sz+szz)//2],dtype=np.float32)
    im1 = np.array(read_im(fl1)[-1][(sz-szz)//2:(sz+szz)//2],dtype=np.float32)
    txyz,txyzs = get_txyz(im0,im1,sz=500,sz_norm=30,nelems=5)
    return txyz,txyzs
from scipy import ndimage as nd
def get_int_im1_im2(im1,im2,th_int=0.5):
    inters = ((im1>0)&(im2>0)).astype(int)
    im1_in2 = im1*inters
    N1max = np.max(im1)+1
    im2_in1 = im2*inters*N1max
    iint,cts = np.unique(im1_in2+im2_in1,return_counts=True)
    c1,cts1 = np.unique(im1,return_counts=True)
    dic_c1 = {c_:ct_ for c_,ct_ in zip(c1,cts1) if c_>0}
    c2,cts2 = np.unique(im2,return_counts=True)
    dic_c2 = {c_:ct_ for c_,ct_ in zip(c2,cts2) if c_>0}
    dic_int= {(c1,c2):(ct/dic_c1[c1],ct/dic_c2[c2]) for c2,c1,ct in zip(iint//N1max,iint%N1max,cts) 
         if (c1>0) and (c2>0) and (c1!=c2)}
    objs1 = nd.find_objects(im1)
    objs2 = nd.find_objects(im2)
    for cch in dic_int:
        c1,c2 = cch
        ic1,ic2 = dic_int[cch]
        obj1,obj2 = objs1[c1-1],objs2[c2-1]
        if (ic1>th_int) or (ic2>th_int):
            c_ = np.min([c1,c2])
            im1[obj1][im1[obj1]==c1]=c_
            im2[obj2][im2[obj2]==c2]=c_
    return im1,im2
def stitch3D(im_segm,niter=5,th_int=0.5):
    for it_ in range(niter):
        for iim in range(len(im_segm)-1):
            im_segm[iim],im_segm[iim+1] = get_int_im1_im2(im_segm[iim],im_segm[iim+1],th_int=th_int)
    return im_segm
def load_segmentation(dec,folder_tag = 'SegmentationLamin'):
    dec.fl_dapi = glob.glob(dec.save_folder+os.sep+folder_tag+os.sep+dec.fov+'*'+dec.set_+'*.npz')[0]
    dic = np.load(dec.fl_dapi)
    im_segm = dic['segm']
    dec.shape = dic['shape']
    dec.im_segm_=im_segm#stitch3D(im_segm,niter=5,th_int=0.75)
def keep_best_per_cell(XH_f,nbest=20):
    Rs = XH_f[:,-2].astype(int) #readout
    CELLs = XH_f[:,-1].astype(int) #cell indesx
    cells = np.unique(CELLs)
    cells = cells[cells>0]
    uRs = np.unique(Rs)
    XH_fs=None
    for R in tqdm(uRs):
        kp = (Rs==R)
        XH_ff = XH_f[kp]
        H = XH_ff[:,-4] #brightness
        C = CELLs[kp]
        for cell in cells:
            kpc = C==cell
            if np.sum(kpc):
                kpH = np.argsort(H[kpc])[-nbest:]
                if XH_fs is None:
                    XH_fs = XH_ff[kpc][kpH]
                else:
                    XH_fs = np.concatenate([XH_fs,XH_ff[kpc][kpH]])
    return XH_fs
    
    
def keep_best_per_cell_fast(XH_f,nbest=20):
    
    ###sort molecules by brightness and keep only within cells (CELLs>0)
    XH_f_ = XH_f.copy()
    H = XH_f_[:,-4] #brightness
    XH_f_ = XH_f[np.argsort(H)]
    CELLs = XH_f_[:,-1].astype(int) #cell index
    keep = CELLs>0
    XH_f_ = XH_f_[keep]
    
    
    #################################
    CELLs = XH_f_[:,-1].astype(int) #cell index
    H = XH_f_[:,-4] #brightness
    ### find index combining cell id an redout id
    Rs = XH_f_[:,-2].astype(int) #readout
    
    
    
    
    cellsu = np.unique(CELLs)
    Rsu = np.unique(Rs)
    MR = np.max(Rsu)+1
    mix_index = CELLs*MR+Rs ### unique index combining cell and readout info
    
    #### find the inverse and prealocate space that you fill cyclically
    miu,rinv = np.unique(mix_index,return_inverse=True)
    miu_,cts_ = np.unique(mix_index,return_counts=True)
    XH_f_pruned = np.zeros([len(miu),nbest,XH_f.shape[1]],dtype=XH_f.dtype)+np.nan
    ind_best = np.argsort(np.argsort(rinv))%nbest
    XH_f_pruned[rinv,ind_best,:]=XH_f_######
    XH_f_pruned = XH_f_pruned.reshape([-1,XH_f_pruned.shape[-1]])
    good = ~np.isnan(XH_f_pruned)[:,0]
    XH_f_pruned = XH_f_pruned[good]
    return XH_f_pruned
def get_X(XH_fsTB,cell,pix=[0.2,0.10833,0.10833]):
    XH_cell = XH_fsTB[XH_fsTB[:,-1]==cell]
    hs = XH_cell[:,-4]
    Xs = XH_cell[:,:3]*pix
    Rs = XH_cell[:,-2]
    icols = XH_cell[:,-3]
    return Xs,hs,Rs,icols # For a given cell, Xs: 3D coordinate in nm; hs: brightness; Rs: bit index; icols: colors
def get_cells(XH):
    cells = np.unique(XH[:,-1])
    return cells[cells>0]
    
def get_X_cands(self,XH_fs,nchr_=2,pix=[0.300,0.1083,0.1083],
                     radius_chr = 2,radius_cand =4,fr_th=0.4,nelems=2,plt_val = False):
        self.pix = pix
        XH_fsTB = keep_best_per_cell_fast(XH_fs,nbest=nelems) # find nbest brightest spots per cell
        cells = get_cells(XH_fsTB) # fetch cell id
        self.cells = cells
        X_cands = []
        icell_cands = []
        for icell in tqdm(cells):
            Xs,hs,Rs,icols = get_X(XH_fsTB,icell,pix=pix) # get X
            X_chrs = determine_number_of_chromosomes(Xs,hs,Rs,nchr_=nchr_,radius_chr = radius_chr,fr_th=fr_th,plt_val=False)
            
            
            if len(X_chrs)>0:
                Xs,hs,Rs,icols = get_X(XH_fs,icell,pix=pix)
                mat = cdist(X_chrs,Xs)
                nchr = len(X_chrs)
                best_asign = np.argmin(mat,axis=0)

                for ichr in range(nchr):
                    keep = (best_asign==ichr)&(mat[ichr]<radius_cand)
                    X_cands_ = np.array([Xs[keep,0],Xs[keep,1],Xs[keep,2],hs[keep],hs[keep],icols[keep],Rs[keep]]).T
                    X_cands.append(X_cands_)
                    icell_cands.append(icell)

        self.X_cands =X_cands
        self.icell_cands=icell_cands


        print("Detected number of chromosomes:" + str(len(self.icell_cands)))
        ploidy,ncells = np.unique(np.unique(self.icell_cands,return_counts=True)[-1],return_counts=True)
        for pl,nc in zip(ploidy,ncells):
            print("Number of cells with "+str(pl) +" chromosomes: "+str(nc))           
            
def initialize_with_max_brightness(self,nkeep = 8000,Rs_u = np.arange(177)+1):
    ### Initialize with maximum brightness########
    X_cands = self.X_cands
    self.Rs_u = Rs_u
    zxys_f,hs_f,hs_bk_f  = [],[],[]

    for pfits_cands_ in tqdm(X_cands[:nkeep]):
        zxys_T,hs_T,hs_bk_T = get_maxh_estimate(pfits_cands_,Rs_u=Rs_u)
        zxys_f.append(zxys_T)
        hs_f.append(hs_T)
        hs_bk_f.append(hs_bk_T)
    #hs_f = np.array(hs_f)# -np.array(hs_bk_f)

    self.zxys_f,self.hs_f,self.hs_bk_f = zxys_f,hs_f,hs_bk_f 

def normalize_color_brightnesses(self):
    zxys_f,hs_f,hs_bk_f  = self.zxys_f,self.hs_f,self.hs_bk_f
    ### get dic_col
    X_cands = self.X_cands
    dic_col = {}
    for X in X_cands:
        Rs = X[:,-1]
        icols = X[:,-2]
        for R,icol in zip(Rs,icols):
            dic_col[R] = icol
    self.dic_col=dic_col

    cols = np.unique(list(dic_col.values()))

    hmed = np.nanmedian(np.array(hs_f),axis=0)
    Hths = np.array([np.nanmedian(hmed[[list(self.Rs_u).index(R) for R in dic_col if dic_col[R]==icol]]) 
                     for icol in cols])
    X_cands_ = [X.copy() for X in X_cands]
    for X in X_cands_:
        Rs = X[:,-1]
        icols = X[:,-2].astype(int)
        X[:,3]=X[:,3]/Hths[icols]
    self.X_cands_ = X_cands_
def plot_std_col(self):
    hs_f = self.hs_f
    hs_bk_f = self.hs_bk_f
    iRs = np.arange(np.array(hs_f).shape[-1]//3)
    for icol in range(3):
        plt.figure()
        plt.title(str(icol))
        plt.plot(np.nanmedian(np.array(hs_f)[::2],axis=0)[icol::3],'o-')
        plt.plot(np.nanmedian(np.array(hs_f)[1::2],axis=0)[icol::3],'o-')
        plt.plot(np.nanmedian(np.array(hs_bk_f)[::2],axis=0)[icol::3],'o-')
        plt.plot(np.nanmedian(np.array(hs_bk_f)[1::2],axis=0)[icol::3],'o-')
        y = np.nanmedian(np.array(hs_f),axis=0)[icol::3]
        x = np.arange(len(y))
        for iR_,x_,y_ in zip(iRs,x,y):
            plt.text(x_,y_,str(iR_+1))

def run_EM(self,nkeep = 8000,niter = 4,Rs_u = np.arange(175)+1):
    self.Rs_u = Rs_u
    #nkeep - Number of chromsomes to keep if want to check a subset of data 
    # iter = 4 #number of EM steps
    X_cands_ = self.X_cands_
    ### Initialize with maximum brightness########
    #from tqdm import tqdm_notebook as tqdm
    zxys_f,hs_f,hs_bk_f  = [],[],[]

    for pfits_cands_ in tqdm(X_cands_[:nkeep]):
        zxys_T,hs_T,hs_bk_T = get_maxh_estimate(pfits_cands_,Rs_u = Rs_u)
        zxys_f.append(zxys_T)
        hs_f.append(hs_T)
        hs_bk_f.append(hs_bk_T)
    #hs_f = np.array(hs_f)# -np.array(hs_bk_f)

    ### Run to converge #########
    def refine_set(pfits_cands,zxys_f,hs_f,use_local=True,use_center=True,resample=1):
        Dhs,Ddists,DdistsC = get_Ddists_Dhs(zxys_f[::resample],hs_f[::resample],nint=5)
        zxys_f2,hs_f2,cols_f2,scores_f2,all_scores_f2  = [],[],[],[],[]
        i_ = 0
        for pfits_cands_ in tqdm(pfits_cands):
                zxys_T,hs_T,scores_T,all_scores = get_statistical_estimate(pfits_cands_,Dhs,Ddists,DdistsC,
                                         zxys_T=zxys_f[i_],nint=5,use_local=use_local,use_center=use_center,Rs_u = Rs_u)
                zxys_f2.append(zxys_T)
                hs_f2.append(hs_T)
                scores_f2.append(scores_T)
                all_scores_f2.append(all_scores)
                i_+=1
        return zxys_f2,hs_f2,scores_f2,all_scores_f2

    saved_zxys_f=[zxys_f[:nkeep]]
    save_hs_f=[hs_f[:nkeep]]

    for num_ref in range(niter):
        use_local = True#num_ref>=niter/2
        print('EM iteration number: ',num_ref+1)

        zxys_f,hs_f,scores_f,all_scores_f = refine_set(X_cands_[:nkeep],zxys_f[:nkeep],hs_f[:nkeep],use_local=use_local)
        saved_zxys_f.append(zxys_f)
        save_hs_f.append(hs_f)

        #check convergence
        dif = np.array(saved_zxys_f[-1])-np.array(saved_zxys_f[-2])
        nan =  np.all(np.isnan(dif),axis=-1)
        same = nan|np.all(dif==0,axis=-1)
        print("fraction the same:",np.sum(same)/float(np.prod(same.shape)))
        print("fraction nan:",np.sum(nan)/float(np.prod(nan.shape)))
    self.zxys_f,self.hs_f,self.scores_f = zxys_f,hs_f,scores_f
    self.all_scores_f = all_scores_f

def get_scores_and_threshold(self,th_score = -6):
    scores_f,all_scores_f = self.scores_f,self.all_scores_f

    scores_all_ = [sc_ for scs in all_scores_f for sc in scs for sc_ in np.sort(sc)[:-1]]
    scores_good_ = [sc_ for scs in scores_f for sc_ in scs]
    scores_all_ = np.array(scores_all_)
    scores_all_ = scores_all_[~np.isnan(scores_all_)]
    scores_good_ = np.array(scores_good_)
    scores_good__ = scores_good_[~np.isnan(scores_good_)]

    plt.figure()
    plt.ylabel('Probability density')
    plt.xlabel('Log-score')
    plt.hist(np.log(scores_good__),density=True,bins=100,alpha=0.5,label='good spots');
    plt.hist(np.log(scores_all_),density=True,bins=100,alpha=0.5,label='background spots');
    plt.legend()



    plt.figure()
    plt.plot(np.mean(np.log(scores_f)>th_score,axis=0),'o-')
    plt.ylabel('Detection efficiency')
    plt.xlabel('Region')
    plt.figure()
    det_ef = np.mean(np.log(scores_f)>th_score,axis=1)
    plt.plot(det_ef)
    plt.title("Median detection efficiency: "+str(np.round(np.median(det_ef),2)))
    plt.ylabel('Detection efficiency')
    plt.xlabel('Chromosome')


def plot_matrix(self,th_score=-5,lazy_color_correction = True,vmin=0.2,vmax=1):
    self.th_score = th_score
    Xf = np.array(self.zxys_f)
    bad = np.log(self.scores_f)<th_score
    Xf[bad] = np.nan
    if lazy_color_correction:
        ncol=3
        cm = np.nanmean(Xf[:,:,:],axis=1)[:,np.newaxis]
        for icol in range(ncol):
            Xf[:,icol::ncol,:]-=np.nanmedian(Xf[:,icol::ncol,:],axis=1)[:,np.newaxis]+cm

    from scipy.spatial.distance import pdist,squareform
    mats = np.array([squareform(pdist(X_)) for X_ in Xf])

    plt.figure(figsize=(10,10))
    keep = np.arange(mats.shape[1])#[icol::3]
    plt.imshow(np.nanmedian(mats[:,keep][:,:,keep],0),vmax=vmax,vmin=vmin,cmap='seismic_r')

    if False:
        from scipy.spatial.distance import pdist,squareform
        mats = np.array([squareform(pdist(X_)) for X_ in Xf])
        for icol in range(3):
            plt.figure()
            keep = np.arange(mats.shape[1])[icol::3]
            plt.imshow(np.nanmedian(mats[:,keep][:,:,keep],0),vmax=vmax,vmin=vmin,cmap='seismic_r')
            
def get_XB(im_,th=3):
    #im_ = self.im1n
    std_ = np.std(im_[::5,::5,::5])
    keep = im_>std_*th
    XB = np.array(np.where(keep)).T
    from tqdm import tqdm
    for delta_fit in tqdm([1,2,3,5,7,10,15]):
        XI = np.indices([2*delta_fit+1]*3)-delta_fit
        keep = (np.sum(XI*XI,axis=0)<=(delta_fit*delta_fit))
        XI = XI[:,keep].T
        XS = (XB[:,np.newaxis]+XI[np.newaxis])
        shape = im_.shape
        XS = XS%shape

        keep = im_[tuple(XB.T)]>=np.max(im_[tuple(XS.T)],axis=0)
        XB = XB[keep]
    return XB
def get_max_min(P,imn,delta_fit=5,ismax=True,return_ims=False):
    XI = np.indices([2*delta_fit+1]*3)-delta_fit
    keep = (np.sum(XI*XI,axis=0)<=(delta_fit*delta_fit))
    XI = XI[:,keep].T
    XS = (P[:,np.newaxis]+XI[np.newaxis])
    shape = imn.shape
    XSS = XS.copy()
    XS = XS%shape
    #XSS = XS.copy()
    is_bad = np.any((XSS!=XS),axis=-1)


    sh_ = XS.shape
    XS = XS.reshape([-1,3])
    im1n_local = imn[tuple(XS.T)].reshape(sh_[:-1])
    #print(is_bad.shape,im1n_local.shape)
    im1n_local[is_bad]=np.nan

    im1n_local = im1n_local-np.nanmin(im1n_local,axis=1)[:,np.newaxis]
    im1n_local = im1n_local/np.nansum(im1n_local,axis=1)[:,np.newaxis]
    XS = XS.reshape(list(im1n_local.shape)+[3])
    return np.nansum(im1n_local[...,np.newaxis]*XS,axis=1)
def normalize_ims(im0,zm=5,zM=50):
    imn = np.array([cv2.blur(im_,(zm,zm))-cv2.blur(im_,(zM,zM)) for im_ in im0])
    return imn

    
def get_best_translation_pointsV2(fl,fl_ref,resc=5):
    obj = np.load(fl)
    obj_Xh_plus  = obj['Xh_plus']
    obj_Xh_minus  = obj['Xh_minus']
    
    
    obj_ref = np.load(fl_ref)
    obj_ref_Xh_plus  = obj_ref['Xh_plus']
    obj_ref_Xh_minus  = obj_ref['Xh_minus']
    
    tzxyf,tzxy_plus,tzxy_minus,N_plus,N_minus = np.array([0,0,0]),np.array([0,0,0]),np.array([0,0,0]),0,0
    if (len(obj_Xh_plus)>0) and (len(obj_ref_Xh_plus)>0):
        X = obj_Xh_plus[:,:3]
        X_ref = obj_ref_Xh_plus[:,:3]
        tzxy_plus,N_plus = get_best_translation_points(X,X_ref,resc=resc,return_counts=True)
    if (len(obj_Xh_minus)>0) and (len(obj_ref_Xh_minus)>0):
        X = obj_Xh_minus[:,:3]
        X_ref = obj_ref_Xh_minus[:,:3]
        tzxy_minus,N_minus = get_best_translation_points(X,X_ref,resc=resc,return_counts=True)
    if np.max(np.abs(tzxy_minus-tzxy_plus))<=2:
        tzxyf = -(tzxy_plus*N_plus+tzxy_minus*N_minus)/(N_plus+N_minus)
    else:
        tzxyf = -[tzxy_plus,tzxy_minus][np.argmax([N_plus,N_minus])]
    return [tzxyf,tzxy_plus,tzxy_minus,N_plus,N_minus]
def fit_fl_TO_dapiFeat_fl(fitfl):
    return fitfl.split('--col')[0]+'--dapiFeatures.npz'