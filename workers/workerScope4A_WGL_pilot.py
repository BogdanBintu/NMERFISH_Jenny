master_analysis_folder = '/mnt/tscc/wed009/WholeGenome'

### Did you compute PSF and median flat field images?
psf_file = '/mnt/tscc/wed009/WholeGenome/fits_scripts/NMERFISH/psfs/psf_750_Scope4_final.npy'
flat_field_tag = '/mnt/tscc/wed009/WholeGenome/fits_scripts/NMERFISH/flat_field/Scope4_'#med_col_rawXXX.npz

master_data_folder = ['/projects/ps-renlab2/wed009/031524_v1_acry_37C_pilot/']
save_folder ='/projects/ps-renlab2/wed009/analysis_031524_v1_acry_37C_pilot/'


from multiprocessing import Pool, TimeoutError
import time,sys
import os,sys,numpy as np

sys.path.append(master_analysis_folder)
from ioMicro import *

def main_do_compute_fits(save_folder,fld,fov,icol,save_fl,psf,old_method):
    im_ = read_im(fld+os.sep+fov)
    im__ = np.array(im_[icol],dtype=np.float32)
    
    if old_method:
        ### previous method - no deconvolution
        im_n = norm_slice(im__,s=30)
        Xh = get_local_maxfast_tensor(im_n,th_fit=500,im_raw=im__,dic_psf=None,delta=1,delta_fit=3,sigmaZ=1,sigmaXY=1.5,gpu=False)
    else:
        ### new method - using deconvolution
        fl_med = flat_field_tag+'med_col_raw'+str(icol)+'.npz'

        im_med = np.array(np.load(fl_med)['im'],dtype=np.float32)
        im_med = cv2.blur(im_med,(20,20))
        im__ = im__/im_med*np.median(im_med)


        Xh = get_local_max_tile(im__,th=1000,s_ = 500,pad=100,psf=psf,plt_val=None,snorm=30,gpu=True,
                                deconv={'method':'wiener','beta':0.0001},
                                delta=1,delta_fit=3,sigmaZ=1,sigmaXY=1.5) # TODO: th = 1000 set bc of cy3

    np.savez_compressed(save_fl,Xh=Xh)
    
def compute_fits(save_folder,fov,all_flds,redo=False,ncols=4,
                psf_file = psf_file,try_mode=True,old_method=False):
    """This wraps main_do_compute_fits to allow for try_mode"""
    psf = np.load(psf_file)
    
    for fld in tqdm(all_flds):
        for icol in range(ncols-1):
            tag = os.path.basename(fld)
            save_fl = save_folder+os.sep+fov.split('.')[0]+'--'+tag+'--col'+str(icol)+'__Xhfits.npz'
            try:
                np.load(save_fl)['Xh']
                redo2 = False
            except:
                redo2 = True
            if not os.path.exists(save_fl) or redo or redo2:
                if try_mode:
                    try:
                        main_do_compute_fits(save_folder,fld,fov,icol,save_fl,psf,old_method)
                    except:
                        print("Failed",fld,fov,icol)
                else:
                    main_do_compute_fits(save_folder,fld,fov,icol,save_fl,psf,old_method)                   

def get_iH(fld): 
    """
    Given a folder <fld> of type /projects/ps-renlab2/wed009/H1_P1_A1_2_3/. this extracts the hybe index (i.e. 2 in this case)
    """
    try:
        return int(os.path.basename(fld).split('_')[0][1:]) #TODO: modified for WGL
    except:
        return np.inf

def get_files(ifov):

    if not os.path.exists(save_folder): os.makedirs(save_folder)
    all_flds = []
    for master_folder in master_data_folder:
        all_flds += glob.glob(master_folder+r'/H*') # TODO: modified for WGL

        
    
    ### reorder based on hybe index
    all_flds = np.array(all_flds)[np.argsort([get_iH(fld)for fld in all_flds])] 
    
    ### find all the fovs
    fovs_fl = save_folder+os.sep+'fovs__.npy'
    
    if not os.path.exists(fovs_fl):
        folder_map_fovs = all_flds[-1]
        fls = glob.glob(folder_map_fovs+os.sep+'*.zarr')
        fovs = np.sort([os.path.basename(fl) for fl in fls])
        np.save(fovs_fl,fovs)
    else:
        fovs = np.sort(np.load(fovs_fl))
    fov=None
    if ifov<len(fovs):
        fov = fovs[ifov]
        all_flds = [fld for fld in all_flds if os.path.exists(fld+os.sep+fov)]
    return save_folder,all_flds,fov
        

def compute_drift_features(save_folder,fov,all_flds,redo=False,gpu=True):
    fls = [fld+os.sep+fov for fld in all_flds]
    set_ = ''
    for fl in fls:
        get_dapi_features(fl,save_folder,set_,gpu=gpu,im_med_fl = flat_field_tag+r'med_col_raw3.npz',
                    psf_fl = psf_file) ### devonvolve the dapi image and fit local minima and maxima (dapi features)
                    
def get_best_translation_pointsV2(fl,fl_ref,save_folder,resc=5,th=4):
    ### THis loads the dapi features and registers the images for files fl and fl_ref
    set_ = ''
    obj = get_dapi_features(fl,save_folder,set_)
    obj_ref = get_dapi_features(fl_ref,save_folder,set_)
    tzxyf,tzxy_plus,tzxy_minus,N_plus,N_minus = np.array([0,0,0]),np.array([0,0,0]),np.array([0,0,0]),0,0
    if (len(obj.Xh_plus)>0) and (len(obj_ref.Xh_plus)>0):
        X = obj.Xh_plus#[:,:3]
        X_ref = obj_ref.Xh_plus#[:,:3]
        X = X[X[:,-1]>th][:,:3]
        X_ref = X_ref[X_ref[:,-1]>th][:,:3]
        tzxy_plus,N_plus = get_best_translation_points(X,X_ref,resc=resc,return_counts=True)
    if (len(obj.Xh_minus)>0) and (len(obj_ref.Xh_minus)>0):
        X = obj.Xh_minus#[:,:3]
        X_ref = obj_ref.Xh_minus#[:,:3]
        X = X[X[:,-1]>th][:,:3]
        X_ref = X_ref[X_ref[:,-1]>th][:,:3]
        tzxy_minus,N_minus = get_best_translation_points(X,X_ref,resc=resc,return_counts=True)
    if np.max(np.abs(tzxy_minus-tzxy_plus))<=2:
        tzxyf = -(tzxy_plus*N_plus+tzxy_minus*N_minus)/(N_plus+N_minus)
    else:
        tzxyf = -[tzxy_plus,tzxy_minus][np.argmax([N_plus,N_minus])]
    

    return [tzxyf,tzxy_plus,tzxy_minus,N_plus,N_minus]
def compute_drift_V2(save_folder,fov,all_flds,redo=False,gpu=True):
    set_ = ''
    drift_fl = save_folder+os.sep+'driftNew_'+fov.split('.')[0]+'--'+set_+'.pkl'
    try:
        np.load(drift_fl,allow_pickle=True)
    except:
        redo=True
    if not os.path.exists(drift_fl) or redo:
        fls = [fld+os.sep+fov for fld in all_flds]
        fl_ref = fls[len(fls)//2]
        newdrifts = []
        for fl in fls:
            drft = get_best_translation_pointsV2(fl,fl_ref,save_folder,resc=5)
            print(drft)
            newdrifts.append(drft)
        pickle.dump([newdrifts,all_flds,fov,fl_ref],open(drift_fl,'wb'))
def compute_main_f(save_folder,all_flds,fov,ifov,redo_fits,redo_drift,try_mode,old_method):
    print("Computing fitting on: "+str(fov))
    print(len(all_flds),all_flds)
    compute_fits(save_folder,fov,all_flds,redo=redo_fits,try_mode=try_mode,old_method=old_method)
    print("Computing drift on: "+str(fov))
    compute_drift_features(save_folder,fov,all_flds,redo=False,gpu=True)
    compute_drift_V2(save_folder,fov,all_flds,redo=redo_drift,gpu=True)

def main_f(ifov,redo_fits = False,redo_drift=False,try_mode=True,old_method=False):
    save_folder,all_flds,fov = get_files(ifov)
    if try_mode:
        try:
            compute_main_f(save_folder,all_flds,fov,ifov,redo_fits,redo_drift,try_mode,old_method)
        except:
            print("Failed within the main analysis:")
    else:
        compute_main_f(save_folder,all_flds,fov,ifov,redo_fits,redo_drift,try_mode,old_method)
    
    return ifov
    

    
if __name__ == '__main__':
    # start 4 worker processes
    items = np.arange(32) # number of FOVs
              
    
    #main_f(ifov = 5,try_mode=False)
    
    if True:
        with Pool(processes=2) as pool:
            print('starting pool')
            result = pool.map(main_f, items)
#activate cellpose&&python C:\Scripts\NMERFISH_Jenny\workers\workerScope4A_FFBB_DNA-RNA-FISH.py
# on mediator, conda activate cellpose
# CUDA_VISIBLE_DEVICES=0,1 python workerScope4A_WGL_pilot.py