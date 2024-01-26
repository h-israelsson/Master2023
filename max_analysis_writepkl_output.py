import tifffile
import python_test
import pandas as pd
import numpy as np
# import skimage
from scipy import ndimage
from cellpose import models
import matplotlib.pyplot as plt
import os
import pickle

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    else:
        print(f"Folder '{folder_path}' already exists.")

def filtered_masks(masks, common_cells):
    """Set all labels in masks not in common_cells to 0."""
    filter = np.isin(masks, common_cells)
    masks[~filter] = 0
    return masks

def extract_border_values(flt_masks, cell_lbl):
    def get_border_values(mask, cell_lbl):
        padded_mask = np.pad(mask, 1, "constant", constant_values=0) # Wherever there are no neighboring cells, we 
                                                                        # want zeroes, also on edges of image
        fltr = ndimage.binary_dilation(padded_mask==cell_lbl, structure=np.ones((3, 3)))
        bv = padded_mask[fltr]
        bv = bv[bv!=cell_lbl]
        return bv

    bv_count = 10
    bv_final = []
    for m in flt_masks:
        if not np.any(m==cell_lbl):
            continue
        bv = get_border_values(m,cell_lbl)
        if not np.any(bv==0):
            return bv
        q = len(bv[bv==0])/len(bv) # Check how much of the border is 0
        if q<0.05: # We're satisfied if 95% of the border is nonzero
            return bv
        if q<bv_count:
            bv_count = q
            bv_final = bv
    return bv_final

def get_ncc_for_bv(d, c, max_dt):
    bvs = np.array(d[c]['border_values'])
    bvs = bvs[bvs!=0]
    bvs_uniq = np.unique(bvs)
    ncc = 0
    dts = 0
    for bv in bvs_uniq:
        xcorr, dt = python_test.get_cc(np.array(d[c]['intensities']),
                        np.array(d[bv]['intensities']), max_dt)
        w = len(bvs[bvs==bv])/len(bvs[bvs!=0])
        ncc += np.max(xcorr)*w
        dts += dt[np.argmax(xcorr)]*w
    return ncc, dts

def get_max_freq(intensities, T):

    fft = np.fft.fft(intensities)
    freqs = np.fft.fftfreq(len(intensities), T)
    fft_filtered = fft*(freqs>(3/(3600))) #Remove all periods longer than 20 min  
        
    return np.abs(freqs[np.argmax(fft_filtered)])

def do_it2(X, masks, occurrence_limit=50, T=10, max_dt = 30):
    
    common_cells, counts = python_test.get_common_cells(masks, occurrence=occurrence_limit) # Get the cells to include in the intensity measurements
    fltr_masks = filtered_masks(masks, common_cells)
    d = {}

    for c in common_cells:
        print(c)
        # get intensities
        intensity = python_test.get_cell_intensities(c, masks, X, T=T)
        
        # get COMS
        coms = []
        for im in range(len(masks)):
            if np.any(masks[im]==c):
                coms.append(list(python_test.get_centers_of_mass(masks[im], c)[0]))
                break
        
        # get border values
        boarder_values = list(extract_border_values(fltr_masks, c))
        d[c] = {'intensities': intensity, 'com': coms, 'border_values': boarder_values}
    
    # get cross correlation
    for c in common_cells:
        d[c]['Weighted max NCC for border values'], d[c]['Weighted time difference at max NCC for border values'] = get_ncc_for_bv(d, c, max_dt)

        # get most prominent frequency
        d[c]['Most prominent frequency'] = get_max_freq(d[c]['intensities'], T)

    return d
# __________________________________________________________________________________________


force_redo = False
masks_path = '/home/sim/Data/Max/gcamp/masks_tracked/'
tif_path = '/home/sim/Data/Max/gcamp/tif/'
save_path = '/home/sim/Data/Max/gcamp/pkl/'

all_masks = os.listdir(masks_path)
create_folder(save_path)
print(all_masks)

for f in all_masks:
    save_name = f[:-4] + '.pkl'
    if save_name in os.listdir(save_path) and force_redo == False:
        print('PKL for ' + f + ' already exists.')
        continue
    else:
        print(f)
        X = tifffile.imread(tif_path + f)
        masks = tifffile.imread(masks_path + f)
        index = masks.shape[0]
        X = X[:index,:,:]
        d = do_it2(
            X,
            masks,
        )
        with open(save_path + save_name, 'wb') as f:
            pickle.dump(d, f) 
    print('__________________________________________________________________________________________')


