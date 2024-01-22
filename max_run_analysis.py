import tifffile
import python_test
import pandas as pd
import numpy as np
# import skimage
from scipy import ndimage
from cellpose import models
import matplotlib.pyplot as plt
import os

def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    else:
        print(f"Folder '{folder_path}' already exists.")

def do_it(X, masks, occurrence_limit=50, T=10, max_dt = 30):

    # load masks
    print('Get intensities')
    common_cells, counts = python_test.get_common_cells(masks, occurrence=occurrence_limit) # Get the cells to include in the intensity measurements

    intensities = {}
    for c in common_cells:
        i = python_test.get_cell_intensities(c, masks, X, T=T)#.astype('int16')
        if any(np.isnan(i)):
            continue
        else:
            intensities[c] = i

    intensities = pd.DataFrame(intensities)
    print('Get COM')
    COMs = {}
    for c in common_cells:
        for im in range(len(masks)):
            if np.any(masks[im]==c):
                COMs[c] = list(python_test.get_centers_of_mass(masks[im], c)[0])
                break

    COMs = pd.DataFrame([COMs],["Center of mass"])

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
    print('Get border values')
    fltr_masks = filtered_masks(masks, common_cells)
    border_values = {}
    for c in common_cells:
        border_values[c] = list(extract_border_values(fltr_masks, c))

    border_values = pd.DataFrame([border_values],["Border values"])
    
    # get cross correlation
    def get_ncc_for_bv(border_values,intensities, cell_lbl, max_dt):
        bvs = np.array(border_values.loc["Border values", cell_lbl])
        bvs = bvs[bvs!=0]
        bvs_uniq = np.unique(bvs)
        ncc = 0
        dts = 0
        for bv in bvs_uniq:
            xcorr, dt = python_test.get_cc(np.array(intensities.loc[:, cell_lbl]),
                            np.array(intensities.loc[:, bv]), max_dt)
            w = len(bvs[bvs==bv])/len(bvs[bvs!=0])
            ncc += np.max(xcorr)*w
            dts += dt[np.argmax(xcorr)]*w
        return ncc, dts
    print('Do Crossvalidation')
    cross_correlations = {}
    time_diffs = {}
    for c in common_cells:
        cross_correlations[c], time_diffs[c] = get_ncc_for_bv(border_values, intensities, c, max_dt)

    cross_correlations = pd.DataFrame([cross_correlations], ["Weighted max NCC for border values"])
    time_diffs = pd.DataFrame([time_diffs], ["Weighted time difference at max NCC for border values"])
    
    def get_max_freq(intensities, T):
        fouriermax = {}
        for c in intensities.columns:
            fft = np.fft.fft(intensities.loc[:, c])
            freqs = np.fft.fftfreq(len(intensities.loc[:, c]), T)
            fft_filtered = fft*(freqs>(3/(3600))) #Remove all periods longer than 20 min  
            fouriermax[c] = np.abs(freqs[np.argmax(fft_filtered)])
        return fouriermax

    fouriermax = get_max_freq(intensities, T)

    fouriermax = pd.DataFrame([fouriermax], ["Most prominent frequency"])
    df = pd.concat([pd.concat([COMs, border_values, cross_correlations, time_diffs, fouriermax]), intensities], keys=["Analysis results", "Intensities"])
    print('Save file')
    return df

# __________________________________________________________________________________________


force_redo = False
masks_path = '//storage3.ad.scilifelab.se/alm/BrismarGroup/Hanna/240115_correlation_data/masks_new_trained_scratch/'
tif_path = "//storage3.ad.scilifelab.se/alm/BrismarGroup/Hanna/Alejandro's recordings - tif/"
save_path = '//storage3.ad.scilifelab.se/alm/BrismarGroup/Hanna/240115_correlation_data/csvs_new2_trained_from_scratch/'

all_masks = os.listdir(masks_path)
create_folder(save_path)

# def do_it(X, masks, occurrence_limit=50, T=10, max_dt = 30):
for f in all_masks:
    save_name = f[:-4] + '.csv'
    if save_name in os.listdir(save_path) and force_redo == False:
        print('CSV for ' + f + ' already exists.')
        continue
    else:
        print(f)
        X = tifffile.imread(tif_path + f)
        masks = tifffile.imread(masks_path + f)
        index = masks.shape[0]
        X = X[:index,:,:]
        df = do_it(
            X,
            masks,
        )
        df.to_csv(save_path + f[:-4] + '.csv')
    print('__________________________________________________________________________________________')


