from cellpose import models
from cellpose.io import imread, get_image_files, save_masks
import numpy as np
from os import makedirs
from datetime import date
import glob
from scipy import ndimage
from typing import Tuple

def segmentation(image_folder_path: str, model_path: str, diam: int=40,
                 save: bool=False, savedir: str='') -> Tuple[list, str]:
    # Get the model
    model = models.CellposeModel(gpu = True, pretrained_model=model_path)
    
    # Open image files
    imgs, names = open_images(image_folder_path)

    # Segment images
    masks, flows, styles = model.eval(imgs, diameter=diam, channels = [0,0], 
                                            flow_threshold=0.4, do_3D = False)
    
    # Save masks as .tif's in folder savedir
    if save:
        _save(savedir, imgs, masks, flows, names)

    return masks, savedir

def open_images(image_folder_path):
    files = get_image_files(image_folder_path, 'unused_mask_filter_variable')
    imgs = [imread(f) for f in files]
    names = [str(f) for f in files]
    return imgs, names


def _save(masks: list, imgs: list=[], flows: list=[], names: list=[], savedir: str='') -> None:
    # Create a directory where the files can be saved
    if savedir == '':
        savedir = "GeneratedMasks_"+str(date.today())
    makedirs(savedir)

    if names == []:
        names = ["%03d" % i for i in range(len(imgs))]

    # Save the masks in said directory
    save_masks(imgs, masks, flows, names, png=False, tif=True, 
               savedir=savedir, save_txt=False)
    return None


def get_no_of_roi(masks: list) -> list:
    """Get the number of RIO's in each mask"""
    return [np.max(m) for m in masks]


def get_mask_files(folder: str) -> list:
    """Load ready-made masks from specified folder."""
    file_names = glob.glob(folder + '/*_masks.tif')
    masks = [imread(mask) for mask in file_names]
    return masks


def get_centers_of_mass(masks: list) -> Tuple[list, list]:
    """Returns a list with coordinates for centers of mass for each cell in each image
    on the form [[(coordinate of c1, im1), (coordinates)]]"""
    coms = []
    number_of_roi = get_no_of_roi(masks)
    for i in range(len(masks)):
        labels = range(1, number_of_roi[i] + 1)
        comsi = ndimage.center_of_mass(masks[i],masks[i], labels)
        coms.append(comsi)      
    return coms, number_of_roi


def track_cells(masks, limit = 5, save=False):
    new_masks = np.zeros_like(masks)
    new_masks[0] = masks[0]
    COMs, number_of_roi = get_centers_of_mass(masks)

    # Loop through all masks and centers of masses.
    for imnr in range(1, len(masks)):
        for comnr in range(len(COMs[i])):
            ref_image_index = 0
            for k in range(1,5):
                # Get all distances between centers of mass of one image and the one before.
                distances = np.linalg.norm(COMs[imnr-k]-COMs[imnr][comnr])
                min_distance = np.min(distances)
                # If the smallest one is smaller than the limit, exit loop
                if min_distance < limit:
                    min_index = np.argmin(distances)
                    ref_image_index = imnr-k
                    break

            # If no matching cell is found in previous images:
            if ref_image_index == 0:
                min_index = max(number_of_roi[:imnr]+1)

            # Give area in new mask value corresponding to matched cell
            new_masks[imnr] += (masks[imnr] == comnr)*min_index//comnr

    if save:
        _save(new_masks, savedir="NewMasks_"+str(date.today))

    return new_masks


def main():
    image_folder_path = r"\\storage3.ad.scilifelab.se\alm\BrismarGroup\Hanna\Ouabain 1st image seq\short"
    # image_folder_path = "//storage3.ad.scilifelab.se/alm/BrismarGroup/Hanna/Data_from_Emma/onehourconfluent"
    model_path = 'C:/Users/workstation3/Documents/CP_20230705_confl'

    masks, savedir = segmentation(image_folder_path, model_path, save = False)

if __name__ == "__main__":
    main()




# How to continue: Track cells. Change the values in each mask so that a certain cell has a certain value?
# This will be complicated. Will it take to much work?