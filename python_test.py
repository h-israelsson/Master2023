from cellpose import models
from cellpose.io import imread, get_image_files, save_masks
import numpy as np
from os import makedirs
from datetime import date
import glob
from scipy import ndimage
from typing import Tuple
from tifffile import imsave
from os.path import abspath, basename, splitext
from statistics import mode

def segmentation(image_folder_path: str, model_path: str, diam: int=40,
                 save: bool=False, savedir: str=None) -> Tuple[list, str]:
    # Get the model
    model = models.CellposeModel(gpu = True, pretrained_model=model_path)
    
    # Open image files
    imgs, names = open_images(image_folder_path)

    # Segment images
    masks, flows, styles = model.eval(imgs, diameter=diam, channels = [0,0], 
                                            flow_threshold=0.4, do_3D = False)
    
    # Save masks as .tif's in folder savedir
    if save:
        _save_masks(savedir=savedir, masks=masks, names=names)

    return masks, savedir

def open_images(image_folder_path):
    files = get_image_files(image_folder_path, 'unused_mask_filter_variable')
    imgs = [imread(f) for f in files]
    names = [basename(f) for f in files]
    return imgs, names


def _save_masks(masks: list, names: list=None, savedir: str=None) -> None:
    # Create a directory where the files can be saved
    if savedir == None:
        print("This works")
        savedir = "GeneratedMasks_"+str(date.today())
    print("Savedir: " + str(savedir))
    makedirs(savedir)
    path = abspath(savedir)

    # Generate names if not given beforehand
    if names == None:
        names = [f"{i:04}" for i in range(len(masks))]
    else:
        names = [splitext(name)[0] for name in names]
    print(names)

    # Save the masks in said directory
    for (mask, name) in zip(masks, names):
        imsave(path+"\\"+name + "_cp_masks.tif", mask)
    return None


def get_no_of_roi(masks: list) -> list:
    """Get the number of ROI's in each mask"""
    # return [np.max(m) for m in masks]
    return [len(np.unique(m))-1 for m in masks] # Have to take -1 bc regions with 0 do not count as roi:s


def open_masks(folder: str) -> list:
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
        labels = range(1, number_of_roi[i]+1)
        comsi = ndimage.center_of_mass(masks[i],masks[i], labels)
        coms.append(comsi)      
    return coms, number_of_roi


def track_cells_com(masks: list, limit: int = 10, save: bool = False) -> list:
    new_masks = np.zeros_like(masks)
    new_masks[0] = masks[0]
    COMs, number_of_roi = get_centers_of_mass(masks)
    # print("COMs: " + str(COMs[0]))
    # print("COMs: " + str(COMs[1]))
    # input("Press enter to continue")

    # Loop through all masks and centers of masses.
    # print("Len(masks): " + str(len(masks)))
    for imnr in range(1, len(masks)):
        nr_of_COMs = len(COMs[imnr])
        # print("nr_of_COMs: " + str(nr_of_COMs))
        # input("Press enter to continue")
        new_cells = 0
        for comnr in range(nr_of_COMs):
            ref_image_index = -10
            for k in range(1,5):
                # Get all distances between centers of mass of one image and the one before.
                if imnr-k<0:
                    break
                distances = np.linalg.norm(np.array(COMs[imnr-k])-np.array(COMs[imnr][comnr]), axis=1)
                # print("np.array(COMs[imnr-k]): " + str(np.array(COMs[imnr-k])))
                # print("len(np.array(COMs[imnr-k])): " + str(len(np.array(COMs[imnr-k]))))
                # print("np.array(COMs[imnr][comnr]): " + str(np.array(COMs[imnr][comnr])))
                # print("Distances: " + str(distances))
                # input("Press enter to continue")
                min_distance = np.min(distances)
                # print("min_distance: " + str(min_distance))
                # input("Press enter to continue")
                # If the smallest one is smaller than the limit, exit loop
                if min_distance < limit:
                    ref_image_index = imnr-k
                    matched_cell_coord = COMs[ref_image_index][np.argmin(distances)]
                    cell_value = new_masks[ref_image_index][round(matched_cell_coord[0])][round(matched_cell_coord[1])]
                    # print("Cell value: " + str(cell_value))
                    # input("Press enter to continue")
                    break

            # If no matching cell is found in previous images:
            if ref_image_index == -10:
                print("No matching cell")
                new_cells += 1
                cell_value = np.max(new_masks[:imnr].flatten()) + new_cells
                print("Cell value: " + str(cell_value))

            # Give area in new mask value corresponding to matched cell
            roi_coords = np.argwhere(masks[imnr].flatten() == comnr)
            # print("roi_coords: " + str(roi_coords))
            # input("Press enter to continue")
            np.put(new_masks[imnr], roi_coords, cell_value)

            # to_add_to_new_masks = np.array((masks[imnr]==comnr)*cell_value//comnr)
            # new_masks[imnr] += to_add_to_new_masks

        # print(new_masks[imnr])
        # input("Press enter to continue")

    if save:
        savedir = "NewMasks_"+str(date.today())
        _save_masks(new_masks, savedir = savedir)

    return new_masks


def track_cells_overlap(masks):
    new_masks = np.zeros_like(masks)
    new_masks[0] = masks[0]
    for i in range(len(masks)-1):
        values = np.unique(masks[i+1])
        for value in values:
            roi = np.ma.masked_array(new_masks[i], mask = (masks[i+1]!=value))
            coordinates = np.argwhere(mask[i+1] == value)
            references = np.ravel_multi_index(coordinates.T, new_masks[i])
            # print("roi: " + str(roi))
            # input("Press enter to continue")
            # Find most common value in roi. If 0, complicate things. Otherwise just put this number at position in new_masks


    return None


def main():
    image_folder_path = r"\\storage3.ad.scilifelab.se\alm\BrismarGroup\Hanna\Ouabain 1st image seq\short"
    # image_folder_path = "//storage3.ad.scilifelab.se/alm/BrismarGroup/Hanna/Data_from_Emma/onehourconfluent"
    model_path = 'C:/Users/workstation3/Documents/CP_20230705_confl'

    masks = open_masks("GeneratedMasks_2023-07-07")
    # masks, savedir = segmentation(image_folder_path, model_path, save = True)

    print(get_no_of_roi(masks))

    new_masks = track_cells_overlap(masks)#, save=True)
    print(get_no_of_roi(new_masks))

if __name__ == "__main__":
    main()




# How to continue: Track cells. Change the values in each mask so that a certain cell has a certain value?
# This will be complicated. Will it take to much work?