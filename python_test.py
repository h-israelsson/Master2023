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
import matplotlib.pyplot as plt


def segmentation(image_folder_path: str, model_path: str, diam: int=40, save: bool=False, 
                 savedir: str=None) -> Tuple[list, str]:
    # Get the model
    model = models.CellposeModel(gpu = True, pretrained_model=model_path)
    # Open image files
    imgs, names = open_images(image_folder_path)
    print(imgs)
    # Segment images
    masks, flows, styles = model.eval(imgs, diameter=diam, channels = [0,0], 
                                      flow_threshold=0.4, do_3D = False)
    # Save masks as .tif's in folder savedir
    if save:
        _save_masks(savedir=savedir, masks=masks, names=names)
    return masks, savedir


def open_images(image_folder_path):
    imgs0 = imread(image_folder_path)
    imgs = [np.array(f) for f in imgs0] # Make each image separate
    # files = get_image_files(image_folder_path, 'unused_mask_filter_variable')
    # imgs = [imread(f) for f in files]
    # names = [basename(f) for f in files]
    name = basename(image_folder_path)
    return imgs, name


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
    # Save the masks in said directory
    # for (mask, name) in zip(masks, names):
    #     imsave(path+"//"+name + "_cp_masks.tif", mask)
    imsave(path+"//"+ "0000" + "_cp_masks.tif", masks) 
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
    tracked_masks = np.zeros_like(masks)
    tracked_masks[0] = masks[0]
    COMs, number_of_roi = get_centers_of_mass(masks)
    # Loop through all masks and centers of masses.
    for imnr in range(1, len(masks)):
        new_cells = 0
        for comnr in range(number_of_roi[imnr]):
            ref_image_index = -10
            for k in range(1,5):
                # Get all distances between centers of mass of one image and the one before.
                if imnr-k<0:
                    break
                distances = np.linalg.norm(np.array(COMs[imnr-k])-np.array(COMs[imnr][comnr]), axis=1)
                min_distance = np.min(distances)
                # If the smallest one is smaller than the limit, exit loop
                if min_distance < limit:
                    ref_image_index = imnr-k
                    matched_cell_coord = COMs[ref_image_index][np.argmin(distances)]
                    cell_value = tracked_masks[ref_image_index][round(matched_cell_coord[0])][round(matched_cell_coord[1])]
                    break
            # If no matching cell is found in previous images:
            if ref_image_index == -10:
                new_cells += 1
                cell_value = np.max(tracked_masks[:imnr].flatten()) + new_cells
            # Give area in new mask value corresponding to matched cell
            roi_coords = np.argwhere(masks[imnr].flatten() == comnr+1)
            np.put(tracked_masks[imnr], roi_coords, cell_value)

    if save:
        savedir = "NewMasks_"+str(date.today())
        _save_masks(tracked_masks, savedir = savedir)

    return tracked_masks


def get_cell_intensities(cell_number: int, tracked_cells: list, images: list, plot: bool=False):
    no_of_images = len(images)
    mean_intensities = np.zeros(no_of_images)
    for i in range(no_of_images):
        intensities = images[i]
        mean_intensities[i] = np.mean(intensities[tracked_cells[i] == cell_number])

    relative_intensities = (mean_intensities - np.min(mean_intensities))/(np.mean(mean_intensities)-np.min(mean_intensities))

    if plot:
        x = np.linspace(0,0+(10*no_of_images), no_of_images, endpoint=False)
        plt.figure()
        plt.plot(x, relative_intensities)
        plt.ylabel("Relative intensity")
        plt.xlabel("Time (s)")
        plt.title("Relative intensity of cell no. " + str(cell_number))
        plt.show()

    return relative_intensities


def correlation(tracked_cells, images, cell_numbers=None, all_cells=False, plot=False):
    intensities = []
    if all_cells:
        cell_numbers = range(np.max(tracked_cells))
    
    for cell_number in cell_numbers:
        intensities.append(get_cell_intensities(cell_number, tracked_cells, images, plot=False))
    corrcoefs = np.corrcoef(intensities)

    if plot:
        plt.matshow(corrcoefs)
        plt.title("Correlation coefficients")
        plt.colorbar(label="Correlation", orientation="vertical", fraction=0.046, pad=0.04)
        plt.show()

    return corrcoefs


def compare_intensities(tracked_cells, images, cell_numbers=None, all_cells=False, plot=True):
    intensities = []
    if all_cells:
        cell_numbers = range(np.max(tracked_cells))

    for cell_number in cell_numbers:
        intensities.append(get_cell_intensities(cell_number, tracked_cells, images, plot=False))

    


def main():
    # image_folder_path = r"//storage3.ad.scilifelab.se/alm/BrismarGroup/Hanna/Ouabain 1st image seq/short"
    # image_folder_path = "//storage3.ad.scilifelab.se/alm/BrismarGroup/Hanna/Master2023/2023-07-11-imaging-2/2023-07-11/Ouabain_image_stack/short"
    image_folder_path = "//storage3.ad.scilifelab.se/alm/BrismarGroup/Hanna/Master2023/2023-07-11-imaging-2/2023-07-11/CBX-ouabain-10.tif"
    # model_path = 'C:/Users/workstation3/Documents/CP_20230705_confl'
    model_path = "C:/Users/workstation3/Documents/Hanna's models/CBXoua20230719"

    # masks = open_masks("GeneratedMasks_2023-07-07")
    masks, savedir = segmentation(image_folder_path, model_path, save = True)

    # images, image_names = open_images(image_folder_path)

    # tracked_masks = track_cells_com(masks, save=False)

    # # corrcoefs = correlation(tracked_masks, images, all_cells=True, plot=True)

    # get_cell_intensities(57, tracked_masks, images, plot=True)
    # get_cell_intensities(58, tracked_masks, images, plot=True)
    # get_cell_intensities(59, tracked_masks, images, plot=True)
    # get_cell_intensities(60, tracked_masks, images, plot=True)




if __name__ == "__main__":
    main()