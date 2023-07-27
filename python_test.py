from cellpose import models
from cellpose.io import imread, get_image_files
import numpy as np
from os import makedirs
from datetime import date
import glob
from scipy import ndimage
from typing import Tuple
from tifffile import imwrite
from os.path import abspath, basename, splitext, exists
import matplotlib.pyplot as plt


def get_segmentation(image_path: str, model_path: str, diam: int=40,
                     save: bool=False, savedir: str=None,
                     track: bool=True) -> list:
    """Takes an image and segments it with the given model. 
    Returns segmentation mask."""
    model = models.CellposeModel(gpu = True, pretrained_model=model_path)
    imgs = open_image_stack(image_path)   # Change to open_images if
                                                # images are separate and not
                                                # in a stack
    # imgs, name = open_images(image_path)
    masks, flows, styles = model.eval(imgs, diameter=diam, channels = [0,0],
                                      flow_threshold=0.4, do_3D = False)
    masks = np.array(masks)
    
    name=splitext(basename(image_path))[0]

    if track:
        masks = get_tracked_masks(masks=masks, save=save, name=name, savedir=savedir)

    if save and not track:
        _save_masks(savedir=savedir, masks=masks,
                    name=name)
    return masks


def open_images(image_folder_path):
    """Opens separate images."""
    files = get_image_files(image_folder_path, 'unused_mask_filter_variable')
    imgs = [imread(f) for f in files]
    # names = [basename(f) for f in files]
    names = "Hello!"
    return imgs, names


def open_image_stack(image_path: str):
    """Opens a tiff file."""
    img = imread(image_path)
    stack = [np.array(i) for i in img] # Necessary for correct segmentation
    return stack


def _save_masks(masks: list, name: str=None, savedir: str=None) -> None:
    """Saves masks as single tif file."""
    if name == None:
        name = str(date.today())
    if savedir == None:
        imwrite(name + "_masks.tif", masks)
    else:
        if exists(savedir) == False:
            makedirs(savedir)
        imwrite(savedir+"\\"+name+"_masks.tif", masks)
    return None


def get_roi_count(masks):
    """Returns the number of ROI's in each image (mask) as a list."""
    if len(masks.shape) == 3:
        roi_count = [len(np.unique(m))-1 for m in masks] # Have to take -1
                                                         # bc regions with
                                                         # 0 do not count as
                                                         # roi:s
    if len(masks.shape) == 2:
        roi_count = len(np.unique(masks))-1
    return roi_count


# def open_masks(folder: str) -> list:
#     """Load ready-made masks from specified folder."""
#     file_names = glob.glob(folder + '/*_masks.tif')
#     masks = [imread(mask) for mask in file_names]
#     return masks


def open_masks(file_path):
    return imread(file_path)


def get_centers_of_mass(masks: list) -> Tuple[list, list]:
    """Returns a list with coordinates for centers of mass for each
    cell in each image."""
    coms = []
    roi_count = get_roi_count(masks)
    if len(masks.shape) == 3:
        for i in range(masks.shape[0]):
            labels = range(1, roi_count[i]+1)
            comsi = ndimage.center_of_mass(masks[i],masks[i], labels)
            coms.append(comsi)
    if len(masks.shape) == 2:
        labels = range(1, roi_count+1)
        coms = ndimage.center_of_mass(masks,masks, labels)
    return coms, roi_count


def get_tracked_masks(masks: list, limit: int = 10, name: str=None,
                      save: bool = False, savedir: str=None) -> list:
    """Tracks cells and returns a list of masks where each cell is given
    the same number in every mask."""
    tracked_masks = np.zeros_like(masks)
    tracked_masks[0] = masks[0]
    COMs, roi_count = get_centers_of_mass(masks)
    backtrack_limit = 5
    # Loop through all masks and centers of masses.
    for imnr in range(1, len(masks)):
        new_cells = 0
        for comnr in range(roi_count[imnr]):
            ref_im_idx = -10
            for k in range(1,backtrack_limit):
                # Get all distances between centers of mass of one
                # image and the one before.
                if imnr-k<0:
                    break
                distances = np.linalg.norm(np.array(COMs[imnr-k])
                                           - np.array(COMs[imnr][comnr]),
                                           axis=1)
                min_distance = np.min(distances)
                # If the smallest one is smaller than the limit, exit loop
                if min_distance < limit:
                    ref_im_idx = imnr-k
                    mcc = COMs[ref_im_idx][np.argmin(distances)] # matched
                                                                 # cell
                                                                 # coordinate
                    cell_value = tracked_masks[ref_im_idx][
                        round(mcc[0])][
                            round(mcc[1])]
                    break
            # If no matching cell is found in previous images:
            if ref_im_idx == -10:
                new_cells += 1
                cell_value = (np.max(tracked_masks[:imnr].flatten())
                             + new_cells)
            # Give area in new mask value corresponding to matched cell
            roi_coords = np.argwhere(masks[imnr].flatten() == comnr+1)
            np.put(tracked_masks[imnr], roi_coords, cell_value)

    if save:
        # savedir = "NewMasks_"+str(date.today())
        _save_masks(tracked_masks, name=name, savedir=savedir)

    return tracked_masks


def get_cell_intensities(cell_number: int, tracked_cells: list, images: list):
    """Get the mean intensities of a specified cells across all images."""
    images_count = len(images)
    mean_intensities = np.zeros(images_count)
    for i in range(images_count):
        intensities = images[i]
        mean_intensities[i] = np.mean(intensities[tracked_cells[i]
                                                  == cell_number])
    relative_intensities = (mean_intensities - np.min(mean_intensities))/\
        (np.mean(mean_intensities)-np.min(mean_intensities))
    return relative_intensities


def plot_cell_intensities(cell_numbers: list, tracked_cells: list,
                          images: list):
    """Plot mean intensities for specified cell numbers."""
    image_count = len(images)
    x = np.linspace(0,0+(10*image_count), image_count, endpoint=False)
    plt.figure()
    plt.ylabel("Relative intensity")
    plt.xlabel("Time [s]")
    for c in cell_numbers:
        y = get_cell_intensities(c, tracked_cells, images)
        plt.plot(x, y, label="Cell " + str(c))
    plt.legend()
    plt.show()
    
    return None


def get_correlation_matrix(tracked_cells, images, cell_numbers=None, all_cells=False,
                plot=True):
    """Calculate correlation of the intensity between all cells
    in the images."""
    intensities = []
    if all_cells:
        cell_numbers = range(1, np.max(tracked_cells)+1)
    corrcoefs = np.zeros((len(cell_numbers), len(cell_numbers)))
    
    for cell_number, j in zip(cell_numbers, range(len(cell_numbers))):
        intensities.append(get_cell_intensities(cell_number, tracked_cells,
                                                images))
        for i in range(j+1):
            corrcoefs[i, j] = np.correlate(intensities[i], intensities[j])

    if plot:
        if all_cells:
            idx = np.round(np.linspace(0, len(list(cell_numbers)) - 1, 10,
                                       dtype='int'))
            tick_labels = [str(c) for c in idx]
        else:
            idx = cell_numbers
            tick_labels = [str(c) for c in cell_numbers]
        print(tick_labels)
        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(corrcoefs)
        plt.title("Correlation coefficients")
        fig.colorbar(cax, label="Correlation", orientation="vertical")#, fraction=0.046, pad=0.04)
        ax.set_xticks(idx)
        ax.set_yticks(idx)
        ax.set_xticklabels(tick_labels)
        ax.set_yticklabels(tick_labels)
        plt.show()

    return corrcoefs


def get_common_cells(tracked_masks,
                     percentage=100 # The percentage of images the cell must
                                    # be in to be counted.
                     ):
    """Returns the cell numbers that are common for all images,
    i.e. cells that never disappear."""
    cell_numbers = []
    commons = []
    counts = []
    for i in range(len(tracked_masks)):
        cell_numbers.append(np.unique(tracked_masks[i]))
    limit = (percentage/100)*len(tracked_masks)
    for i in np.unique(cell_numbers.flatten()):
        count = np.count_nonzero(cell_numbers = i)
        if i == 0:
            continue
        if count >= limit:
            commons.append(i)
            counts.append(count)
    return commons

    # numbers_lists = []
    # for image in tracked_masks:
    #     numbers_lists.append(np.unique(image))
    # commons = numbers_lists[0]
    # for i in range(1,len(numbers_lists)):
    #     commons = np.intersect1d(commons, numbers_lists[i])
    # common_cells = commons[1:]  # First input is the number 0,
    #                             # which is not a cell
    # return common_cells
    

def get_cross_correlation_by_distance(ref_cell: int, tracked_masks,
                                      images, plot=True):
    """Get cross correlation as a function of distance from a specified cell.
    Only uses cells that are common for all images."""
    cell_numbers = get_common_cells(tracked_masks)
    coms, roi_count = get_centers_of_mass(tracked_masks[0])
    com_ref = coms[ref_cell-1]
    distances = np.linalg.norm(np.array(coms)-np.array(com_ref), axis=1)
    dist_dict = dict(zip(cell_numbers, distances[cell_numbers-1]))
    sorted_dist_dict = dict(sorted(dist_dict.items(),
                                   key=lambda item: item[1]))

    dist_list = []
    cross_correlation_list = []
    intensity_ref = get_cell_intensities(ref_cell, tracked_masks, images)

    for c in sorted_dist_dict:
        dist_list.append(sorted_dist_dict[c])
        intensity_c = get_cell_intensities(c, tracked_masks, images)
        cross_correlation_list.append(np.corrcoef(intensity_ref,
                                                  intensity_c)[0,1]) # Have to
                        # think through whether to use corrcoef or correlate

    if plot:
        plt.figure()
        plt.plot(dist_list, cross_correlation_list)
        plt.xlabel("Distance from reference cell (pixels)")
        plt.ylabel("Cross correlation")
        plt.title("Cross correlation as a function of distance from reference cell.")
        plt.show()

    return dist_list, cross_correlation_list


def main():
    model_path = "C:/Users/workstation3/Documents/Hannas_models/CellPoseModel-01"
    images_path = "//storage3.ad.scilifelab.se/alm/BrismarGroup/Hanna/Master2023/Recordings/2023-07-25/ouabain2.tif"
    masks = get_segmentation(images_path, model_path, diam = 35, save=True, savedir="//storage3.ad.scilifelab.se/alm/BrismarGroup/Hanna/Master2023/Recordings/2023-07-25")

    # masks_path = "//storage3.ad.scilifelab.se/alm/BrismarGroup/Hanna/Master2023/Recordings/2023-07-25/control_masks.tif"
    # masks = open_masks(masks_path)
    images = open_image_stack(images_path)

    # get_cross_correlation_by_distance(17, masks, images, plot=True)
    cells = get_common_cells(masks)
    plot_cell_intensities(cells, masks, images)

    # print(get_common_cells(masks))

if __name__ == "__main__":
    main()