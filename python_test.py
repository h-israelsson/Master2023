from cellpose import models
from cellpose.io import imread, get_image_files
import numpy as np
from os import makedirs
from datetime import date
from scipy import ndimage
from typing import Tuple
from tifffile import imwrite
from os.path import basename, splitext, exists
import matplotlib.pyplot as plt

def get_segmentation(image_path, model_path, diam=40, save=False, savedir=None,
                     track=True, name=None):
    """ segment an image
    
    Takes an image stack and segments each image separately with
    the given model.
    Parameters
    ---------------
    image_path: str
        the path to the image to be segmented
    model_path: str
        the path to the model to use for segmenting
    diam: int (optional)
        approximate diameter of the cells in pixels. View the image in
        Cellpose API to get an idea of this value.
    save: bool (optional)
        if True, the masks will be saved as a single .tif file
    savedir: str (optional)
        the path to the directory in which to save the generated masks.
    track: bool (optional)
        if True, the cells are tracked using the standard variables in
        get_tracked_cells. For tuning of the tracking, use get_tracked_cells
        directly.
    name: str (optional)
        name of the .tif file to which the masks will be saved. "_masks.tif"
        is always added to the end of the name. If None, the name will be
        "[image name]_masks.tif". See _save_masks() for further details.
    Returns
    ---------------
    masks: 3D array
        the generated masks.
    """

    model = models.CellposeModel(gpu = True, pretrained_model=model_path)
    imgs = open_image_stack(image_path)
    masks, flows, styles = model.eval(imgs, diameter=diam, channels=[0,0],
                                      flow_threshold=0.4, do_3D=False)
    masks = np.array(masks)

    if name == None:
        name=splitext(basename(image_path))[0]
    if track:
        masks = get_tracked_masks(masks=masks, save=save, name=name, savedir=savedir)
    if save and not track:
        _save_masks(savedir=savedir, masks=masks,
                    name=name)
    return masks


# def open_images(image_folder_path):
#     """Opens separate images."""
#     files = get_image_files(image_folder_path, 'unused_mask_filter_variable')
#     imgs = [imread(f) for f in files]
#     # names = [basename(f) for f in files]
#     names = "Hello!"
#     return imgs, names


def open_image_stack(image_path):
    """ open a .tif image stack
    
    Modifies the data to work with the segmentation function.
    Parameters
    ---------------
    image_path: str
        the path to the image to be opened
    Returns
    ---------------
    stack: 3D array
        the image stack as a 3D array
    """
    img = imread(image_path)
    stack = [np.array(i) for i in img] # Necessary for correct segmentation
    return stack


def _save_masks(masks, name=None, savedir=None) -> None:
    """ save masks as single .tif file

    Parameters
    ---------------
    masks: 3D array
        the masks to be saved.
    name: str (optional)
        name of the file to be saved. Default name is 
        '[today's date]_masks.tif'. '_masks.tif' is always added to the name.
    savedir: str (optional)
        the directory in which to save the file. Default to save in current
        folder.
    Returns
    ---------------
        None
    """
    if name == None:
        name = str(date.today())
    if savedir == None:
        imwrite(name + "_masks.tif", masks)
    else:
        if exists(savedir) == False:
            makedirs(savedir)
        imwrite(savedir+"\\"+name+"_masks.tif", masks)
    return None


def open_masks(file_path):
    """ open a mask file
    
    This function only exists because of previous problems with reading the
    mask files. It is actally really simple and the function is probably 
    totally unnecessary.
    Parameters
    ---------------
    file_path: str
        the path to the .tif file of the segmented images (masks).
    Returns
    ---------------
    masks: 3D array
        an array with all the masks for the images.
    """
    masks = imread(file_path)
    return masks


def get_cell_labels(masks):
    """ obtain a list of cell labels

    Finds all labels in each segmented image and returns these as a 
    2D array consisting of lists with all labels for each image.
    Parameters
    ---------------
    masks: 2D or 3D array
        previously generated masks
    Returns
    ---------------
    all_labels: 1D or 2D list
        If masks are 2D, a list of all labels is returned.
        If masks are 3D, a list of this kind of lists is returned.
    """
    if len(masks.shape) == 3:
        all_labels = []
        for mask in masks:
            labels = np.unique(mask[mask!=0])
            all_labels.append(labels)
        # all_labels = np.array(all_labels)
    if len(masks.shape) == 2:
        all_labels = np.unique(masks[masks!=0])
    return all_labels


def get_centers_of_mass(masks):
    """ get centers of mass for each cell

    Calculates the coordinates for the center of mass for each cell in each
    image.
    Parameters
    ---------------
    masks: 2D or 3D array
        previously generated segmentation mask(s)
    Returns
    ---------------
    COMs: 1D or 2D list of tuples
        all coordinates of centers of mass, sorted by image if masks are 3D
    labels: 1D or 2D list
        the labels of all the cells, in the same order as the centers of mass
    """
    labels = get_cell_labels(masks)
    if len(masks.shape) == 3:
        COMs = []
        for i in range(masks.shape[0]):
            COMs_i = np.array(ndimage.center_of_mass(masks[i], masks[i], labels[i]))
            COMs.append(COMs_i)
    if len(masks.shape) == 2:
        COMs = np.array(ndimage.center_of_mass(masks, masks, labels))
    return COMs, labels


def get_tracked_masks(masks, dist_limit=20, backtrack_limit=15, random_labels=False,
                      save=False, name=None, savedir=None):
    """ track the cells

    Tracks cells and returns a list of masks, where each separate 
    cell is given the same label (integer number) in every mask.
    Parameters
    ---------------
    masks: 3D array
        previously generated segmentation of cells
    dist_limit: int (optional)
        the longest distance, in pixels, that the center of mass is allowed
        to move from one image to the next for it to still count as the same
        cell. Make bigger if cells are moving a lot.
    backtrack_limit: int (optional)
        the maximum number of images back that the algorithm will search
        through to find a center of mass within the distance limit
        (dist_limit). Make smaller if cells are moving a lot.
    random_labels: bool
        if True, the cells will be assigned random lables from the start,
        rather than keeping the labels from the first image in masks
    save: bool (optional)
        if True, the masks are saved as a single .tif file
    name: str (optional)
        name of the file to be saved. Default name is '[today's date]'.
        '_masks.tif' is always added to the name.
    savedir: str (optional)
        the directory in which to save the file. Default to save in current
        folder.
    Returns
    ---------------
    tracked_masks: 3D array
        an aray with the same masks as given as input, but updated labels to
        match between images.
    """
    tracked_masks = np.zeros_like(masks)
    COMs, roi_labels = get_centers_of_mass(masks)

    if random_labels:
        tracked_masks[0] = assign_random_cell_labels(masks[0])
    else:
        tracked_masks[0] = masks[0]

    for imnr in range(1, len(masks)):
        new_cells = 0
        ROI_labels_imnr = roi_labels[imnr]
        for COM_idx, COM_label in zip(range(len(COMs[imnr])), ROI_labels_imnr):
            ref_im_idx = -10
            for k in range(1,backtrack_limit):
                # Get all distances between centers of mass of one
                # image and the one before.
                if imnr-k<0:
                    break
                distances = np.linalg.norm(np.array(COMs[imnr-k])
                                           - np.array(COMs[imnr][COM_idx]),
                                           axis=1)
                # If the smallest one is smaller than the dist_limit, exit loop
                if np.min(distances) < dist_limit:
                    ref_im_idx = imnr-k
                    mcc = COMs[ref_im_idx][np.argmin(distances)] # matched
                                                                 # cell
                                                                 # coordinate
                    cell_label = tracked_masks[ref_im_idx][
                        round(mcc[0])][round(mcc[1])]
                    break
            # If no matching cell is found in previous images:
            if ref_im_idx == -10:
                new_cells += 1
                cell_label = (np.max(tracked_masks[:imnr].flatten())
                             + new_cells)
            # Give area in new mask value corresponding to matched cell
            roi_coords = np.argwhere(masks[imnr].flatten() == COM_label)
            np.put(tracked_masks[imnr], roi_coords, cell_label)

    if save:
        # savedir = "NewMasks_"+str(date.today())
        _save_masks(tracked_masks, name=name, savedir=savedir)

    return tracked_masks


def get_cell_intensities(cell_label, tracked_cells, images):
    """ get the realtive mean intensities for specified cell in every image
    
    Calculates the relative mean intensity for specified cell in each image.
    If the cell does not appear in one of the images, the intensity will be
    set to zero in that image.
    Parameters
    ---------------
    cell_label: int
        the label of the cell for which the intensities should be calculated
    tracked_cells: 3D array
        previously tracked masks from which to get the cell locations
    images: 3D array
        The images from which the tracked cells mask was generated and the
        intensity should be retrieved from
    Returns
    ---------------
    relative_intensities: 1D array
        mean relative intensity of the cell for each image
    """
    images_count = len(images)
    mean_intensities = np.zeros(images_count)

    for i in range(images_count):
        intensities = images[i][tracked_cells[i] == cell_label]
        if np.any(intensities):
            mean_intensities[i] = np.mean(intensities)
        else:
            mean_intensities[i] = np.nan

    relative_intensities = (mean_intensities - np.min(mean_intensities))/\
        (np.mean(mean_intensities)-np.min(mean_intensities))
    return relative_intensities


def assign_random_cell_labels(mask):
    """ assign random cell labels to mask

    Please, only provide one single mask, not a whole 3D array.
    Zero labels remain zero.
    Parameters
    ---------------
    mask: 2D array
        the mask for which the labels should be changed
    Returns
    ---------------
    randomized_mask: 2D array
        a mask with the labels shuffled around, but the general
        segmentation still remaining intact
    """
    labels = np.unique(mask[mask!=0])
    random_labels = labels.copy()
    np.random.shuffle(random_labels)
    randomized_mask = np.zeros_like(mask)
    for lbl, rnd_lbl in zip(labels, random_labels):
        lbl_coords = np.argwhere(mask.flatten() == lbl)
        np.put(randomized_mask, lbl_coords, rnd_lbl)
    return randomized_mask


def plot_cell_intensities(cell_labels, tracked_cells, images):
    """ plot relative mean intensities for specified cells
    
    Parameters
    ---------------
    cell_labels: list of ints
        list of of the labels of the cells which intensities should be plotted
    tracked_cells: 3D array
        tracked segmentation mask
    images: 3D array
        the images from which the intensities should be taken
    Returns
    ---------------
    None
    """
    period = 10 # The period between each image
    image_count = len(images)
    x = np.linspace(0,0+(period*image_count), image_count, endpoint=False)

    plt.figure()

    for c in cell_labels:
        y = get_cell_intensities(c, tracked_cells, images)
        plt.plot(x, y, label="Cell " + str(c))

    plt.ylabel("Relative intensity")
    plt.xlabel("Time [s]")
    plt.legend()
    plt.show()
    
    return None


def get_correlation_matrix(tracked_cells, images, cell_labels=None,
                           plot=True):
    """ calculate and plot correlation matrix
    
    Uses cross correlation to generate a matrix of correlations between the
    cells given by cell_labels.
    Parameters
    ---------------
    tracked_cells: 3D array
        previously tracked segmentation mask
    images: 3D array
        images from which the tracked_cells segmentation mask was generated
    cell_labels: list (optional)
        the cells for which the correlation should be calculated. Default all cells.
    plot: bool (optional)
        whether to plot the resulting matrix or not.
    Returns
    ---------------
    corrcoefs: 2D array
        the correlation coefficients as a matrix
    """
    intensities = []
    if not cell_labels:
        cell_labels = range(1, np.max(tracked_cells)+1)
    corrcoefs = np.zeros((len(cell_labels), len(cell_labels)))
    
    for cell_label, j in zip(cell_labels, range(len(cell_labels))):
        intensities.append(get_cell_intensities(cell_label, tracked_cells,
                                                images))
        for i in range(j+1):
            corrcoefs[i, j] = np.correlate(intensities[i], intensities[j])

    if plot:
        if len(cell_labels) > 100:
            idx = np.round(np.linspace(0, len(list(cell_labels)) - 1, 10,
                                       dtype='int'))
            tick_labels = [str(c) for c in idx]
        else:
            idx = cell_labels
            tick_labels = [str(c) for c in cell_labels]
        
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


def get_common_cells(tracked_masks, percentage=98):
    """ get the cells that appear in at least [percentage] number of images
    
    Parameters
    ---------------
    tracked_masks: 3D array
        previously tracked segmentation masks
    percentage: int (optional)
        the smallest percentage of images that the cell is allowed to appear
        in to still be taken into account
    Returns
    ---------------
    commons: list
        a list of labels of the cells that fullfill the given requirements of
        how many images they need to appear in
    counts: list
        a list of the number of times a specific cell appears in the images,
        in the same order as 'commons'
    """

    cell_labels = get_cell_labels(tracked_masks)
    commons = []
    counts = []
    limit = (percentage/100)*len(tracked_masks)
    cell_labels_flat = np.array([i for image in cell_labels for i in image])

    for i in np.unique(cell_labels_flat):
        count = np.count_nonzero(cell_labels_flat == i)
        if count >= limit:
            commons.append(i)
            counts.append(count)
    commons = np.array(commons)
    counts = np.array(counts)
    return commons, counts
    

def plot_cross_correlation_by_distance(ref_cell, tracked_masks, images,
                                       perc_req = 100, plot=True):
    """ plot cross correlation as a function of distance from a reference cell

    The reference cell has to appear in the first image.
    Parameters
    ---------------
    ref_cell: int
        the label of the cell that the distances should be compared with
    tracked_masks: 3D array
        previously tracked segmentation masks
    images: 3D array
        images from which the tracked_cells segmentation mask was generated
    perc_req: int
        requirement on how many percent of the images the cells have to be in
        in order to be included in the calculations of cross correlation
    plot: bool (optional)
        whether to plot the results or not
    Returns
    ---------------
    dist_sort: list
        a list of all the distances between the reference cell and the other
        cells, sorted by distance
    xcorr_list:list
        normalized cross correlation between reference cell and the other
        cells, sorted by distance from reference cell.
    """
    coms, cell_lbls = get_centers_of_mass(tracked_masks[0])
    comparison_cells, count = get_common_cells(tracked_masks, perc_req)
    coms = coms[np.isin(cell_lbls, comparison_cells)]
    com_ref = coms[comparison_cells == ref_cell]

    dists = np.linalg.norm(np.array(coms)-np.array(com_ref), axis=1)

    dists_sort, cell_lbls_sort = (list(t) for t in 
                                  zip(*sorted(zip(dists, comparison_cells))))
    xcorr_list = []
    ref_cell_intensity = get_cell_intensities(ref_cell, tracked_masks, images)

    for cell in cell_lbls_sort:
        intensity = get_cell_intensities(cell, tracked_masks, images)
        xcorr_list.append(np.corrcoef(ref_cell_intensity, intensity)[0,1])

    if plot:
        plt.figure()
        plt.plot(dists_sort, xcorr_list, '.--')
        plt.xlabel("Distance from reference cell (pixels)")
        plt.ylabel("Cross correlation")
        plt.title("Cross correlation as a function of distance from reference cell.")
        plt.show()

    return dists_sort, xcorr_list


def plot_xcorr_map(ref_cell, tracked_masks, images):
    """ plot a map of the cross correlation between ref_cell and the other cells
    in tracked_masks[0]
    
    Parameters
    ---------------
    ref_cell: int
        index of the cell to compare with
    tracked_masks: 3D array
        previously tracked segmentation masks
    images: 3D array
        images from which tracked_masks was generated
    Returns
    ---------------
    matrix: 2D array
        the matrix corresponding to an image where every cell is given the
        value ofr the cross correlation between it and the reference cell
    """
    cell_labels = get_cell_labels(tracked_masks[0])
    ref_cell_intensity = get_cell_intensities(ref_cell, tracked_masks, images)

    matrix = np.zeros_like(tracked_masks[0], dtype=float)

    for lbl in cell_labels:
        intensity = get_cell_intensities(lbl, tracked_masks, images)
        xcorr = np.corrcoef(ref_cell_intensity, intensity)[0,1]
        matrix[tracked_masks[0]==lbl] = xcorr

    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    plt.title("Correlation to cell no. " + str(ref_cell))
    fig.colorbar(cax, label="Correlation coefficient")
    plt.show()

    return matrix


def main():
    # model_path = "C:/Users/workstation3/Documents/Hannas_models/CellPoseModel-01"
    images_path = "//storage3.ad.scilifelab.se/alm/BrismarGroup/Hanna/Master2023/Recordings/2023-07-25/ouabain2.tif"
    savedir = "//storage3.ad.scilifelab.se/alm/BrismarGroup/Hanna/Master2023/Recordings/2023-07-25"
    # name = "ouabain2_generoustracking"
    # masks = get_segmentation(images_path, model_path, diam = 35, save=True, savedir=savedir, name=name)

    masks_path = "//storage3.ad.scilifelab.se/alm/BrismarGroup/Hanna/Master2023/Recordings/2023-07-25/ouabain2_btl15_distl20_masks.tif"
    # masks_path2 = "//storage3.ad.scilifelab.se/alm/BrismarGroup/Hanna/Master2023/Recordings/2023-07-25/ouabain2_generoustracking_masks.tif"
    masks = open_masks(masks_path)
    # masks2 = open_masks(masks_path2)

    images = open_image_stack(images_path)
    # tracked = get_tracked_masks(masks, name='ouabain2_btl15_distl20', save=True, savedir=savedir,
                    #   backtrack_limit=15, dist_limit=20, random_labels=False)

    # get_cross_correlation_by_distance(17, masks, images, plot=True)
    # cells, counts = get_common_cells(masks)
    # cells2, counts2 = get_common_cells(masks2)
    # print("Cells " + str(cells))
    # print("Cells2 " + str(cells2))
    # plot_cell_intensities(cells, masks, images)

    # commons, count = get_common_cells()
    # print(get_common_cells(masks, 100))

    # plot_xcorr_map(113, masks, images)
    plot_cross_correlation_by_distance(100, masks, images)

    # print(get_common_cells(masks))

if __name__ == "__main__":
    main()