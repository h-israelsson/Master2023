from cellpose import models
from cellpose.io import imread
import numpy as np
from os import makedirs
from datetime import date
from scipy import ndimage
from tifffile import imwrite
from os.path import basename, splitext, exists
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.signal import butter, filtfilt, convolve2d

def get_segmentation(image_path, model_path, diam=40, save=False, savedir=None,
                     track=True, name=None):
    """ segment an image
    
    Takes an image stack and segments each image separately with
    the given model.
    Parameters
    ---------------
    image_path: str or 3D array
        the path to the image to be segmented or the 3D array of the image.
        The 3D array should follow the structure given from open_image_stack
        in order to be segmented correctly.
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
        the generated masks, where the area of each cell has a unique integer
        value (the cell label) greater than or eaqual to one and the
        background has the value 0.
    """

    model = models.CellposeModel(gpu = True, pretrained_model=model_path)
    if type(image_path) == str:
        imgs = open_image_stack(image_path)
    else:
        imgs = image_path
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


def _save_masks(masks, name=None, savedir=None):
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


def extract_images_for_training(images_path, interval=50, savedir=None,
                                name=None):
    """Extracts images with a certain interval from image stack.
    
    Parameters
    ---------------
    images_path: str
        the path to the image stack to extract the images from
    interval: int (optional)
        interval between the images to be extracted
    savedir: str (optional)
        directory in which to save the extracted images. Default to save
        the images in current directory.
    name: str (optional)
        name of the files. To the name, the image number will be added.
        Default to save files with the name of the image stack file,
        followed by the image number, e.g. ctl0035. 
    Returns
    ---------------
        None
    """

    stack = open_image_stack(images_path)
    im_idxs = np.arange(0,len(stack),interval)
    if name == None:
        name=splitext(basename(images_path))[0]
    for i in im_idxs:
        if savedir == None:
            imwrite(name + str("{0:03}".format(i)) + ".tif", stack[i])
        else:
            if exists(savedir) == False:
                makedirs(savedir)
            imwrite(savedir + "\\" + name + str("{0:03}".format(i)) + ".tif",
                    stack[i])
        
    return None


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


def get_cell_intensities(cell_label, tracked_cells, images, normalize=True, hpf=False, lpf=False):
    """ get the realtive mean intensities for specified cell in every image
    
    Calculates the relative mean intensity for specified cell in each image.
    If the cell does not appear in one of the images, the intensity will be
    set to the same value as previous image.
    Parameters
    ---------------
    cell_label: int
        the label of the cell for which the intensities should be calculated
    tracked_cells: 3D array
        previously tracked masks from which to get the cell locations
    images: 3D array
        The images from which the tracked cells mask was generated and the
        intensity should be retrieved from
    normalize: bool (optional)
        if True, the intensities are normalized so that the intensities of
        different cells match better
    hpf: bool (optional)
        whether to do high-pass filtering or not
    lpf: bool (optional)
        whether to do low-pass filtering or not
    Returns
    ---------------
    mean_intensities: 1D array
        mean (normalized) intensity of the cell for each image
    """
    images_count = len(images)
    mean_intensities = np.zeros(images_count)
    hpf_cutoff_freq = 0.0025
    lpf_cutoff_freq = 0.005
    T = 10                  # period of sampling

    for i in range(images_count):
        intensities = images[i][tracked_cells[i] == cell_label]
        if np.any(intensities):
            mean_intensities[i] = np.mean(intensities)
        else:
            mean_intensities[i] = mean_intensities[i-1]
    if hpf:
        # freqs = np.fft.fftfreq(len(mean_intensities), T)
        # filter_mask = np.abs(freqs) > hpf_cutoff_freq
        # intensities_fft = np.fft.fft(mean_intensities)
        # filtered_signal = np.real(np.fft.ifft(intensities_fft*filter_mask))
        # mean_intensities = filtered_signal
        a, b = butter(3, 0.025, 'highpass')
        mean_intensities = filtfilt(a, b, mean_intensities)
    if lpf:
        freqs = np.fft.fftfreq(len(mean_intensities), T)
        filter_mask = np.abs(freqs) < lpf_cutoff_freq
        intensities_fft = np.fft.fft(mean_intensities)
        filtered_signal = np.real(np.fft.ifft(intensities_fft*filter_mask))
        mean_intensities = filtered_signal
    if normalize:
        mean_intensities = (mean_intensities - np.min(mean_intensities))/\
            (np.mean(mean_intensities)-np.min(mean_intensities))
    return mean_intensities


def assign_random_cell_labels(mask):
    """ assign random cell labels to mask. I didn't find much use in this.

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


def plot_cell_intensities(cell_labels, tracked_cells, images, normalize=True, hpf=False, lpf=False):
    """ plot relative mean intensities for specified cells
    
    Parameters
    ---------------
    cell_labels: list of ints
        list of of the labels of the cells which intensities should be plotted
    tracked_cells: 3D array
        tracked segmentation mask
    images: 3D array
        the images from which the intensities should be taken
    normalize: bool (optional)
        if True, the intensities are normalized so that the intensities of
        different cells match better
    hpf: bool (optional)
        whether to do high-pass filtering or not
    lpf: bool (optional)
        whether to do low-pass filtering or not
    Returns
    ---------------
    None
    """
    T = 10 # The period between each image
    image_count = len(images)
    x = np.linspace(0,0+(T*image_count), image_count, endpoint=False)

    plt.figure()

    for c in cell_labels:
        y = get_cell_intensities(c, tracked_cells, images, normalize, hpf, lpf)
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


def get_common_cells(tracked_masks, occurrence=100):
    """ get the cells that appear in at least [occurrence] percent of images
    
    Parameters
    ---------------
    tracked_masks: 3D array
        previously tracked segmentation masks
    occurrence: int (optional)
        the smallest percent of images that the cell is allowed to appear
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
    limit = (occurrence/100)*len(tracked_masks)
    cell_labels_flat = np.array([i for image in cell_labels for i in image])

    for i in np.unique(cell_labels_flat):
        count = np.count_nonzero(cell_labels_flat == i)
        if count >= limit:
            commons.append(i)
            counts.append(count)
    commons = np.array(commons)
    counts = np.array(counts)
    return commons, counts
    

def plot_xcorr_vs_distance(ref_cell, tracked_masks, images, perc_req = 100,
                           normalize=False, hpf=False, lpf=False, plot=True):
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
    normalize: bool (optional)
        if True, the intensities are normalized so that the intensities of
        different cells match better
    hpf: bool (optional)
        whether to do high-pass filtering or not
    lpf: bool (optional)
        whether to do low-pass filtering or not
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
    ref_cell_intensity = get_cell_intensities(ref_cell, tracked_masks, images,
                                              normalize, hpf, lpf)

    for cell in cell_lbls_sort:
        intensity = get_cell_intensities(cell, tracked_masks, images)
        xcorr_list.append(np.corrcoef(ref_cell_intensity, intensity)[0,1])

    if plot:
        plt.figure()
        plt.plot(dists_sort, xcorr_list, '.')
        plt.xlabel("Distance from reference cell (pixels)")
        plt.ylabel("Cross correlation")
        plt.title("Cross correlation as a function of distance from cell " + str(ref_cell))
        plt.show()

    return dists_sort, xcorr_list


def plot_xcorr_map(tracked_masks, images, mode='single', ref_cell=1, occurrence=97, normalize=False,
                   hpf=False, lpf=False, show_labels=False, diam=30):
    """ plot a map of the cross correlation per cell
    
    Parameters
    ---------------
    tracked_masks: 3D array
        previously tracked segmentation masks

    images: 3D array
        images from which tracked_masks was generated

    mode: {'single', 'total_sum', 'nearest_neighbor'} (optional)
        'single':
            By default, mode is 'single'. This returns a map with the
            cross correlations between a reference cell (ref_cell) and every
            other cell. With this mode, ref_cell must be given.
        'total_sum':
            Mode 'total_sum' returns a map with the summed cross correlation
            with all cells for each cell.
        'nearest_neighbor':
            Mode 'nearest_neighbor' returns a map where each cell gets a value
            corresponding to the weighted sum of the cross correlations with
            the cells sharing a border with it.

    ref_cell: int (optional)
        index of the reference cell when using mode 'single'.

    occurrence: int (optional)
        the percentage of images a cell has to appear in in order to be used
        in the cross correlation.

    normalize: bool (optional)
        if True, the intensities are normalized so that the intensities of
        different cells match better

    hpf: bool (optional)
        whether to do high-pass filtering or not

    lpf: bool (optional)
        whether to do low-pass filtering or not

    show_labels: bool (optional)
        whether to show the labels of the cells or not

    Returns
    ---------------
    matrix: 2D array
        the first mask from tracked_masks, but the values of the cells
        have been exchanged for the corresponding correlation coefficients.
    """

    ref_image = tracked_masks[10]
    cell_labels, xxx = get_common_cells(tracked_masks, occurrence)
    matrix = np.zeros_like(tracked_masks[0], dtype=float)
    correlation_mean = 0
    correlation_variance = 0

    if mode=="total_sum":
        intensities = []
        for lbl in cell_labels:
            intensities.append(get_cell_intensities(lbl, tracked_masks,
                                                    images, normalize, hpf,
                                                    lpf))
        xcorr = np.corrcoef(intensities)
        for i, lbl in enumerate(cell_labels):
            matrix[ref_image==lbl] = float(np.sum(xcorr[i]))
        matrix /= len(cell_labels)

    if mode=="single":
        ref_cell_intensity = get_cell_intensities(ref_cell, tracked_masks,
                                                  images, normalize, hpf, lpf)
        for lbl in cell_labels:
            intensity = get_cell_intensities(lbl, tracked_masks, images)
            xcorr = np.corrcoef(ref_cell_intensity, intensity)[0,1]
            matrix[ref_image==lbl] = xcorr

    if mode=="nearest_neighbor":        # CONTROL SOMEHOW IF THIS ALL ACTUALLY WORKS!
        ones = np.ones([3,3])
        intensities = []
        correlations_list = []
        for lbl in cell_labels:
            intensities.append(get_cell_intensities(lbl, tracked_masks,
                                                    images, normalize, hpf,
                                                    lpf))
        xcorr_matrix = np.corrcoef(intensities)
        print(np.shape(xcorr_matrix))
        for lbl in cell_labels:
            mask = convolve2d(ref_image==lbl, ones, mode="same")
            all_border_values = ref_image[mask!=0]
            all_border_values = all_border_values[all_border_values!=lbl]
            # We only want to take the elements in cell_labels into account
            border_values = all_border_values[np.in1d(all_border_values,
                                                      cell_labels)]
            bv_count = len(border_values)
            correlations = 0
            for i in np.unique(border_values):
                weight = float(np.count_nonzero(border_values==i))/\
                    float(bv_count)
                weighted_corr = weight*float(xcorr_matrix[cell_labels==lbl,
                                                          cell_labels==i])
                correlations += weighted_corr
            matrix[ref_image==lbl] = correlations
            if (len(all_border_values)-bv_count)/len(all_border_values) < 0.3:
                correlations_list.append(correlations)
        correlation_mean = np.mean(np.array(correlations_list))
        correlation_variance = np.var(np.array(correlations_list))
        


    # Get the average correlation of all cells, not including cells at border
    ones = np.ones([3,3])
    non_edge_labels = []
    for lbl in cell_labels:
        mask = convolve2d(ref_image==lbl, ones, mode="same")
        border_values = ref_image[mask!=0]
        border_values = border_values[border_values!=lbl]
        if len(border_values==0) < 0.3 len(border_values):
            non_edge_labels.append(lbl)
    


    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(111)
    masked_matrix = np.ma.masked_where(matrix == 0, matrix)
    cmap = cm.hot   # This doesn't seem to do anything
    cmap.set_bad(color='white')
    cax = ax.imshow(masked_matrix)
    if mode=='total_sum':
        plt.title("Summed correlation with all cells")
        fig.colorbar(cax, label="Sum of correlation coefficients")
    if mode=='single':
        plt.title("Correlation to cell no. " + str(ref_cell))
        fig.colorbar(cax, label="Correlation coefficient")
    if mode=='nearest_neighbor':
        plt.title("Correlation to nearest neighboring cells")
        fig.colorbar(cax, label="Correlation coefficient")

    # # Add annotation to all the cells in the image
    if show_labels:
        coms, lbls = get_centers_of_mass(ref_image)
        coms_commons = np.zeros((len(cell_labels),2))
        for i, lbl in enumerate(cell_labels):
            coms_commons[i] = coms[np.where(lbls==lbl)][0]
        y = coms_commons[:, 0]
        x = coms_commons[:, 1]
        plt.scatter(x,y, marker='.', color="red")
        for i, lbl in enumerate(cell_labels):
            ax.annotate(lbl, (x[i], y[i]))

    plt.show()
    return matrix, correlation_mean, correlation_variance
