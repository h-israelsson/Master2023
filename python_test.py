from cellpose import models
from cellpose.io import imread, masks_flows_to_seg, get_image_files, save_masks
import numpy as np
import matplotlib.pyplot as plt
from os import makedirs
from datetime import date
import glob

def segmentation(image_folder_path, model_path, diam = 40, save = False, savedir=None):
    """Savedir is the directory in which to save the masks"""

    # Get the model
    model = models.CellposeModel(gpu = True, pretrained_model=model_path)
    # Open image files
    files = get_image_files(image_folder_path, 'unused_mask_filter_variable')
    imgs = [imread(f) for f in files]
    names = [str(f) for f in files]
    # Segment images
    masks, flows, styles = model.eval(imgs, diameter=diam, channels = [0,0], 
                                            flow_threshold=0.4, do_3D = False)
    # Save masks as .tif's in folder savedir
    if save:
        _save(savedir, imgs, masks, flows, names)
    return masks, savedir


def _save(savedir, imgs, masks, flows, names):
    # Create a directory where the files can be saved
    if savedir == None:
        savedir = "GeneratedMasks_"+str(date.today())
    makedirs(savedir)
    # Save the masks in said directory
    save_masks(imgs, masks, flows, names, png=False, tif=True, 
               savedir=savedir, save_txt=False)
    return None


def get_no_of_roi(masks):
    """Get the number of RIO's in each mask"""
    return [np.max(m) for m in masks]


def get_mask_files(folder):
    """Load ready-made masks from specified folder."""
    file_names = glob.glob(folder + '/*_masks.tif')
    masks = [imread(mask) for mask in file_names]
    return masks


def main():
    image_folder_path = "//storage3.ad.scilifelab.se/alm/BrismarGroup/Hanna/Data_from_Emma/Confluent_images/short"
    # image_folder_path = "//storage3.ad.scilifelab.se/alm/BrismarGroup/Hanna/Data_from_Emma/onehourconfluent"
    model_path = 'C:/Users/workstation3/Documents/Hanna/Master2023/CP_20230705_confl'

    # masks, savedir = segmentation(image_folder_path, model_path, save = False)
    masks = get_mask_files('GeneratedMasks_2023-07-07')

    print(masks)
    print(type(masks))
    print(len(masks))

    print(get_no_of_roi(masks))

if __name__ == "__main__":
    main()




# How to continue: Track cells. Change the values in each mask so that a certain cell has a certain value?
# This will be complicated. Will it take to much work?