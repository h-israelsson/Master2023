from cellpose import models
from cellpose.io import imread, masks_flows_to_seg, get_image_files, save_masks
import numpy as np
import matplotlib.pyplot as plt
from os import makedirs
from datetime import date

def segmentation(image_folder_path, model_path, diam = 40, save = False, savedir=None):
    # dir_path is the path to the 
    model = models.CellposeModel(gpu = True, pretrained_model=model_path)

    files = get_image_files(image_folder_path, 'stupid_variable')
    imgs = [imread(f) for f in files]

    masks, flows, styles = model.eval(imgs, diameter=diam, channels = [0,0], 
                                            flow_threshold=0.4, do_3D = False)

    if save:
        # Create a directory where the files can be saved
        if savedir == None:
            savedir = "GeneratedMasks_"+str(date.today())
        makedirs(savedir)

        # Create an array of names for the files.
        names = np.arange(len(imgs)).astype(str)
        names = np.char.zfill(names, 4)

        save_masks(imgs, masks, flows, "GeneratedMasks", png=False, tif=True, 
                   savedir=savedir, save_txt=False)
    return masks



def main():
    image_folder_path = "//storage3.ad.scilifelab.se/alm/BrismarGroup/Hanna/Data_from_Emma/Confluent_images/short"
    # image_folder_path = "//storage3.ad.scilifelab.se/alm/BrismarGroup/Hanna/Data_from_Emma/onehourconfluent"
    model_path = 'C:/Users/workstation3/Documents/CP_20230705_confl'

    segmentation(image_folder_path, model_path, save = True)

if __name__ == "__main__":
    main()