from cellpose import models
from cellpose.io import imread, masks_flows_to_seg, get_image_files
import numpy as np
import matplotlib.pyplot as plt
from cellpose import plot


def segmentation(image_folder_path, model_path, diam = 40, save_masks = False):

    model = models.CellposeModel(gpu = True, pretrained_model=model_path)

    files = get_image_files(image_folder_path, 'stupid_variable')
    imgs = [imread(f) for f in files]

    masks, flows, styles = model.eval(imgs, diameter=diam, channels = [0,0], 
                                            flow_threshold=0.4, do_3D = False)

    if save_masks:
        masks_flows_to_seg(imgs, masks, flows, diam, 'generated_masks')

    return masks



def main():
    image_folder_path = "//storage3.ad.scilifelab.se/alm/BrismarGroup/Hanna/Data_from_Emma/Confluent_images/short"
    image_folder_path = "//storage3.ad.scilifelab.se/alm/BrismarGroup/Hanna/Data_from_Emma/onehourconfluent"
    model_path = 'C:/Users/workstation3/Documents/CP_20230705_confl'

    segmentation(image_folder_path, model_path, save_masks = True)

if __name__ == "__main__":
    main()