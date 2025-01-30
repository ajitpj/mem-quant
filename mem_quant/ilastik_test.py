import os, nd2
from pathlib import Path
import numpy as np
from xarray import DataArray
from napari.viewer import Viewer
from ilastik.experimental.api import from_project_file
import tifffile as tif
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects

project_dir = Path('./Models')
project_file = project_dir / Path('GFP-mCherry-GFP.ilp')
ilastik_model = from_project_file(project_file)


def readND2(im_path: Path) -> np.array:
    '''
    Function opens an ND2 file and returns it as a numpy array
    Inputs:
    im_path = Path obejct to the ND2 image file
    Outputs:
    im_arr  = Image stack as a numpy array
    '''
    try:
        im_arr = nd2.imread(im_path)
        return im_arr
    except:
        raise ValueError(f"Could not read {im_path} file")

def _3Dpredictor(im_arr: np.array, model_path: Path):
    '''
    Inputs:
    img_arr: nd2 files have zcyx as dims
    model_path: Path object for the model file 
    '''
    ndims = im_arr.shape

    ilastik_model = from_project_file(model_path)
    foreground = ilastik_model.predict(DataArray(im_arr, 
                                                 dims=["z","y","x"]))[:,:,:,1]


    return foreground

def predictor(im_arr: np.array, channel: int, ilastik_model):
    '''
    Inputs:
    img_arr: nd2 files have zcyx as dims
    channel: Channel number to process
    ilastik_model: 
    '''
    ndims = im_arr.shape
    print(ndims)
    try: 
        channel < ndims[1]
        foreground = np.zeros((ndims[0], ndims[2], ndims[3]), dtype = float)

        for i in np.arange(ndims[0]):
            foreground[i,:,:] = ilastik_model.predict(DataArray(im_arr[i, channel, :, :], 
                                                                dims=["y","x"]))[:,:,0]

        return foreground
    except:
        raise ValueError(f"Specified channel number exceeds number of channels: {ndims[1]}")

def createMask(roi: np.array, roi_coords: list, mask_shape: tuple) -> np.array:
    '''
    Function creates a mask of the same dimensions as the original
    image with the ROI inserted at the correct location. This will
    simplify post-processing using saved analysis files.
    Inputs:
    roi = 2-D or 3-D boolean array
    roi_coord = [xstart:xend,ystart:yend]
    shape = shape of the final mask [z, x, y]
    Outputs:
    mask = final mask
    '''
    try:
        len(roi_coords) == 4
    except:
        raise ValueError(f"Wrong roi coordinates: {roi_coords}")
    else:
        mask = np.zeros(mask_shape, dtype=bool)
        xstart, xend, ystart, yend = roi_coords
        if len(mask_shape) == 2:
            mask[xstart:xend,ystart:yend] = roi.astype(bool)
        elif len(mask_shape) == 3:
            for i in np.arange(mask_shape[0]):
                mask[i,xstart:xend,ystart:yend] = roi[i,:,:]


    return mask

    