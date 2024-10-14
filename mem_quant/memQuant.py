
import datetime, pickle
import nd2
from pathlib import Path
import numpy as np
import pandas as pd
from qtpy import QtWidgets
import napari.utils.notifications as notifications
from typing import Optional, List, Dict
from dataclasses import dataclass, field
from xarray import DataArray
from ilastik.experimental.api import from_project_file
from mem_quant import ui # type: ignore
# from skimage.measure import label
from skimage.morphology import  opening, skeletonize, dilation, closing
from skimage.morphology import remove_small_objects, ball, disk
from skimage.segmentation import clear_border

@dataclass
class exptInfo:
    data_dir: Path
    model_path: Path
    name: str
    analysis_date: str
    metadata: dict
    channel_names: dict
    color_maps: list
    
    
@dataclass
class cell_data:
    file_name:   str       # filepath
    ref_channel: str       # ref. channel for pixel classification
    roi_coords:  Optional[list]=None      # user-selected [xstart, xend, ystart, yend]
    raw_mask:  Optional[np.array]=None  # ilastik classification confidence 
    processed_mask: Optional[np.array]=None  # post-processed foreground
    background_mask: Optional[np.array]=None # background mask based on processed mask
    threshold: float=0.5    # Default threshold for foregroundpixels
    z_index:   int=10       # User-selected plan

@dataclass
class summary_data:
    summary: dict = field(default_factory= lambda: {'Filename'   : str,
                                                    'Model_used' : str,
                                                    'Ref_channel': str,
                                                    'Threshold'  : float,
                                                    "ROI"        : list,
                                                    'Z_plane'    : int,
                                                    'GFP_signal' : float,
                                                    'GFP_bkg'    : float,
                                                    'Cy5_signal' : float,
                                                    'Cy5_bkg'    : float,
                                                    'GFP_list'   : list,
                                                    'Cy5_list'   : list,
                                                    'GFP_bkg_list': list,
                                                    'Cy5_bkg_list': list,
                                                    "npix_signal"    : int,
                                                    "npix_background": int
                                                   }
                        ) 

def select_dir(mQwidget: ui.mQWidget):
    '''
    Returns the user selected Path,
    updates the file-selection-list
    '''

    dir_path = Path(QtWidgets.QFileDialog.getExistingDirectory())
    nd2_list = []
    for filename in dir_path.glob('*.nd2'):
        nd2_list.append(filename)
    
    if not nd2_list:
        notifications.show_error(f'No Nd2 files in {dir_path}!')
        return
    else:

        # Initialize experiment metadata
        # colormaps for the three channels
        channel_cmap = {"Brightfield": "gray",
                        "Cy5"         : "magenta",
                        "GFP"         : "green"}

        # Initialize the dictionary holding available models
        # currently - trained for GFP and Cy5 separately.

        models_dict = {"GFP" : Path('./mem_quant/Models/membrane_GFP_v1.ilp'),
                       "Cy5" : Path('./mem_quant/Models/membrane_Cy5_v1.ilp')
                      }

        # Initialize experiment data
        mQwidget.exptInfo = {"data_dir"      : dir_path,
                             "models_dict"   : models_dict,
                             "name"          : dir_path.stem,
                             "analysis_date" : f'{datetime.date.today()}',
                             "colormap_dict" : channel_cmap,
                             "metadata"      : {},
                             "unstrcut_metadata" : {},
                             "channel_names" :[],
                             "ordered_cmpas" : [],
                            }
        
        _retrievemetadata(mQwidget, nd2_list[0])

        # Dictionary for storing cell_data structures
        # {key =current_cell_index, value = cell_data}
        mQwidget.current_cell_index = 0
        mQwidget.all_data = {}
        # Summary dataframe

        mQwidget.summary_df = pd.DataFrame([summary_data().summary])


        # populate the file_selector_combobox
        for name in sorted(nd2_list):
            mQwidget.file_selector.addItem(name.name)
        

def _retrievemetadata(mQWidget: ui.mQWidget, im_path: Path):
    '''
    '''
    with nd2.ND2File(im_path) as nd2_file:
        unstruct_metadata = nd2_file.unstructured_metadata()['ImageTextInfoLV']
    
    with nd2.ND2File(im_path) as nd2_file:    
        metadata = nd2_file.metadata
   
    mQWidget.metadata = metadata
    mQWidget.unstruct_metadata = unstruct_metadata

    # Retrieve channel names
    channel_names = []
    for i in np.arange(metadata.contents.channelCount):
        channel_names.append(metadata.channels[i].channel.name)

    mQWidget.exptInfo["channel_names"] = channel_names
    
    # Assemble the ordered colormap list - passed to the viewer.add_image
    # function for proper display
    ordered_cmaps = []
    for name in channel_names:
        ordered_cmaps.append(mQWidget.exptInfo["colormap_dict"][name])

    mQWidget.exptInfo["ordered_cmaps"] = ordered_cmaps
    # Write a text file with the channel exposure times and intensities
    data_dir = mQWidget.exptInfo["data_dir"]
    metadata_name = data_dir.as_posix().split(data_dir.root)[-1] + '_metadata.txt'
    metadata_path =  data_dir / metadata_name 
    print(metadata_path, metadata_name)

    with open(metadata_path, 'w') as txtfile:
        txtfile.write(unstruct_metadata['SLxImageTextInfo']['TextInfoItem_5'])

    return

def loadND2(mQWidget: ui.mQWidget):
    '''
    Function opens an ND2 file
    '''
    im_name = mQWidget.file_selector.currentText()
    im_path = mQWidget.exptInfo["data_dir"] / im_name

    try:
        im_arr = nd2.imread(im_path)
    except:
        notifications.show_error(f"Could not read {im_path} file")
    else:
        # Display setup
        # remove previous layers
        mQWidget.viewer.layers.clear()

        mQWidget.viewer.add_image(im_arr, channel_axis=1,
                                  name = mQWidget.exptInfo["channel_names"],
                                  colormap=mQWidget.exptInfo["ordered_cmaps"],
                                 )
        
        mQWidget.viewer.layers['Brightfield'].visible = False
        mQWidget.viewer.layers['GFP'].visible = False

        # Add a layer to store the user-defined ROI
        mQWidget.viewer.add_shapes(name="cell_ROI")

        # Select the rectangle selection mode
        mQWidget.viewer.layers["cell_ROI"].mode = 'add_rectangle'
        notifications.show_info("Select an ROI and hit the process button!")

        #Activate GUI controls
        mQWidget.ref_channel_selector.setEnabled(True)
        if mQWidget.ref_channel_selector.count() == 0:
            for name in mQWidget.exptInfo["channel_names"]:
                if name != "Brightfield":
                    mQWidget.ref_channel_selector.addItem(name)
        
        mQWidget.select_cell_button.setEnabled(True)
        mQWidget.accept_segmentation_button.setEnabled(True)
        mQWidget.foreground_thresh.setEnabled(True)
        mQWidget.save_data_button.setEnabled(True)
        

        mQWidget.current_cell =  {"file_name"      : im_name,
                                  "ref_channel"    : mQWidget.ref_channel_selector.currentText(),
                                  "roi_coords"     : None,
                                  "raw_mask"       : None,
                                  "processed_mask" : None,
                                  "background_mask": None,
                                  "threshold"      : None,
                                  "z_index"        : None
                                 }

        print(f"Currently viewing: {im_path.name}")

    return

def select_cell_button_callback(mQWidget: ui.mQWidget):
    ''' 
    Function prompts the user to select an ROI,
    uses an ilastik pixel classification model to segment
    the ROI, and then displays the result to the user as a new
    prediction layer.
    '''

    if  mQWidget.viewer.layers["cell_ROI"].data:
        # clear previous cell_ROI layer
        current_roi = mQWidget.viewer.layers["cell_ROI"].data[-1]
        xstart, xend, ystart, yend = _roi_to_range(current_roi)
        
        # Obtain the user-selected ref. channel for segmentation
        # and select the corresponding ilastik model
        ref_channel = mQWidget.ref_channel_selector.currentText()
        ilastik_model = mQWidget.exptInfo["models_dict"][ref_channel]

        # Obtain ROI and predict
        roi = mQWidget.viewer.layers[ref_channel].data[:, xstart:xend, ystart:yend]
        roi_pred = _3Dpredictor(roi, ilastik_model)
        
        #post-process the predictions
        roi_proc = _postprocess(roi_pred)

        mask_shape = mQWidget.viewer.layers[ref_channel].data.shape
        roi_coords = [xstart, xend, ystart, yend]

        # raw_mask - ilastik probabilities float64
        raw_mask        = _createMask(roi_pred, roi_coords, mask_shape)
        # processed_mask - boolean
        processed_mask  = _createMask(roi_proc, roi_coords, mask_shape)
        # background_mask - boolean
        background_mask = _defineBackground(processed_mask)


        # Stow data
        mQWidget.current_cell = {"file_name"      : mQWidget.file_selector.currentText(),
                                 "model_used"     : ilastik_model,
                                 "ref_channel"    : ref_channel,
                                 "roi_coords"     : roi_coords,
                                 "raw_mask"       : raw_mask,
                                 "processed_mask" : processed_mask,
                                 "background_mask": background_mask,
                                 "threshold"      : mQWidget.foreground_thresh.value()/100,
                                 "z_index"        : int(mQWidget.viewer.cursor.position[0])
                                }
        
        # 

        mQWidget.viewer.layers["cell_ROI"].opacity = 0.0
        
        mQWidget.viewer.add_labels(data=processed_mask.astype(bool)*39+
                                        background_mask.astype(bool)*222, 
                                   name = "predicted mask",
                                   opacity = 0.25
                                   )

    return

def thresh_slider_callback(mQWidget: ui.mQWidget):
    '''
    Takes the threshold value and re-performs post-processing
    displays the updated image.
    '''
    if mQWidget.viewer.layers["cell_ROI"].data:
        threshold = mQWidget.foreground_thresh.value()/100
        raw_mask = mQWidget.current_cell["raw_mask"]
        # mQWidget.viewer.add_labels(data=raw_mask, name="raw mask")
        processed_mask = _postprocess(raw_mask, threshold)
        background_mask = _defineBackground(processed_mask)

        if "predicted mask" in mQWidget.viewer.layers:
            mQWidget.viewer.layers.remove("predicted mask")

        mQWidget.viewer.add_labels(data=processed_mask.astype(bool)*39+
                                            background_mask.astype(bool)*222, 
                                    name = "predicted mask",
                                    opacity = 0.25,
                                    )
        mQWidget.current_cell["processed_mask"] = processed_mask
        mQWidget.current_cell["background_mask"]= background_mask
        mQWidget.current_cell["threshold"] = threshold

    return

def ref_channel_selector_callback(mQWidget):
    '''
    '''
    if mQWidget.ref_channel_selector.currentText() == "Cy5":
        mQWidget.foreground_thresh.setValue(75)
        mQWidget.viewer.layers["Cy5"].visible = True
        mQWidget.viewer.layers["GFP"].visible = False
    elif mQWidget.ref_channel_selector.currentText() == "GFP":
        mQWidget.foreground_thresh.setValue(50)
        mQWidget.viewer.layers["Cy5"].visible = False
        mQWidget.viewer.layers["GFP"].visible = True
    

def accept_segmentation_button_callback(mQWidget):
    '''
    Store the mQWidget.current_cell into the all_data structure.
    Once stored, the data cannot be manipulated.
    '''
    # Calculate signal intensities before saving all data
    channel_names = mQWidget.exptInfo["channel_names"]
    z_plane = int(mQWidget.viewer.cursor.position[0])
    processed_mask = mQWidget.current_cell["processed_mask"][z_plane,:,:]
    raw_mask       = mQWidget.current_cell["raw_mask"][z_plane,:,:]
    background_mask= mQWidget.current_cell["background_mask"][z_plane,:,:]
    #                              }
    # This summary will be appended to the dataframe for export
    current_cell = summary_data()
    current_cell.summary["Filename"]    = mQWidget.current_cell["file_name"]
    current_cell.summary["Model_used"]  = mQWidget.current_cell["model_used"]
    current_cell.summary["Ref_channel"] = mQWidget.current_cell["ref_channel"]
    current_cell.summary["Threshold"]   = mQWidget.current_cell["threshold"]
    current_cell.summary["ROI"]         = mQWidget.current_cell["roi_coords"]
    current_cell.summary["Z_plane"]     = mQWidget.current_cell["z_index"]

    for name in channel_names:
        if name != "Brightfield":
            im = mQWidget.viewer.layers[name].data[z_plane,:,:]
            signal_pixels     = _export_pixels(im, processed_mask)
            background_pixels = _export_pixels(im, background_mask)
            
            current_cell.summary[name + "_signal"] = np.mean(signal_pixels)
            current_cell.summary[name + "_bkg"]    = np.median(background_pixels)
            current_cell.summary[name+"_list"]     = signal_pixels
            current_cell.summary[name+"_bkg_list"] = background_pixels
            current_cell.summary["npix_signal"]    = len(signal_pixels)
            current_cell.summary["npix_background"]= len(background_pixels)

    current_cell_summary = pd.DataFrame([current_cell.summary])
    # Append to the overall summary
    mQWidget.summary_df = pd.concat([mQWidget.summary_df, current_cell_summary])

    mQWidget.all_data[mQWidget.current_cell_index] = mQWidget.current_cell
    notifications.show_info(f"Data for cell# {mQWidget.current_cell_index} finalized")
    print(f"Data for cell# {mQWidget.current_cell_index} finalized")
    mQWidget.current_cell_index += 1

    return

def save_data_button_callback(mQWidget: ui.mQWidget):
    '''
    Saves a summary and analysis file in the same directory as the
    nd2 files.
    '''
    timestamp = datetime.datetime.now().strftime("_%Y-%m-%d-%H.%M.%S")
    save_dir = mQWidget.exptInfo["data_dir"]
    save_file = mQWidget.exptInfo["name"] + timestamp +  '.xlsx'
    print(save_file)
    save_path = save_dir / save_file
    print(save_path)
    mQWidget.summary_df.to_excel(save_path)
    notifications.show_info(f"Python datafile {save_path} saved.")
    print(f"Python datafile {save_path} saved.")

    if mQWidget.save_pickle_checkbox.isChecked():
        save_file = mQWidget.exptInfo["name"] + '_' + timestamp + '.pickle'
        save_path = save_dir / save_file
        with open(save_path, 'wb') as f:
            pickle.dump(mQWidget.all_data, f,
                        protocol=pickle.HIGHEST_PROTOCOL)
            notifications.show_info(f"Python datafile {save_path} saved.")
            print(f"Python datafile {save_path} saved.")
    
    return


def _export_pixels(im: np.array, bool_mask:np.array):
    '''
    Calculate the signal and background intensities from the given
    image array and mask pair.
    '''
    try:
        im.shape == bool_mask.shape
        bool_mask.dtype == bool
    except:
        raise ValueError("Arrays not of the same size")
    else:
        intensity_map = im * bool_mask

    return intensity_map[intensity_map > 0]


def _roi_to_range(roi: np.array):
        roi = roi.astype(np.int16)
        
        if roi.shape[1] == 2:
            xstart = np.min(roi[:,0])
            xend   = np.max(roi[:,0])
            ystart = np.min(roi[:,1])
            yend   = np.max(roi[:,1])
        elif roi.shape[1] == 3:
            xstart = np.min(roi[:,1])
            xend   = np.max(roi[:,1])
            ystart = np.min(roi[:,2])
            yend   = np.max(roi[:,2])
        
        return xstart, xend, ystart, yend

def _2Dpredictor(im_arr: np.array, model_path: Path):
    '''
    Inputs:
    img_arr: nd2 files have zcyx as dims
    model_path: Path object for the model file 
    '''
    ndims = im_arr.shape

    ilastik_model = from_project_file(model_path)
    
    foreground = np.zeros(im_arr.shape, dtype=float)

    for i in np.arange(im_arr.shape[0]):
        foreground[i,:,:] = ilastik_model.predict(DataArray(im_arr[i, :, :], 
                                                     dims=["y","x"]))[:,:,0]
    return foreground

def _3Dpredictor(im_arr: np.array, model_path: Path):
    '''
    Inputs:
    img_arr: nd2 files have zcyx as dims
    model_path: Path object for the model file 
    '''
    ndims = im_arr.shape

    ilastik_model = from_project_file(model_path)
    foreground = ilastik_model.predict(DataArray(im_arr, 
                                                 dims=["z","y","x"]))[:,:,:,0]


    return foreground

def _postprocess(foreground: np.array, threhsold=0.5) -> np.array:
    '''
    Function removes all non-membrane pixels from the foreground 
    and membrane-pixels from the background
    Inputs:
    foreground = ilastik foreground pixels
    background = ilastic background pixels
    Outputs:
    post_fore = post-processed foreground
    '''
    # 
    min_size = 500
    ball_size = 3
    #
    foreground = foreground > threhsold
    foreground = closing(foreground, ball(ball_size))

    for i in np.arange(foreground.shape[0]):
        foreground[i,:,:] = clear_border(foreground[i,:,:])
        
        foreground[i,:,:] = remove_small_objects(foreground[i,:,:], 
                                                 min_size = min_size)

    return foreground

def _defineBackground(processed_mask: np.array) -> np.array:
    '''
    Function takes in the foreground array and define the background
    as concentric pixel rings on either side. To be used for membrane-
    localized proteins. The region definition can be used to calculate
    the local background.
    Inputs:
    foreground: boolean array defining foreground pixels
    pars      : optional parameters
    Outputs:
    background: boolean array definining background pixels
    '''
    try:
        processed_mask.dtype == bool
    except:
        notifications.show_error(f"the passed array must be boolean")
        raise ValueError("Input must be a boolean array.")
    else:
        # skeletonize & grow
        footprint = disk(7)
        if len(processed_mask.shape) == 2:
            background = skeletonize(processed_mask)
            background = dilation(background, footprint=footprint)
            background = background - processed_mask
        else:
            background = np.zeros_like(processed_mask)
            for i in np.arange(processed_mask.shape[0]):
                background[i,:,:] = skeletonize(processed_mask[i,:,:])
                background[i,:,:] = dilation(background[i,:,:], footprint=footprint)
                background[i,:,:] = np.logical_xor(background[i,:,:], processed_mask[i,:,:])

    return background

def _createMask(roi: np.array, roi_coords: list, mask_shape: tuple) -> np.array:
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
        mask = np.zeros(mask_shape)
        xstart, xend, ystart, yend = roi_coords
        if len(mask_shape) == 2:
            mask[xstart:xend,ystart:yend] = roi
        elif len(mask_shape) == 3:
            for i in np.arange(mask_shape[0]):
                mask[i,xstart:xend,ystart:yend] = roi[i,:,:]


    return mask


   