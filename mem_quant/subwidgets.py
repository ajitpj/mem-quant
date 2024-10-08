from __future__ import annotations
from qtpy import QtWidgets
from PyQt5.QtCore import Qt

def create_dir_selector_widget() -> dict[str,  QtWidgets.QWidget]:
    '''
    Selects directory storing ND2 files Widgets 
    ------------------------------
    RETURNS:
        widgets: dict
            - select_dir_button: QtWidgets.QPushButton
    '''

    path_selector_button = QtWidgets.QPushButton()
    path_selector_button.setText('ND2 data folder')
    path_selector_button.setToolTip(
        'Select folder containig ND2 files'
    )

    file_selector = QtWidgets.QComboBox()

    model_selector_button = QtWidgets.QPushButton()
    model_selector_button.setText('ilastik model')
    model_selector_button.setToolTip(
        'Select PixelClassification model'
    )

    widgets = {'path_selector_button' :  ("", path_selector_button)}
    widgets["file_selector"] = ("", file_selector)
    widgets["model_selector_button"] = ("", model_selector_button)

    return widgets

def create_roi_selection_widgets() -> dict[str, QtWidgets.QWidget]:
    '''
    Combo box containing all ND2 files in the directory
    ------------------------------
    RETURNS:
        widgets: dict
            file_selector: QtWidgets.QComboBox
            select_cell_button: QtWidgets.QPushButton
            accept_segmentation_button: 
    '''
    ref_channel_selector = QtWidgets.QComboBox()
    ref_channel_selector.setEnabled(False)

    select_cell_button = QtWidgets.QPushButton()
    select_cell_button.setText('Segment')
    select_cell_button.setToolTip("ilastik pixel classification")
    select_cell_button.setEnabled(False)
    
    foreground_thresh = QtWidgets.QSpinBox() #Qt.Orientation.Horizontal
    foreground_thresh.label = "Foreground threshold (%)"
    foreground_thresh.setMaximum(100)
    foreground_thresh.setMinimum(0)
    foreground_thresh.setValue(50)
    foreground_thresh.setSingleStep(2)
    foreground_thresh.setEnabled(False)

    accept_segmentation_button = QtWidgets.QPushButton()
    accept_segmentation_button.setText('Finalize')
    accept_segmentation_button.setToolTip("Process and save cell")
    accept_segmentation_button.setEnabled(False)

    # progress_bar = QtWidgets.QProgressBar()
    # progress_bar.setMaximum(100)
    # progress_bar.setMinimum(0)
    # progress_bar.setValue(0)
    # progress_bar.label = "Writing..."
    # progress_bar.setEnabled(False)

    widgets = {'ref_channel_selector' : ("ref. ch.", ref_channel_selector)}
    widgets['select_cell_button'] = ("", select_cell_button)
    widgets['foreground_thresh'] = ("threshold", foreground_thresh)
    widgets['accept_segmentation_button'] = ("", accept_segmentation_button)
    # widgets['progress_bar'] = ("", progress_bar)

    return widgets

def create_saving_widgets() -> dict[str, QtWidgets.QWidget]:
    '''
    Button for saving files
    ------------------------------
    RETURNS:
        widgets: dict
            file_selector: QtWidgets.QComboBox
            select_cell_button: QtWidgets.QPushButton
            accept_segmentation_button: 
    '''
    save_data_button = QtWidgets.QPushButton()
    save_data_button.setText('Save data')
    save_data_button.setToolTip("Save analysis to files...")
    widgets = {'save_data_button' : save_data_button}
    save_data_button.setEnabled(False)

    return widgets

# def create_channel_lineEdits_widgets() -> dict[str, QtWidgets.QWidget]:
#     '''Creates TextEdit boxes for the four channels '''
#     ch1_LineEdit = QtWidgets.QLineEdit('phs')
#     widgets = {'ch1_LineEdit' : ("Ch1", ch1_LineEdit)}

#     ch2_LineEdit = QtWidgets.QLineEdit()
#     ch3_LineEdit = QtWidgets.QLineEdit()
#     ch4_LineEdit = QtWidgets.QLineEdit()

#     widgets['ch2_LineEdit'] = ("Ch2", ch2_LineEdit)
#     widgets['ch3_LineEdit'] = ("Ch3", ch3_LineEdit)
#     widgets['ch4_LineEdit'] = ("Ch4", ch4_LineEdit)
    
#     return widgets

# def create_progressbar_widget() ->dict[str, QtWidgets.QWidget]:
#     progress_bar = QtWidgets.QProgressBar()
#     progress_bar.setMaximum(100)
#     progress_bar.setMinimum(0)
#     progress_bar.setValue(0)
#     progress_bar.label = "Writing..."
#     widgets = {"progress_bar":  progress_bar}

#     return widgets
# def create_config_widgets() -> dict[str, tuple[str, QtWidgets.QWidget]]:
#     '''
#     Creates Configuration Widgets 
#     ------------------------------
#     RETURNS:
#         widgets: dict
#             - thresholder: QtWidgets.QDoubleSpinBox
#             - confluency_est: QtWidgets.QSpinBox
#             - set_configs: QtWidgets.QPushButton
#     '''

#     thresholder = QtWidgets.QDoubleSpinBox()
#     thresholder.setRange(0, 100)
#     thresholder.setValue(0.5)
#     thresholder.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
#     thresholder.setToolTip(
#         'Set Confidence Hyperparameter'
#     )
#     thresholder.setWrapping(True)
#     widgets = {'thresholder': ('Confidence Threshold', thresholder)}

#     confluency_est = QtWidgets.QSpinBox()
#     confluency_est.setRange(100, 2000)
#     confluency_est.setValue(500)
#     confluency_est.setStepType(QtWidgets.QAbstractSpinBox.AdaptiveDecimalStepType)
#     confluency_est.setToolTip(
#         'Estimate the number of cells in a frame'
#     )

#     widgets['confluency_est'] = ('Number of Cells (Approx.)', confluency_est)

#     set_configs = QtWidgets.QPushButton('Push')
#     set_configs.setToolTip(
#         'Set Configurations'
#     )

#     widgets['set_configs'] = ('Set Configurations', set_configs)


#     return widgets 



    



