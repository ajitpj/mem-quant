from __future__ import annotations
import logging
import napari
import numpy as np
import os
# os.environ["QT_API"] = "pyqt6"
from qtpy import QtWidgets
from mem_quant import memQuant # type: ignore
import mem_quant.ui as ui  # type:ignore

def create_memQuant_widget() -> ui.mQwidget:
    "Creates instance of ui.IXNwidget and sets callbacks"

    mQWidget = ui.mQWidget(
        napari_viewer=napari.current_viewer()
    )

    mQWidget.path_selector_button.clicked.connect(lambda: memQuant.select_dir(mQWidget))
    mQWidget.file_selector.currentIndexChanged.connect(lambda: memQuant.loadND2(mQWidget))
    mQWidget.select_cell_button.clicked.connect(lambda: memQuant.select_cell_button_callback(mQWidget))
    mQWidget.model_selector_button.clicked.connect(lambda: memQuant.model_selector_button_callback(mQWidget))
    mQWidget.foreground_thresh.valueChanged.connect(lambda: memQuant.thresh_slider_callback(mQWidget))
    mQWidget.accept_segmentation_button.clicked.connect(lambda: memQuant.accept_segmentation_button_callback(mQWidget))
    mQWidget.save_data_button.clicked.connect(lambda: memQuant.save_data_button_callback(mQWidget))
    return mQWidget

