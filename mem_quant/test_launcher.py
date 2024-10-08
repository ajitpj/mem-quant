from napari import Viewer, run
from ui import mQWidget
from memQuant import exptInfo
from pathlib import Path

viewer = Viewer()
dock_widget, plugin_widget = viewer.window.add_plugin_dock_widget(
    "mem-quant", "Quantify membrane eSAC"
)
