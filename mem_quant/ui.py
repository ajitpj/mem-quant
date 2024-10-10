from __future__ import annotations
from napari.viewer import Viewer
from qtpy import QtWidgets
import mem_quant.subwidgets as subwidgets # type: ignore


class mQWidget(QtWidgets.QScrollArea):
    "memQuantWidget GUI Class"

    def __getitem__(self, key: str) -> QtWidgets.QWidget:
        return self._widgets[key]
    
    def __init__(self, napari_viewer: Viewer) -> None:
        """Instantiates the primary widget in napari.

        Args:
            napari_viewer: A napari viewer instance
        """
        super().__init__()
        
        # We will need to viewer for various callbacks
        self.viewer = napari_viewer
        # Let the scroll area automatically resize the widget
        self.setWidgetResizable(True) 

        self._main_layout = QtWidgets.QVBoxLayout() # main layout - outline
        self._main_widget = QtWidgets.QWidget() 
        self._main_widget.setLayout(self._main_layout)
        self.setWidget(self._main_widget)
        self._tabs = QtWidgets.QTabWidget()

        # Create widgets and add to layout
        self._widgets = {}
        self._add_dir_selector_widgets()
        self._add_roi_selection_widgets()
        self._add_save_data_widgets()
        self._add_configuration_widgets()

        self._main_layout.addWidget(self._tabs, stretch=0)

        # Make widgets into GUI attributes
        for name, widget in self._widgets.items():
            self.__setattr__(
                name, 
                widget
            )

    def _add_dir_selector_widgets(self):
        '''Adds expt_selector_widgets which are the output of sub_widgets.create_expt_selector_widgets. 
         The aforementioned function outputs -> dict[str: QtWidgets.QWidget]'''

        dir_selector_widgets = subwidgets.create_dir_selector_widget()
        self._widgets.update({key: value[1] for key, value in dir_selector_widgets.items()})

        widget_holder = QtWidgets.QGroupBox('Select ND2 files folder')
        layout = QtWidgets.QFormLayout()
        for label, widget in dir_selector_widgets.values():
            label_widget = QtWidgets.QLabel(label)
            layout.addRow(widget)

        widget_holder.setLayout(layout)
        self._main_layout.addWidget(widget_holder, stretch = 0)

    def _add_roi_selection_widgets(self):

        roi_selection_widgets = subwidgets.create_roi_selection_widgets()
        self._widgets.update({key: value[1] for key, value in roi_selection_widgets.items()})
        
        widget_holder = QtWidgets.QGroupBox('Select and Segment ROIs')
        layout = QtWidgets.QFormLayout()

        for label, widget in roi_selection_widgets.values():
            label_widget = QtWidgets.QLabel(label)
            layout.addRow(label_widget, widget)

        widget_holder.setLayout(layout)
        self._main_layout.addWidget(widget_holder, stretch = 0)

    def _add_save_data_widgets(self):
        save_data_widgets = subwidgets.create_saving_widgets()
        self._widgets.update(save_data_widgets)
        widget_holder = QtWidgets.QGroupBox('Save data...')
        layout = QtWidgets.QFormLayout()
        for value in save_data_widgets.values():
            layout.addRow(value)
        
        widget_holder.setLayout(layout)

        tab = QtWidgets.QWidget()
        tab.setLayout(layout)
        self._tabs.addTab(tab, "\N{GEAR}" + " Save")

    def _add_configuration_widgets(self):
        
        save_pickle_checkbox_widgets = subwidgets.create_save_pickle_checkbox_widgets()
        widget_holder = QtWidgets.QGroupBox('Configuration parameters')
        layout = QtWidgets.QFormLayout()
        for value in save_pickle_checkbox_widgets.values():
            layout.addRow(value)
        widget_holder.setLayout(layout)

        tab = QtWidgets.QWidget()
        tab.setLayout(layout)
        self._tabs.addTab(tab, "\N{GEAR}" + " Configure")

    # def _add_channel_lineEdits_widgets(self):

    #     chLineEdit_widgets = subwidgets.create_channel_lineEdits_widgets()

    #     self._widgets.update({key: value[1] for key, value in chLineEdit_widgets.items()})

    #     widget_holder = QtWidgets.QGroupBox('Channels')
    #     layout = QtWidgets.QFormLayout()
    #     for label, widget in chLineEdit_widgets.values():
    #         label_widget = QtWidgets.QLabel(label)
    #         layout.addRow(label_widget, widget)

        
    #     widget_holder.setLayout(layout)
    #     self._main_layout.addWidget(widget_holder, stretch = 0)

    # def _add_progressbar_widgets(self):
    #     pbar_widget = subwidgets.create_progressbar_widget()
        
    #     self._widgets.update(pbar_widget)
        
    #     widget_holder = QtWidgets.QGroupBox()
    #     layout = QtWidgets.QFormLayout()
    #     for widget in pbar_widget.values():
    #         layout.addRow(widget)
    #     widget_holder.setLayout(layout)
    #     self._main_layout.addWidget(widget_holder, stretch = 0)


    # def _add_file_widgets(self):
    #     "Adds file_widgets which are the output of sub_widgets.create_file_selector_widgets. The aforementioned function outputs -> dict[str: QtWidgets.QWidget]"

    #     file_widgets = subwidgets.create_file_selector_widgets()
    #     self._widgets.update(file_widgets)

    #     layout = QtWidgets.QFormLayout()
    #     for widget in file_widgets.values():
    #         layout.addRow(widget)

    #     tab = QtWidgets.QWidget()
    #     tab.setLayout(layout)
    #     self._tabs.addTab(tab, "FileIO")


    # def _add_config_widgets(self):
    #     "Adds config_widgets which are the output of sub_widgets.create_config_widgets. The aforementioned function outputs -> dict[str: tuple[str, QtWidgets.QWidget]]"

    #     config_widgets = subwidgets.create_config_widgets()
    #     self._widgets.update(
    #     {key: value[1] for key, value in config_widgets.items()}
    #     )

    #     layout = QtWidgets.QFormLayout()
    #     for label, widget in config_widgets.values():
    #         label_widget = QtWidgets.QLabel(label)
    #         label_widget.setToolTip(widget.toolTip())
    #         layout.addRow(label_widget, widget)

    #     tab = QtWidgets.QWidget()
    #     tab.setLayout(layout)
    #     self._tabs.addTab(tab, "Configs")