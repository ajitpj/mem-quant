# mem-quant
A napari plug-in to quantify membrane-localized protein using ilastik models

#Usage
1. Select the folder containing nd2 files. 
2. Select a file and the reference channel.
3. Select a pre-trained ilastik pixel classification model (to be added later, currently using a default model)
4. Create an ROI.
5. Threshold the ilastik confidence values to select background and foreground pixels.
6. finalize the pixel classification. 
7. Data export needs to be added; currently dumping a pickle (will get very large).
