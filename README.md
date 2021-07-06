# 4D Segmentation
 
pre_process_original.py - Resizes input images from (80,608,400) to (608,608,608) by *wrap padding*.

pre_process.py - Resizes input images from (80,608,400) to (128,640,512) by *wrap padding*.

pre_process_2.py - Resizes input images from (608,608,608) to (128,128,128) by *Image.nearest*.

pre_process_3.py - Resizes input images from (128,640,512) to 20 crops of (128,128,128) by *array slicing*.

dataset.py - contains the dataset class for input image. 

train.py - trains the model on input parameters.

unet.py - contains the UNET architecture.

utils.py - utility functions.

test_x.py - when using cropped images, this file takes an input image, crops it, passes the cropped input to the model and aligns the output sequentially to produce the resized output.


_Example Images_:

Data - Original Input Images with dimensions (80,608,400).

Labeled - Original Input Image Segmented Masks with dimensions (80,608,400).

Data_vol - Cropped patches of Input Images. 

Labeled_vol -  Cropped patches of Input Image Segmented Masks. 

Data_Resized - Original Input Images resized to (128,128,128).

Labeled_Resized - Original Input Image Segmented Masks resized to (128,128,128).

