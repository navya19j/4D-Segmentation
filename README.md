# 4D Segmentation

This repository contains the code for automated segmentation and tracking of endosomes in a living cells acquired using LLSM.

![4D Endosome Segmentation](doc/img/4d_segmentation.gif)

## Getting Started

1. Install Anaconda (https://docs.anaconda.com/anaconda/install/index.html)

2. Create the 4d_segmentation environment
```
$ conda env create --file=environment.yml
```

3. Activate the environment

```
$ conda activate 4d_segmentation
```

4. (Optional) Download the pre-trained weights from here: [Google Drive](https://drive.google.com/file/d/1t8V405peURVxjs-JG6N7L6nbR9Mi6H7d/view?usp=sharing)
5. (Optional) Download the sample dataset [Google Drive](https://drive.google.com/file/d/1LfjeXSPr-iYLuYKrL6f9i0d8IfX_s_ET/view?usp=sharing)

You are now ready to run the segmentation and tracking pipeline.

## Running the Pipeline

The pipeline expects a directory of .tif volumetric image repesenting the cell imaging at each timestep. 

To run the entire pipeline:
```
$ python3 main.py -c config.yaml
```

The configuration of the pipeline can be changed by editing the config.yaml file (or by supplying your own).

### config.yaml
```
train:
    run: do you want to run model training
    train_data: the directory containing the training data
    label_data: the directory containing the labelled data
    cellname: the name of the cell containing the training / labelled data
    train_test_split: do you want to split the dataset into train and test sets. (train_test_split)
    truth:  do you want to 'not' reduce the images to 128x128 size. (downsample)
    num_layers: the number of layers in the model(U-Net).
    batch_size: the size of the training batch.
    num_epochs: the number of epochs to train for.
    checkpoint: the pretrained model checkpoint to continue training from (null if starting from scratch)
test:
    run: do you want to run model testing (segmentation)
    test_data: the directory containing the testing data.
    cellname: the name of the cell containing the testing data.
    truth: do you want to 'not' reduce the images to 128x128 size. (downsample)
    num_layers: the number of layers in the model (U-Net).
    checkpoint: the model checkpoint to load
track:
    run: do you want to run tracking.
    cellname: the name of the cell containing the testing data.
    min_area: the minimum area for a bounding box per slice.
    min_volume: the minimum volume for a bounding box per volume. 
    min_iou_threshold_2d: the minimum iou threshold (2-dimensional)
    min_iou_threshold_3d: the minimum iou threshold (3-dimensional)

```

<!-- You are also able to run each component individually using the following instructions:  
### Train

To train a model on a custom dataset:
```
$ python3 train.py
```

### Test

To test the segmentation model on a custom dataset:


```
$ python3 test.py
```

TODO: add a note about thresholding -->

Test data must be organized in the format "Directory Name" > "Cellname" > All images to be tested.

### Track
Multi-stage pipeline
- Extract 2D Bounding Boxes for each image slice (z-dimension) (boundary_box.py)
- Track objects (2D) in z-dimension (sort.py)
- Extract 3D Bounding Boxes for each image volume (newcoord.py)
- Track objects (3D) in time (track.py). 

### Visualise

You can visualise the results of both the segmentation and tracking using:

```
$ streamlit run app.py
```

To view the images, labels and segmented masks:
- Select your imaging directories, and use the sliders to select a z-slice, and a timestep.
To view the tracked endosome objects:
- Select the relevant objects.txt file (the result of tracking pipeline). 



## File Description

pre_process_original.py - Resizes input images from (80,608,400) to (608,608,608) by *wrap padding*.

pre_process_2.py - Resizes input images from (608,608,608) to (128,128,128) by *INTER_CUBIC*.

dataset.py - contains the dataset class for input image. 

train.py - trains the model on input parameters.

unet.py - contains the UNET architecture.

utils.py - utility functions.

helpers.py - 2D Box class

helper3D.py - 3D box class

boundary_box.py - creating bounding boxes from contours

sort.py - finding box of maximum overlap i.e. finding a 3D bounding box for endosomes in the cell image

newcoord.py - getting the 3D box coordinates for all 3D bounding boxes

threshold.py - thresholds the grayscale segmented masks.

tracker.py - contains the class for tracker algorithm

track.py - tracking endosomes using linear assignment

### Example Images

- Data - Original Input Images with dimensions (80,608,400).

- Labeled - Original Input Image Segmented Masks with dimensions (80,608,400).

### Tracking Data

- track.txt - contains the map from object identifiers to the time frames containing the box

- object.txt - contains the map from object identifiers to the coordinates of the bounding box of that object with time frame.

## Citation
TODO
