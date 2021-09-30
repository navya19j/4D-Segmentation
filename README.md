# 4D Segmentation

This repository contains the code for automatically segmenting and tracking endosomes in a $TYPE_OF_CELL cell using $TYPE_OF_IMAGING imaging.

TODO: check the correct wording on this

TODO: add an image here showing the output

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

4. (Optional) Download the pre-trained weights from here: [TODO: Add Link]()
5. (Optional) Download the sample dataset [TODO: Add Link]()

You are now ready to run the segmentation and tracking pipeline.

## Running the Pipeline

The pipeline expects a directory of .tif volumetric image repesenting the cell imaging at each timestep. 
TODO: check what is the correct wording for this. 

To run the entire pipeline:
```
$ python3 main.py
```
and follow the prompts


You are also able to run each component individually using the following instructions:  
### Train

To train a model on a custom dataset:
```
$ python3 train.py
```

### Test

To test the segmentation model on a custom dataset:
Test data must be organized in the format "Directory Name" > "Cellname" > All images to be tested.

```
$ python3 test.py
```

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

tracker.py - contains the class for tracker algorithm

track.py - tracking endosomes using linear assignment

### Example Images

- Data - Original Input Images with dimensions (80,608,400).

- Labeled - Original Input Image Segmented Masks with dimensions (80,608,400).

### Tracking Data

- track.txt - contains the map from object identifiers to the time frames containing the box

- object.txt - contains the map from object identifiers to the coordinates of the bounding box of that object with time frame.

- _3Dboxes.txt - 3D coordinates (output of newcoord.py)

- _bb.txt - frame wise bounding coordinates (output of boundary_box.py)

- _track.txt - tracked pairs of boxes (i,i+1 : output of dsort.py)

## Citation
TODO
