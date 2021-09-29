from subprocess import call
import os

to_train = input("Do you wish to train the model? (Y/N)?")

if (to_train == "Y"):
    call(["python", "train.py"])

to_test = input("Do you wish to test on new images? (Y/N)")

if (to_test == "Y"):

    call(["python", "test.py"])

to_track = input("Do you wish to track the segmented images? (Y/N)?")

if (to_track == "Y"):
    root = os.getcwd()
    dir = input("Enter directory containing masks to be tracked: ")
    path_bb = os.path.join(root,dir,"bounding_box")
    path_cbb = os.path.join(root,dir,"complete_bounding_box")
    path_3d = os.path.join(root,dir,"3D_Box")

    if (not os.path.isdir(path_bb)):
        os.mkdir(path_bb)
    if (not os.path.isdir(path_cbb)):
        os.mkdir(path_cbb)
    if (not os.path.isdir(path_3d)):
        os.mkdir(path_3d)

    print("Generating Bounding Boxes")
    call(["python", "boundary_box.py","--dir",dir])
    print("Tracking across z-direction")
    call(["python", "sort.py","--dir",dir])
    print("Generating 3D bounding Boxes")
    call(["python", "newcoord.py","--dir",dir])
    print("Tracking across time")
    call(["python", "track.py","--dir",dir])
    print("Saved final trajectory in object.txt and track.txt")

    # add visualisation here


