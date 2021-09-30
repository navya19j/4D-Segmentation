#!/usr/bin/env python3

from subprocess import call
import os

import boundary_box
import sort
import newcoord
import track
import test
import train

if __name__ == "__main__":

    to_train = input("Do you wish to train the model? (Y/N)?")

    if (to_train == "Y"):
        train.main()
        # call(["python", "train.py"])

    to_test = input("Do you wish to test on new images? (Y/N)")

    if (to_test == "Y"):

        test.main()
        # call(["python", "test.py"])

    to_track = input("Do you wish to track the segmented images? (Y/N)?")

    if (to_track == "Y"):
        root = os.getcwd()
        cellname = input("Enter cellname: ")
        dir = os.path.join(root,"output",cellname,"predicted_mask")
        path_bb = os.path.join(dir,"bounding_box")
        path_cbb = os.path.join(dir,"complete_bounding_box")
        path_3d = os.path.join(dir,"3D_Box")

        if (not os.path.isdir(path_bb)):
            os.mkdir(path_bb)
        if (not os.path.isdir(path_cbb)):
            os.mkdir(path_cbb)
        if (not os.path.isdir(path_3d)):
            os.mkdir(path_3d)

        # print("Generating Bounding Boxes")
        # call(["python", "boundary_box.py","--dir",dir])
        # print("Tracking across z-direction")
        # call(["python", "sort.py","--dir",dir])
        # print("Generating 3D bounding Boxes")
        # call(["python", "newcoord.py","--dir",dir])
        # print("Tracking across time")
        # call(["python", "track.py","--dir",dir])
        
        boundary_box.create_bound_box(dir)
        sort.run(dir)
        newcoord.run(dir)
        track.run(dir)
        
        print("Saved final trajectory in object.txt and track.txt")

        # add visualisation here


