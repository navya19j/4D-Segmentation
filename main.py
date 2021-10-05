#!/usr/bin/env python3

from subprocess import call
import os
import shutil
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

    to_test = input("Do you wish to test on new images? (Y/N)")

    if (to_test == "Y"):

        test.main()

    to_track = input("Do you wish to track the segmented images? (Y/N)?")

    if (to_track == "Y"):
        root = os.getcwd()
        cellname = input("Enter cellname: ")
        dir = os.path.join(root,"output",cellname,"predicted_mask")

        os.makedirs(os.path.join(dir,"bounding_box"), exist_ok=True)
        os.makedirs(os.path.join(dir,"complete_bounding_box"), exist_ok=True)
        os.makedirs(os.path.join(dir,"3D_Box"), exist_ok=True)

        boundary_box.create_bound_box(dir)
        sort.run(dir)
        newcoord.run(dir)
        track.run(dir)

        shutil.rmtree(os.path.join(dir,"bounding_box"))
        shutil.rmtree(os.path.join(dir,"complete_bounding_box"))
        shutil.rmtree(os.path.join(dir,"3D_Box"))
        
        print("Saved final trajectory in object.txt and track.txt")


        # add visualisation here