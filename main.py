#!/usr/bin/env python3

from subprocess import call
import os
import shutil
import boundary_box
import sort
import newcoord
import sys
import track
import test
import train

if __name__ == "__main__":

    

    with open(sys.argv[1],'r') as main_file:
        main_inputs = [line.strip() for line in main_file]

    to_train, to_test, to_track = main_inputs[0],main_inputs[1],main_inputs[2]

    if (to_train == "Y"):
        
        with open(sys.argv[2], 'r') as file:
            train_inputs = [line.strip() for line in file]

        train.main(train_inputs)

    if (to_test == "Y"):

        with open(sys.argv[3], 'r') as file:
            test_inputs = [line.strip() for line in file]

        test.main(test_inputs)

    if (to_track == "Y"):

        with open(sys.argv[4], 'r') as file:
            track_inputs = [line.strip() for line in file]

        root = os.getcwd()
        # cellname = input("Enter cellname: ")
        # min_area = input("Minimum area of Box: ")
        # min_vol = input("Minimum Volume of Box: ")

        cellname,min_area,min_vol,iou_square,iou_cube = track_inputs[0],int(track_inputs[1]),int(track_inputs[2]),int(track_inputs[3]),int(track_inputs[4])

        dir = os.path.join(root,"output",cellname,"predicted_mask")
        os.makedirs(os.path.join(dir,"bounding_box"), exist_ok=True)
        os.makedirs(os.path.join(dir,"complete_bounding_box"), exist_ok=True)
        os.makedirs(os.path.join(dir,"3D_Box"), exist_ok=True)

        boundary_box.create_bound_box(dir,min_area)
        sort.run(dir,iou_square)
        newcoord.run(dir,min_vol)
        track.run(dir,iou_cube)

        shutil.rmtree(os.path.join(dir,"bounding_box"))
        shutil.rmtree(os.path.join(dir,"complete_bounding_box"))
        shutil.rmtree(os.path.join(dir,"3D_Box"))
        
        print("Saved final trajectory in object.txt and track.txt")


        # add visualisation here