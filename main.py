#!/usr/bin/env python3

import os
import shutil
import test

import yaml

import boundary_box
import newcoord
import sort
import track
import train

if __name__ == "__main__":

    with open("config.yaml", "r") as f:
        try:
            conf = yaml.safe_load(f)
        except Exception as e:
            print(e)
            print("Unable to load config.yaml")

    if conf["train"]["run"]:

        train.main(config=conf["train"])

    if conf["test"]["run"]:

        test.main(conf["test"])

    if conf["track"]["run"]:

        # load tracking config
        ROOT = os.getcwd()
        CELLNAME = conf["track"]["cellname"]
        MIN_AREA = conf["track"]["min_area"]
        MIN_VOLUME = conf["track"]["min_volume"]
        MIN_IOU_THRESHOLD_2D = conf["track"]["min_iou_threshold_2d"]
        MIN_IOU_THRESHOLD_3D = conf["track"]["min_iou_threshold_3d"]

        # setup temp directories
        TRACKING_DIR = os.path.join(ROOT, "output", CELLNAME, "predicted_mask")
        os.makedirs(os.path.join(TRACKING_DIR, "bounding_box"), exist_ok=True)
        os.makedirs(os.path.join(TRACKING_DIR, "complete_bounding_box"), exist_ok=True)
        os.makedirs(os.path.join(TRACKING_DIR, "3D_Box"), exist_ok=True)

        # run tracking pipeline
        boundary_box.create_bound_box(TRACKING_DIR, MIN_AREA)
        sort.run(TRACKING_DIR, MIN_IOU_THRESHOLD_2D)
        newcoord.run(TRACKING_DIR, MIN_VOLUME)
        track.run(TRACKING_DIR, MIN_IOU_THRESHOLD_3D)

        # cleanup temp directories
        shutil.rmtree(os.path.join(TRACKING_DIR, "bounding_box"))
        shutil.rmtree(os.path.join(TRACKING_DIR, "complete_bounding_box"))
        shutil.rmtree(os.path.join(TRACKING_DIR, "3D_Box"))

        print("Saved final trajectory in object.txt and track.txt")

        # add visualisation here
