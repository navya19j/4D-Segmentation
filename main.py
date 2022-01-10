#!/usr/bin/env python3

import os
import sys
import shutil
import test
import argparse

import yaml


import boundary_box
import newcoord
import sort
import track
import train

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Segmentation pipeline command line tool"
    )
    parser.add_argument(
        "-c", "--config", type=str, default="config.yaml", 
        help="the pipeline configuration file (yaml)"
    )

    args = parser.parse_args()
    conf_filename = args.config

    # load config
    with open(conf_filename, "r") as f:
        print(f"Loading {conf_filename} config file")
        try:
            conf = yaml.safe_load(f)
        except Exception as e:
            print(e)
            print(f"Unable to load {conf_filename}")
    # TODO: validate the yaml file

    # run training
    if conf["train"]["run"]:

        train.main(config=conf["train"])

    # run testing
    if conf["test"]["run"]:

        test.main(conf["test"])

    # run tracking
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
