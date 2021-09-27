import numpy as np
import matplotlib.pyplot as plt
import glob
import ast
import cv2
from helper3D import *
from tifffile import *
import os
from collections import OrderedDict
from scipy.optimize import linear_sum_assignment as linear_assignment

class Tracker:

    # objects contains all current ids and positions of objects being tracked
    # active counter for sequential tracks of each object

    def __init__(self,active_threshold = 0, non_active_threshold = 10, iou_threshold = 0.4):
        self.next_box_id = 0
        self.objects = OrderedDict()
        self.active = OrderedDict()
        self.lost = OrderedDict()
        self.finished = OrderedDict()
        self.non_active_threshold = non_active_threshold
        self.active_threshold = active_threshold
        self.iou_threshold = iou_threshold

    def add_object(self,new_location):
        self.objects[self.next_box_id] = new_location
        self.lost[self.next_box_id] = 0
        self.active[self.next_box_id] = 0
        self.next_box_id += 1

    def remove_object(self,object_id):
        del self.objects[object_id]
        del self.lost[object_id]
        del self.active[object_id]

    def assign_detection_to_tracker(self,detections, trackers, iou_threshold = 0.4):

        iou_matrix = np.zeros((len(detections),len(trackers)))

        for d,det in enumerate(detections):
            for t,trk in enumerate(trackers):
                iou_matrix[d,t] = IOU(det,trk)

        matched_indices = linear_assignment(-iou_matrix)
        # print(matched_indices)
        unmatched_detections = []

        for d,det in enumerate(detections):
            if d not in matched_indices[0][:]:
                unmatched_detections.append(d)

        unmatched_trackers = []

        for t,trk in enumerate(trackers):
            if t not in matched_indices[1][:]:
                unmatched_trackers.append(t)

        matches = []

        for i in range (0, len(matched_indices[0])):
            if iou_matrix[matched_indices[0][i],matched_indices[1]][i] < iou_threshold:
                unmatched_detections.append(matched_indices[0][i])
                unmatched_trackers.append(matched_indices[1][i])

            else:
                matches.append((matched_indices[0][i],matched_indices[1][i]))


        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

    def update(self,detections):

        if len(detections) == 0:
            lost_box = list(self.lost.keys())
            for object_id in lost_box:
                self.lost[object_id] += 1
                self.active[object_id] = 0
                if self.lost[object_id] > self.non_active_threshold :
                    self.remove_object(object_id)

            return self.objects

        if len(self.objects) == 0:
            for i in range (0,len(detections)):
                self.add_object(detections[i])

        else:
            object_ids = list(self.objects.keys())
            previous_objects = np.array(list(self.objects.values()))
            matches, unmatched_detections, unmatched_trackers = self.assign_detection_to_tracker(detections,previous_objects,self.iou_threshold)

            for (row,col) in matches:
                object_id = object_ids[col]
                self.objects[object_id] = detections[row]
                self.lost[object_id] = 0
                self.active[object_id] += 1

            for row in unmatched_trackers:
                object_id = object_ids[row]
                self.active[object_id] -=1
                self.lost[object_id] += 1

                if self.lost[object_id] > self.non_active_threshold:
                    self.remove_object(object_id)

            for col in unmatched_detections:
                self.add_object(detections[col])

        # print(self.objects)
        # print("Active")
        # print(self.active)
        active_objects = dict(filter(lambda elem: self.active[elem[0]]>=self.active_threshold,self.objects.items()))

        return active_objects


    