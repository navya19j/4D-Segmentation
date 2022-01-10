class Box:

    """
        Class for 2D Bounding Boxes

        args:
            x: X coordinate
            y: Y coordinate

            w: width
            h: height

    """

    def __init__(self,x,y,w,h):
        self.x = int(x)
        self.y = int(y)
        self.w = int(w)
        self.h = int(h)

    def area(self):
        if (self.w > 0 and self.h > 0):
            return(self.w*self.h)
        else:
            return 0

def overlap_box(Box_1,Box_2):

    """
        Gives the bounding box at the intersection of Box_1 and Box_2

        args:
            Box_1: A bounding box
            Box_2: A bounding box
    """

    x_1 = Box_1.x
    y_1 = Box_1.y
    w_1 = Box_1.w
    h_1 = Box_1.h
    x_12 = x_1 + w_1
    y_12 = y_1 + h_1

    x_2 = Box_2.x
    y_2 = Box_2.y
    w_2 = Box_2.w
    h_2 = Box_2.h
    x_22 = x_2 + w_2
    y_22 = y_2 + h_2

    x_box = max(x_1,x_2)
    y_box = max(y_1,y_2)
    x2_box = min(x_12,x_22)
    y2_box = min(y_12,y_22)
    h_box = y2_box - y_box
    w_box = x2_box - x_box

    overlap = Box(x_box,y_box,w_box,h_box)

    return overlap

def intersection(Box_1,Box_2):

    """
        Gives the volume of the bounding box at the intersection of Box_1 and Box_2

        args:
            Box_1: A bounding box
            Box_2: A bounding box
    """

    Box_overlap = overlap_box(Box_1,Box_2)
    
    return (Box_overlap.area())

def union(Box_1,Box_2):

    """
        Gives the union of Box_1 and Box_2

        args:
            Box_1: A bounding box
            Box_2: A bounding box
    """

    area_1 = Box_1.area()
    area_2 = Box_2.area()
    inter = intersection(Box_1,Box_2)

    return (area_1+area_2-inter)

def IOU(Box_1,Box_2):

    """
        Gives the IOU measure of Box_1 and Box_2
        # added 1e-7 to incoporate the case with union area = 0

        args:
            Box_1: A bounding box
            Box_2: A bounding box
    """
    
    area_overlap = intersection(Box_1,Box_2)
    area_union = union(Box_1,Box_2)

    return (area_overlap/(area_union+1e-7))