class Box:
    def __init__(self,x,y,z,w,h,d):
        self.x = int(x)
        self.y = int(y)
        self.w = int(w)
        self.h = int(h)
        self.z = int(z)
        self.d = int(d)

    def vol(self):
        if (self.w > 0 and self.h > 0 and self.z > 0):
            return(self.w*self.h*self.z)
        else:
            return 0

def overlap_box(Box_1,Box_2):
    x_1 = Box_1.x
    y_1 = Box_1.y
    z_1 = Box_1.z

    w_1 = Box_1.w
    h_1 = Box_1.h
    d_1 = Box_1.d

    x_12 = x_1 + w_1
    y_12 = y_1 + h_1
    z_12 = z_1 + d_1

    x_2 = Box_2.x
    y_2 = Box_2.y
    z_2 = Box_2.z

    w_2 = Box_2.w
    h_2 = Box_2.h
    d_2 = Box_1.d

    x_22 = x_2 + w_2
    y_22 = y_2 + h_2
    z_22 = z_2 + d_2

    x_box = max(x_1,x_2)
    y_box = max(y_1,y_2)
    z_box = max(z_1,z_2)

    x2_box = min(x_12,x_22)
    y2_box = min(y_12,y_22)
    z2_box = min(z_12,z_22)

    h_box = y2_box - y_box
    w_box = x2_box - x_box
    d_box = z2_box - z_box

    overlap = Box(x_box,y_box,z_box,w_box,h_box,d_box)

    return overlap

def intersection(Box_1,Box_2):

    Box_overlap = overlap_box(Box_1,Box_2)
    
    return (Box_overlap.vol())

def union(Box_1,Box_2):

    vol_1 = Box_1.vol()
    vol_2 = Box_2.vol()
    inter = intersection(Box_1,Box_2)

    return (vol_1+vol_2-inter)

def IOU(Box_1,Box_2):
    
    area_overlap = intersection(Box_1,Box_2)
    area_union = union(Box_1,Box_2)

    return (area_overlap/(area_union+1e-7))