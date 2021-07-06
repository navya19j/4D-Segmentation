import numpy as np
from PIL import Image, ImageSequence
import sys
from tifffile import imsave


np.set_printoptions(threshold=sys.maxsize)

def get_binary_image():

    path_dir = os.getcwd() + "/Data_Scaled/" + cell
    path_fin = os.getcwd() + "/Data_Scaled_Reized/"+ cell
    for j in range (0,200):
        if (j!=66) and (j!=164):
            if (int(j/10)==0):
                path_dir_1 = path_dir + "/" + cell + "_T00" + str(j) + ".tif"
                path_fin_1 = path_fin + "/" + cell + "_T00" + str(j) + ".tif"
            elif (int(j/100) == 0):
                path_dir_1 = path_dir + "/" + cell + "_T0" + str(j) + ".tif"
                path_fin_1 = path_fin + "/" + cell + "_T0" + str(j) + ".tif"
            else:
                path_dir_1 = path_dir + "/" + cell + "_T" + str(j) + ".tif"
                path_fin_1 = path_fin + "/" + cell + "_T" + str(j) + ".tif"
    
    img = Image.open(path_dir_1)
    images = []
    for i in range (335,463):
        img.seek(i)
        x = img.resize((128,128),resample=Image.NEAREST)
        images.append(np.array(x))
    images = np.array(images)
    print(np.shape(images))
    #np.pad(images, ())
    imsave(path_fin_1,images)
            
    return 0

get_binary_image()