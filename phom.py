from matplotlib import image as im
from matplotlib import pyplot as plt
import gzip
import numpy as np
import math


idimx=28 #Max x dimension of image (pixels going down)
idimy=28 #Max y dimension of image (pixels going across)

def formpho(fname):
    """Formats a raw photo for use in the neural network 
    """
    
    rpic = im.imread(fname)
    
    x=len(rpic)
    y=len(rpic[0])
    
    if len(rpic.shape)==3:
        rpic = rgb2gray(rpic)
        
    if x > idimx and y > idimy:
        rpic = rpic[:x-x%idimx,:y-y%idimy]
        fpic = rpic[::x//idimx,::y//idimy]   
        
    elif x < idimx and y < idimy:
        temp = np.zeros((idimx,idimy))
        temp[:,:]=255
        xpos = math.floor(0.5*(idimx-x))
        ypos = math.floor(0.5*(idimy-y))
        temp[xpos:xpos+x,ypos:ypos+y]=rpic
        fpic=temp
        
    return fpic


def rgb2gray(rgb):
    """Converts an rgb photo to grayscale
    """
    
    r = rgb[:,:,0]
    g = rgb[:,:,1]
    b = rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    
    return gray


def showGrayScale(pic):
    """Prints the grayscale image to the terminal
    """
    
    plt.imshow(pic,cmap='gray',vmin=0, vmax=255)
    plt.show()
    
    
def getGzipped(im_path, la_path, num_images=None):
    """Takes the path to the images, path to the labels and an optional number 
    of images to load.Strictly reads the gz files used int the number 
    recognition neural network. These images are (28x28) pixels and are 
    grayscale. Returns a list of numpy arrays containing the images and a 
    numpy array of the correct labels
    """
    
    f = gzip.open(im_path,'r')

    image_size = 28
    images = []
    labels = []
    
    f.read(16)
    if num_images is None:
        buf = f.read()
        num_images = int(len(buf)/(image_size * image_size))
    else:
        buf = f.read(image_size * image_size * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    data = data.reshape(num_images, image_size, image_size, 1)
    for val in data:
        images.append(val.squeeze()/255)    #REMINDER: Squeeze removes the unused 
                                        #          3rd dimension
    
    
    f = gzip.open(la_path,'r')
    f.read(8)
    for i in range(0,len(images)):   
        buf = f.read(1)
        labels.append(int.from_bytes(buf, "big"))
        
    return images, labels













    
    