import numpy as np
from PIL import Image
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.patches import Circle, Polygon
import math
import cv2
from pathlib import Path
import imutils

def select_four_points(filename):
    image = Path(filename)
    img = Image.open(image)
    plt.imshow(img)     
    Xi = plt.ginput(4)
    Xi = list(Xi)
    Xi = list(map(list,Xi))

    return Xi, img

def coordinates_world(Xi):
    dist = math.dist(Xi[0],Xi[1])
    world = [[Xi[0][0],Xi[0][1]]]
    world.append([Xi[0][0]+dist,Xi[0][1]])
    world.append([Xi[0][0],Xi[1][1]])
    world.append([Xi[0][0]+dist,Xi[1][1]])

    return world
def show_points(Xi,Xw,img):
    color = 'yellow'
    fig, ax = plt.subplots(1,2, figsize=(15, 10), dpi = 80)
    p1 = Circle((Xi[0][0],Xi[0][1]), 10, facecolor = color)
    p2 = Circle((Xi[1][0],Xi[1][1]), 10, facecolor = color)
    p3 = Circle((Xi[2][0],Xi[2][1]), 10, facecolor = color)
    p4 = Circle((Xi[3][0],Xi[3][1]), 10, facecolor = color)

    p1_w = Circle((Xw[0][0],Xw[0][1]), 10, facecolor = color)
    p2_w = Circle((Xw[1][0],Xw[1][1]), 10, facecolor = color)
    p3_w = Circle((Xw[2][0],Xw[2][1]), 10, facecolor = color)
    p4_w = Circle((Xw[3][0],Xw[3][1]), 10, facecolor = color)

    ax[0].add_patch(p1)
    ax[0].add_patch(p2)
    ax[0].add_patch(p3)
    ax[0].add_patch(p4)
    rect= [Xi[0], Xi[2], Xi[3], Xi[1]]
    ax[0].add_patch(Polygon(rect,facecolor='none', edgecolor='r'))
    ax[0].imshow(img);

    ax[1].add_patch(p1_w)
    ax[1].add_patch(p2_w)
    ax[1].add_patch(p3_w)
    ax[1].add_patch(p4_w)
    rect_w= [Xw[0], Xw[2], Xw[3], Xw[1]]
    ax[1].add_patch(Polygon(rect_w,facecolor='none', edgecolor='w'))
    ax[1].imshow(np.ones((np.shape(img)[0], np.shape(img)[1])));
    plt.show()

def calculate_homography(Xi,Xw):
    """Image coordinates(Xi)
    World Plane coordinates (Xw)"""
    n = len(Xi)
    h = np.zeros((8,1))
    H = np.zeros((3,3)) #Create a 3x3 matrix
    A = np.zeros((2*n,8))
    B = np.zeros((2*n,1))

    for a in range(n):
        A[2*a] = [Xw[a][0], Xw[a][1], 1, 0, 0, 0, -Xi[a][0]*Xw[a][0], -Xi[a][0]*Xw[a][1]]
        A[(2*a)+1] = [0, 0, 0, Xw[a][0], Xw[a][1], 1, -Xi[a][1]*Xw[a][0], -Xi[a][1]*Xw[a][1]]
       
    for b in range(n):
        B[2*b] = Xi[b][0]
        B[(2*b)+1] = Xi[b][1]

    #Least square estimates
    A_transposed = A.transpose()
    A_temp = np.matmul(A_transposed,A)
    inv_mat = np.linalg.inv(A_temp)
    A_temp2 = np.matmul(inv_mat,A_transposed)
    h = np.matmul(A_temp2,B)

    #mapping
    H[0,0] = h[0]
    H[0,1] = h[1]
    H[0,2] = h[2]
    H[1,0] = h[3]
    H[1,1] = h[4]
    H[1,2] = h[5]
    H[2,0] = h[6]
    H[2,1] = h[7]
    H[2,2] = 1
    
    return H

if __name__ == "__main__":
    images = ["1.jpg","buco.jpg","build0.jpg","casaconOcclusion.jpg"]
    filename = images[1]
    Xi, img = select_four_points(filename) # selecting the four points in the image
    
    width = np.shape(img)[1] 
    height = np.shape(img)[0]
    Xw = coordinates_world(Xi) # getting the world coordinates

    #show_points(Xi,Xw,img) #showing the selected points on the image 
    
    H = calculate_homography(Xi,Xw) #calculating the homography to remove projective distortion
    H_inv=np.linalg.pinv(H) #inverse of H
    temp = cv2.imread(filename)
    output=cv2.warpPerspective(temp, H_inv , (width,height))

    #Save the output image
    if (Xi[0][1]>Xi[1][1]):
        output_rot = imutils.rotate(output, angle=180)
        cv2.imwrite(filename.replace(".jpg","_Output.jpg") , output_rot )
    else:
        cv2.imwrite(filename.replace(".jpg", "_Output.jpg") , output)


    