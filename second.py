from PIL import Image 
import numpy as np 
import math 
import cv2
import matplotlib.pyplot as plt

Sigma_List = [1,3,5,7,9,11,13,15]
constant  = 1.414 

def Normalize_Image(image_name):
    image = Image.open(image_name)
    image_matrix = np.divide(np.array(image),255)
    return image_matrix
    
def GenerateLoGFilter(Sigma):
    # Best Window Size according to resources is LeastIntegerFunction(6*Sigma)
    Window_Size = math.ceil(6*Sigma)
    Filter_Matrix = np.zeros((Window_Size,Window_Size))
    for X in range(0,Window_Size):
        for Y in range(0,Window_Size):
            i = math.floor(X - (Window_Size/2))
            j = math.floor(Y - (Window_Size/2))
            Filter_Matrix[X][Y] = (-1/(math.pi*math.pow(Sigma,4)))*(1-((math.pow(i,2)+math.pow(j,2))/(2*math.pow(Sigma,2))))*(math.exp((-1)*((math.pow(i,2)+math.pow(j,2))/(2*math.pow(Sigma,2)))))*math.pow(Sigma,2)
    return Filter_Matrix 

def Extract_Blobs(Sigma_List,ImageMatrix):
    total_result = {}
    for i in Sigma_List:
        total_result[i] = [] 
    for sigma in Sigma_List:
        filter = GenerateLoGFilter(sigma)
        convoluted_result = cv2.filter2D(ImageMatrix,-1,filter)
        convoluted_result = np.square(convoluted_result)
        total_result[sigma] = convoluted_result 
    ImportantPoints = []
    for i in range(1,ImageMatrix.shape[0]-2):
        for j in range(1,ImageMatrix.shape[1]-2):
            Sliding_Window = np.zeros((len(Sigma_List),3,3))
            max = -1
            max_sigma = -1
            max_X = -1 
            max_Y = -1
            for k in range(len(Sigma_List)):
                
                Sliding_Window[k] = total_result[Sigma_List[k]][i-1:i+2,j-1:j+2][0]
            for k in range(len(Sigma_List)):
                for l in range(3):
                    for o in range(3):
                        if(Sliding_Window[k][l][o]>max):
                            max = Sliding_Window[k][l][o] 
                            max_sigma = Sigma_List[k] 
                            max_X = i-1+l 
                            max_Y = j-1+o 
            if(max>0.07):
                ImportantPoints.append([max_X,max_Y,max_sigma])
    return ImportantPoints


ImageMatrix = Normalize_Image('images/all_souls_000000.jpg')
Points = Extract_Blobs(Sigma_List,ImageMatrix)
print(len(Points))