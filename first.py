from PIL import Image 
import numpy as np 
from datetime import datetime 
import os 
import pickle 

def Similarity(corr1,corr2,distance):
    sum = 0 
    for d in distance:
        for i in range(0,64):
            sum = sum + (abs(corr1[d][i]-corr2[d][i])/(1+corr1[d][i]+corr2[d][i]))
    return sum


            
    
def Image2Matrix(image_name):
    image = Image.open(image_name)
    resized_image = image.resize((image.size[0]//4,image.size[1]//4))
    quantized_image = resized_image.quantize(colors=64)
    return np.array(quantized_image)


def GetHistogram(Matrix):
    hist_dict = {}
    for i in range(0,64):
        hist_dict[i] = 0

    for i in Matrix:
        for j in i:
            hist_dict[int(j)]+=1
    return hist_dict
    

def ChessBoard(lis1,lis2):
    number1 = abs(lis1[0]-lis2[0])
    number2 = abs(lis1[1]-lis2[1])
    big = number1 
    if(number2>number1):
        big = number2 
    return big
    
def Matrix2Vector(Matrix):
    AutoCorr = {}
    ColorHistogram = GetHistogram(Matrix)
    distance = [1,3,5,7]
    for d in distance:
        AutoCorr[d] = {}
        for color in range(0,64):
            AutoCorr[d][color] = 0 
    for d in distance:
        for X in range(len(Matrix)):
            for Y in range(len(Matrix[0])):
                colour = Matrix[X][Y]
                for i in range(X-d,X+d):
                    new_x = i 
                    new_y = Y+d  
                    if(new_x>=0 and new_x<len(Matrix) and new_y>=0 and new_y<len(Matrix[0])):
                        if(Matrix[new_x][new_y]==colour):
                            AutoCorr[d][colour]+=1
                for i in range(X-d,X+d):
                    new_x = i 
                    new_y = Y-d  
                    if(new_x>=0 and new_x<len(Matrix) and new_y>=0 and new_y<len(Matrix[0])):
                        if(Matrix[new_x][new_y]==colour):
                            AutoCorr[d][colour]+=1
                for i in range(Y-d+1,Y+d-1): 
                    new_x = X+d 
                    new_y = i 
                    if(new_x>=0 and new_x<len(Matrix) and new_y>=0 and new_y<len(Matrix[0])):
                        if(Matrix[new_x][new_y]==colour):
                            AutoCorr[d][colour]+=1
                for i in range(Y-d+1,Y+d-1): 
                    new_x = X-d 
                    new_y = i 
                    if(new_x>=0 and new_x<len(Matrix) and new_y>=0 and new_y<len(Matrix[0])):
                        if(Matrix[new_x][new_y]==colour):
                            AutoCorr[d][colour]+=1
    for d in AutoCorr:
        for color in AutoCorr[d]:
            if(ColorHistogram[color]!=0):
                AutoCorr[d][color] = AutoCorr[d][color]/(ColorHistogram[color]*8*d) 
    return AutoCorr
    
    
# count = 0

# for (root,dirs,files) in os.walk('images/'):
#     for file in files:
#         inner_file = open('first_vectors/'+file.rstrip('.jpg'),'wb')
#         file_name = root+file 
#         ImageMatrix = Image2Matrix(file_name)
#         AutoCorr = Matrix2Vector(ImageMatrix)
#         pickle.dump(AutoCorr,inner_file) 
#         count+=1 
#         print("Done with file number "+str(count))


def Test(image_name):
    ImageMatrix = Image2Matrix(image_name)
    AutoCorr = Matrix2Vector(ImageMatrix)
    result = {}
    for (root,dirs,files) in os.walk('first_vectors'):
        for file in files:
            inner_file = open(root+'/'+file,'rb')
            vector = pickle.load(inner_file)
            similarity = Similarity(AutoCorr,vector,[1,3,5,7])
            result[similarity] = file 
    answer = []
    count = 0
    for i in sorted(result.keys()):
        if(result[i]!=image_name.split("/")[1].rstrip('.jpg')):
            count+=1 
            answer.append(result[i]) 
            if(count==100):
                break 
    return answer 



def Query(file_name,ground_truth):
    line = open(file_name).read()
    line = line.strip()
    image_name = line.split(" ")[0]
    image_name_list =  image_name.split("_")[1:]
    image_name = ""
    for i in image_name_list:
        image_name = image_name+i+"_"
    image_name = image_name.rstrip("_")
    image_name = 'images/'+image_name+".jpg"
    Results = Test(image_name)
    Precision = 0 
    Recall = 0
    for i in Results:
        if(i in ground_truth):
            Precision+=1
            Recall+=1
    Precision = Precision/100
    Recall = Recall/len(ground_truth)
    return Precision,Recall 


def GroundTruth(query_name):
    lis = []
    for (root,dirs,files) in os.walk('train/ground_truth'):
        for file in files:
            if(file.startswith(query_name)):
                file_handle = open(root+"/"+file)
                inner_lis = file_handle.readlines()
                inner_lis = [i.strip() for i in inner_lis]
                lis.extend(inner_lis)
    return lis

Average_Precision = 0
Average_Recall = 0
Maximum_Precision = 0
Maximum_Recall = 0 
Minimum_Recall = 100
Minimum_Precision = 100
count = 0
for (root,dirs,files) in os.walk('train/query'):
    for file in files:
        count+=1
        inner_file = root+'/'+file 
        random = inner_file.split("/")[2]
        inner_random = random.split("_query")[0]
        ground_truth = GroundTruth(inner_random)
        Precision,Recall = Query(inner_file,ground_truth)
        print(Precision)
        print(Recall)
        Average_Precision+=Precision
        if(Precision>Maximum_Precision):
            Maximum_Precision = Precision
        if(Precision<Minimum_Precision):
            Minimum_Precision = Precision 
        Average_Recall+=Recall
        if(Recall>Maximum_Recall):
            Maximum_Recall = Recall
        if(Recall<Minimum_Recall):
            Minimum_Recall = Recall 

print("Maximum Recall "+str(Maximum_Recall))
print("Minimum Recall "+str(Minimum_Recall))
print("Maximum Precision "+str(Maximum_Precision))
print("Minimum Recall "+str(Minimum_Precision))
print("Average Recall "+str(Average_Recall))
print("Average Precision "+str(Average_Precision))