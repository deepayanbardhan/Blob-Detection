#Deepayan Bardhan (200266399)

import cv2
import numpy as np
import math
import time

# Returns the 'same' size of convoluted image
def convolution(img,kernel): #send 2D matirx as img
    himg=img.shape[0]
    wimg=img.shape[1]
    img=np.asarray(img,dtype=np.int)
    kernel=np.asarray(kernel)
    hker=kernel.shape[0]//2
    wker=kernel.shape[1]//2
    padimg=np.zeros((himg+2*hker,wimg+2*wker))
    padimg[hker:-hker,wker:-wker]=img
    temp=kernel.shape[0]*kernel.shape[1]
    
    img_conv=np.zeros(img.shape)
    
    for i in range(hker,himg+hker):
        for j in range(wker,wimg+wker):
            x=padimg[i-hker:i+hker+1,j-wker:j+wker+1]
            t=np.sum(np.dot(np.reshape(x,(1,temp)),np.transpose(np.reshape(kernel,(1,temp)))))
            img_conv[i-hker][j-wker]=t*t
    
    mi=np.min(img_conv)
    ma=np.max(img_conv)
    
    for i in range(int(img_conv.shape[0])):
        for j in range(int(img_conv.shape[1])):
            img_conv[i][j]=(img_conv[i][j]-mi)*255.0/(ma-mi)
    
    return img_conv


def genlog(sigma, size):
    o=sigma
    if size%2==0:
        size=size+1
    log=np.zeros((size,size))
    i=-1*(size//2)
    for x in range(size):
        j=-1*(size//2)
        for y in range(size):
            log[x][y]= (-1.0)*(2*o**2-(i**2+j**2))*math.exp(-(i**2+j**2)/(2*o**2))
            j=j+1
        i=i+1
    log1=(log*500)
    return log1


start=time.time()  # start of timer


path='D:\\deepayan\\study\\study\\ECE 558\\project\\TestImages4Project\\sunflowers.jpg'  #path of the image
img=cv2.imread(path,0)  
copy=cv2.imread(path)

layers=9
sig=2
k=1.25

arrimg=[]
for i in range(layers):
    ker=genlog(sig,int(6*sig)+1)
    name='sigma='+str(sig)+' kernel_size='+str(int(6*sig)+1)+'.png'
    print name
    sig=sig*k
    p=convolution(img,ker)
    q=np.asarray(p,dtype=np.uint8)
    arrimg.append(q)
    #cv2.imwrite(name,q)
    print "layer ", (i+1), " convolution done ", (layers-i-1)," left"

"""
#To display the convloluted iamges
for i in range(layers):
    cv2.imshow(str(i),arrimg[i])
"""
threshold=0.1
count=0
sig=2
#Non-Max Supression
print "Convolution finished \nNMS started"
rep=[]
for layer in range(1,layers):
    r=sig*np.sqrt(4)
    sig=sig*k
    for x in range(1,int(img.shape[0])-1):
        for y in range(1,int(img.shape[1])-1):
            pix=arrimg[layer][x][y];
            if pix==0 or pix<=int(threshold*255):
                continue
            flag=1
            for i in range((0,-1)[layer==0],(1,2)[layer==layers]):
                for j in range(-1,2):
                    for kk in range (-1,2):
                        if pix<=arrimg[layer+i][x+j][y+kk] and (i!=0 or j!=0 or kk!=0):
                            flag=0
            if flag==1:
                c=(x,y)
                if c in rep:
                    continue
                else:
                    cv2.circle(copy,(y,x),int(r),(0,0,255),1);
                    rep.append(c)
                    count=count+1

print "Number of Maximas detected=",count

cv2.imshow('Blobs detected=',copy)
#cv2.imwrite('final.png',copy)
end=time.time();
print "running time=",(end-start),"seconds"

cv2.waitKey(0)



