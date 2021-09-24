import sys, cv2
import os, os.path
import numpy as np



if len(sys.argv) == 2:
    folder = os.path.join(os.getcwd(),sys.argv[1])
    print(folder)
    #length = len([name for name in os.listdir(folder) if os.path.isfile(name)])
    length = 300
    
    orig = cv2.imread(folder + "0.png")
    height, width, layers = orig.shape
    size = (width,height)

    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out = cv2.VideoWriter(folder + 'project.mp4',fourcc, 15, size)

    print(length)
    for i in range(1,length):
        img = cv2.imread(folder + str(i)+'.png')
        if img is None:
            break
        frame = img*0.5 + orig*0.5
        out.write(np.uint8(frame))
    
    out.release()



    

print("Done")