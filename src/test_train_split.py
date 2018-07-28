import os
import shutil
import random

add = 'D:\\facenet\\datasets\\gt_db\\aligned' # Address of the main folder
train = add + '\\Train' # Train folder
test = add + '\\Test' # Test folder
all_folders = [fn for fn in os.listdir(add) if not fn.endswith('txt')]

percent = 0.9 #Test percentage

if not os.path.exists(test):    
    os.mkdir(test)
if not os.path.exists(train):    
    os.mkdir(train)

for folder in all_folders:
    add1=add+'\\'+folder
    all_images = os.listdir(add1)
    h = int(len(all_images)*percent) # Length of test percent
    random.shuffle(all_images)
    
    os.mkdir(test+'\\'+folder)
    destination = test+'\\'+folder
    for i in range(0,h):
    	destination = test+'\\'+folder
    	shutil.copy(add1+'\\'+all_images[i],destination)

    os.mkdir(train+'\\'+folder)
    destination = train+'\\'+folder
    for i in range(h,len(all_images)):
    	destination = train+'\\'+folder
    	shutil.copy(add1+'\\'+all_images[i],destination)    