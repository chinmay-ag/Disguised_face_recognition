#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import patch_manager as c_patch
import random
import math
import matplotlib.pyplot as plt
import pylab

def get_image_patch_list(file_list):
    image_dict = {}
    for file_name in file_list:
        subject_no = file_name[:file_name.find('_')]
        image_no = file_name[file_name.find('_')+1:file_name.rfind('_')]
        if subject_no in image_dict:
            if image_no not in image_dict[subject_no]:
                image_dict[subject_no].append(image_no)
        else:
            image_dict[subject_no] = []
            image_dict[subject_no].append(image_no)
    return image_dict
    
def get_gallery_probe_list(file_list,start,end):
    gallery=[]
    probe=[]
    image_dict = get_image_patch_list(file_list)
    for i in range(start,end+1):
        subject_no = 'P'+str(i)
        image_nos = image_dict[subject_no]
        gallery.append(subject_no+'_1')
        image_nos.remove('1')
        random.shuffle(image_nos)
        counter = 0
        while counter < 5:
            gallery.append(subject_no+'_'+image_nos[counter])
            counter += 1
        while counter < len(image_nos):
            probe.append(subject_no+'_'+image_nos[counter])
            counter += 1
    return gallery,probe

file_list = c_patch.get_file_list()
X= c_patch.get_X_list()
Y= c_patch.get_Y_list()

def chi_square(vector1,vector2):
    S=0
    for i in range(len(vector1)):
        temp=(vector1[i]-vector2[i])
        temp=math.pow(temp,2)
        temp/=(vector1[i]+vector2[i])
        S+=temp
        return S

def train_svm(patch_count):
    c_patch.train_svm(patch_count)

def set_gallery_probe():
    global gallery,probe
    gallery,probe = get_gallery_probe_list(file_list,10,20)

def test_gallery_probe():
    len_gallery = len(gallery)
    len_probe = len(probe)    
    #setting the threshold
    thresholds = [0.001,0.005,0.010,0.015,0.025,0.030,0.045]
    FAR_list=[]
    GAR_list=[]
    FAR_list_all=[]
    GAR_list_all=[]
    for threshold in thresholds:
        FA = 0
        GA = 0
        FR = 0
        GR = 0
        FA_all=0
        GA_all=0
        FR_all=0
        GR_all=0
        for i in range(5):
            random.shuffle(gallery)
            random.shuffle(probe)
            for j in range(50):
                rand_gallery_idx = random.randint(0,len_gallery-1)
                rand_probe_idx = random.randint(0,len_probe-1)
                image_dist = 0
                bio_patch_count = 0
                image_dist_all = 0
                #print("hi ",rand_gallery_idx)
                #print("bye ",len_gallery)
                gallery_image_name = gallery[rand_gallery_idx]
                probe_image_name = probe[rand_probe_idx]
                #-----------processing for all 25 patches of the images--------------
                for k in range(1,26):
                    #-----gallery patch--------------
                    gallery_patch_name = gallery_image_name + '_' + str(k) +'\n'
                    gallery_patch_idx = file_list.index(gallery_patch_name)
                    gallery_patch_ite = X[gallery_patch_idx]
                    gallery_patch_lbp = gallery_patch_ite[:256]
                    
                    #----probe patch -----------------
                    probe_patch_name = probe_image_name + '_' + str(k) + '\n'
                    probe_patch_idx = file_list.index(probe_patch_name)
                    probe_patch_ite = X[probe_patch_idx]
                    probe_patch_lbp = probe_patch_ite[:256]
                    
                    #---patch classification-----------------
                    x_test = []
                    x_test.append(gallery_patch_ite)
                    x_test.append(probe_patch_ite)
                    patch_label = c_patch.predict_patches(x_test)
                    
                    #-----------finding chi squrare distance for all patches------------
                    patch_dist = chi_square(gallery_patch_lbp,probe_patch_lbp)
                    image_dist_all += patch_dist
                    
                    #-----finding chi square distance if both patch are biometric---------
                    if patch_label[0]==1 and patch_label[1]==1:
                        bio_patch_count += 1
                        image_dist += patch_dist
                
                #-----------image distance calculated--------------------------
                image_dist = image_dist / bio_patch_count
                if gallery_image_name[:gallery_image_name.find('_')] == probe_image_name[:probe_image_name.find('_')]:
                    if image_dist < threshold:
                        GA += 1
                    else:
                        FR += 1
                else:
                    if image_dist < threshold:
                        FA += 1
                    else:
                        GR += 1
                        
                #all patch calculations-----------------------------------------
                image_dist_all = image_dist_all / 25
                if gallery_image_name[:gallery_image_name.find('_')] == probe_image_name[:probe_image_name.find('_')]:
                    if image_dist_all < threshold:
                        GA_all += 1
                    else:
                        FR_all += 1
                else:
                    if image_dist_all < threshold:
                        FA_all += 1
                    else:
                        GR_all += 1
      
        
        FAR = 100 * (float(FA)/(FA+GR))
        GAR = 100 * (float(GA)/(GA+FR))
        Acc = 100 * (float(GA+GR)/(FA+GA+GR+FR))
        print("threshold : ",threshold," Accuracy : ",Acc)
        FAR_list.append(FAR)
        GAR_list.append(GAR)
        
        
        FAR_all = 100 * (float(FA_all)/(FA_all+GR_all))
        GAR_all = 100 * (float(GA_all)/(GA_all+FR_all))
        #Acc_all = 100 * (float(GA_all+GR_all)/(FA_all+GA_all+GR_all+FR_all))
        FAR_list_all.append(FAR_all)
        GAR_list_all.append(GAR_all)
    
    plt.plot(FAR_list,GAR_list,label="bio-metric patches")
    plt.plot(FAR_list_all,GAR_list_all,label="all patches")
    plt.xscale('log')
    pylab.xlim([0.01,100])
    plt.xlabel("False Accept Rate")
    plt.ylabel("Genuine Accept Rate")
    plt.legend(loc="best")
    plt.show()