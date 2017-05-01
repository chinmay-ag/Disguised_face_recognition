#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import extract_feature

#step 1 -------------------execute extract_features()----------------
txt = """Step 1
Running extract_feature.py
it may take few minutes ...."""

print(txt)

extract_feature.extract_features()

txt="""\n..executed succesfully!!\n\nFollwing files generated in files directory -
X.txt
Y.txt
filelist"""

print(txt)

#step 2 -------------------execute svm_grid_search()-----------------

import svm_grid_search
import test_helper

txt="""\n\nStep 2
Running svm_grid_search.py
it may take few minutes ...."""

print(txt)

gamma,C = svm_grid_search.svm_grid_search()

txt="""\n..executed succesfully!!"""

#step 3-----------------execute train_svm()--------------------------
txt="""\n\nStep 3
Running test_helper.train_svm()
it may take few minutes ...."""

print(txt)
#no of patch used for training the svm
patch_count = 4000
test_helper.train_svm(patch_count)

txt="""\n..executed succesfully!!"""
print(txt)

#step 4-----------------execute set_gallery_probe()--------------------------
txt="""\n\nStep 4
Running test_helper.set_gallery_probe()
it may take few minutes ...."""

print(txt)
#no of patch used for training the svm
test_helper.set_gallery_probe()

txt="""\n..executed succesfully!!"""
print(txt)

#step5-----------------execute test_gallery_probe()-------------------------
txt="""\n\nStep 5
Running test_helper.test_gallery_probe()
it may take few minutes ...."""

print(txt)
#no of patch used for training the svm
test_helper.test_gallery_probe()

txt="""\n..executed succesfully!!"""
print(txt)