#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import test_helper

#step 1-----------------execute train_svm()--------------------------
txt="""Step 1
Running test_helper.train_svm()
it may take few minutes ...."""

print(txt)
#no of patch used for training the svm
patch_count = 4500
test_helper.train_svm(patch_count)

txt="""\n..executed succesfully!!"""
print(txt)

#step 2-----------------execute set_gallery_probe()--------------------------
txt="""\n\nStep 2
Running test_helper.set_gallery_probe()
it may take few minutes ...."""

print(txt)
#no of patch used for training the svm
test_helper.set_gallery_probe()

txt="""\n..executed succesfully!!"""
print(txt)

#step3-----------------execute test_gallery_probe()-------------------------
txt="""\n\nStep 3
Running test_helper.test_gallery_probe()
it may take few minutes ...."""

print(txt)
#no of patch used for training the svm
test_helper.test_gallery_probe()

txt="""\n..executed succesfully!!"""
print(txt)