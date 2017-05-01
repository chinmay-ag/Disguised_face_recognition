DATASET URL :-
		website:- "http://www.iab-rubric.org/resources/facedisguise.html" 
		download link :- "http://www.iab-rubric.org/databases/visible_cropped.zip"
		Due to License agreement clauses we are not uploading actual dataset.

Github link :- 


Steps to run and test disguised Face recognition algorithm:-
-------------------------------------------------------------

1. To execute the complete algorithm and re-create all files/Hyper parameters execute "run_all_code.py" (execution time approx 45 mins)

OR

2. To run the final Disguised Face Recognition test on Gallery & Probe images execute "run_gallery_probe_test.py" (execution time approx 15 min) 


Brief Outline of code Structure:-
---------------------------------
Above two programs use below helper code:-


1. extract_feature.py
	This program will read image intensity pixels from tranining_data/patches folder
	And create files 
		X.txt: contains feature vector of all samples as "list of lists"
		Y.txt: contains label of all samples present in X.txt
		filelist: contains filename of all patches in format "Pi_j_k" , i= subject no, j= image no , k = patch no

2. svm_grid_search.py 
	This program will do grid search on gamma and C. It does 5-fold cross validation and prints the result

3. patch_manager.py to be used by test_images training SVM.
 
4. test_helper.py (following 3 functions are used)
	train_svm() : this method will be called to train the SVM classifer (patch_classifier)

	set_gallery_probe(): this will create gallery and probe image lists to be tested
	//get_gallery_probe_list(file_list,i,j) //it will create gallery and probe list from images subject i to j

	test_gallery_probe() :- It will test the algorithm by randomly selecting one image from gallery and probe each and test (do this multiple times)
	Then it prints accuracy accross various thresholds (set in thresholds list)
	And finally generate ROC curve for FAR vs GAR (for biometric patches and all patches)

 
NOTE:- other helper programs are kept in lib and util folders
some of the important lib programs are-

Hist.py :- input: patch intensity vector
	   output: list of 256 features (representing intesity histogram)

lbp.py :- input: patch intensity vector
	  output: list of 256 features (representing lbp histogram)
   
