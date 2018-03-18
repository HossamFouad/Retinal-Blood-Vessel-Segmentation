# Retinal-Blood-Vessel-Segmentation
Segmentation of retinal blood vessels using CNN and random forest machine learning techniques
This project is based on  Wang, S., Yin, Y., Cao, G., Wei, B., Zheng, Y., Yang, G.: Hierarchical retinal blood vessel segmentation based on feature and
ensemble learning, Neurocomputng, Vol.149, 2015.

The project includes the following:
[1] imgproc.m : get train and test data batches from original images
[2] preprocessing.py: read image paths from CSV files, shuffle data. It also includes batches load functions
[3] Model.py: includes CNN architecture
[4] feat_gen.py: generate features from CNN layers and save them as Mat files
[5] random_forest.py: classify features extracted from last layer
