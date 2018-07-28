from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import os
import sys
import math
import pickle
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score
import matplotlib.pyplot as plt

from skimage.feature import local_binary_pattern
from skimage.feature import hog
from skimage.io import imread
from sklearn.externals import joblib
import argparse as ap
import glob
import os

def main(args):
      
    with tf.Graph().as_default():
      
        with tf.Session() as sess:
            
            pos_im_path = args.pospath
            des_type = args.descriptor

            dataset = facenet.get_dataset(pos_im_path)
            paths, labels = facenet.get_image_paths_and_labels(dataset)
            
            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))
            
            pos_feat_ph="D:\\facenet\\descriptors\\new"
            
            # If feature directories don't exist, create them
            if not os.path.isdir(pos_feat_ph):
                os.makedirs(pos_feat_ph)

            print("Calculating the descriptors for the positive samples and saving them")

            nrof_images = len(paths)
            
            for im_path in paths:
                im = imread(im_path, as_grey=True)
                if des_type == "HOG":
                    emb_array= hog(im, orientations=8, pixels_per_cell=(16, 16),cells_per_block=(1, 1))
                fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
                fd_path = os.path.join(pos_feat_ph, fd_name)
                joblib.dump(emb_array, fd_path)
                
            #Directory for saving embeddings
            embfilename='fnet1svc'

            if not os.path.exists('D:\\facenet\\descriptors\\'+embfilename):
                os.mkdir('D:\\facenet\\descriptors\\'+embfilename)
            np.savetxt('D:\\facenet\\descriptors\\'+embfilename+'\\log1.gz', emb_array, fmt='%.32f', delimiter=',', newline='\n')
            print('Saved feature embeddings to file "%s"' % embfilename)
            
            print("Positive features saved in {}".format(pos_feat_ph))

            fds = []
           
            
            for feat_path in glob.glob(os.path.join(pos_feat_ph,"*.feat")):
                fd = joblib.load(feat_path)
                fds.append(fd)
                
            clf = SVC(kernel='linear', probability=True)
            print ("Training a Linear SVM Classifier")
            clf.fit(fds, labels)
            # If feature directories don't exist, create them
            if not os.path.isdir(os.path.split(model_path)[0]):
                 os.makedirs(os.path.split(model_path)[0])
            joblib.dump(clf, model_path)
            print ("Classifier saved to {}".format(model_path))
            
def parse_arguments(argv):
        
    parser = ap.ArgumentParser()
    parser.add_argument('-p', "--pospath", help="Path to positive images",
            required=True)
    parser.add_argument('-d', "--descriptor", help="Descriptor to be used -- HOG",
            default="HOG")
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
