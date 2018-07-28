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
from config import *

def main(args):
  
    with tf.Graph().as_default():
      
        with tf.Session() as sess:
            
            pos_im_path = args.pospath
            des_type = args.descriptor

            dataset = facenet.get_dataset(pos_im_path)
            paths, labels = facenet.get_image_paths_and_labels(dataset)
            
            print('Number of classes: %d' % len(dataset))
            print('Number of images: %d' % len(paths))
            
            # If feature directories don't exist, create them
            if not os.path.isdir(pos_feat_ph):
                os.makedirs(pos_feat_ph)

            print "Calculating the descriptors for the positive samples and saving them"

            nrof_images = len(paths)
            emb_array = np.zeros((nrof_images,))

            for im_path in paths:
                im = imread(im_path, as_grey=True)
                if des_type == "HOG":
                    emb_array = emb_array.append(hog(im, orientations, pixels_per_cell, cells_per_block, visualize, normalize))
                fd_name = os.path.split(im_path)[1].split(".")[0] + ".feat"
                fd_path = os.path.join(pos_feat_ph, fd_name)
                joblib.dump(fd, fd_path)

            print "Positive features saved in {}".format(pos_feat_ph)


            classifier_filename_exp = os.path.expanduser("models\\my_classifier.pkl")

            
            sun=1

            if(sun==1)
                print('Testing classifier')
                with open(classifier_filename_exp, 'rb') as infile:
                    (model, class_names) = pickle.load(infile)

                print('Loaded classifier model from file "%s"' % classifier_filename_exp)

                predictions = model.predict_proba(emb_array)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                
                for i in range(len(best_class_indices)):
                    print('%4d  %s: %.3f' % (i, class_names[best_class_indices[i]], best_class_probabilities[i]))
                    
                accuracy = np.mean(np.equal(best_class_indices, labels))
                print('Accuracy: %.3f' % accuracy)
                
                labels = label_binarize(np.array(labels), classes=range(1,21))
                best_class_indices = label_binarize(np.array(best_class_indices), classes=range(1,21))
                precision, recall, _ = precision_recall_curve(labels.ravel(), best_class_indices.ravel())                
                average_precision = average_precision_score(labels, best_class_indices, average="micro")
                print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision))
                plt.figure()
                plt.step(recall, precision, color='b', alpha=0.2, where='post')
                plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.ylim([0.0, 1.05])   
                plt.xlim([0.0, 1.0])
                plt.title('Average precision score, micro-averaged over all classes: AP={0:0.2f}'.format(average_precision))
           

def parse_arguments(argv):
    
    
    parser = ap.ArgumentParser()
    parser.add_argument('-p', "--pospath", help="Path to positive images",
            required=True)
    parser.add_argument('-d', "--descriptor", help="Descriptor to be used -- HOG",
            default="HOG")
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
