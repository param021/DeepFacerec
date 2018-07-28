## Installation Guide
1. Development environment setup (Windows 8.1):
    * Install Python 3.5.5
    * [Install Tensorflow version 1.8.0 (CPU)](https://www.tensorflow.org/install/install_windows)
    * [OR] [Install Tensorflow version 1.8.0 (GPU)](https://www.tensorflow.org/install/install_windows)
        * [CUDA® Toolkit 9.0](http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows)
        * [cuDNN v7.0](https://developer.nvidia.com/cudnn)
    * External Packages Required:
        * scipy
        * scikit-learn
        * opencv-python
        * H5py
        * numpy
        * matplotlib
        * Pillow
        * Requests
        * pickle
        * psutil
        * shutil  
2. Capturing Images:
    * Run *capture_im.py*
    [Remove meta.csv and WebcamCap.txt before proceeding to train/classify]
    * Using: 
        ```python
        filepath = ' ..\datasets\dataset_name\raw\'
        fileName = ' ..\datasets\dataset_name\raw'+'\WebcamCap.txt'
        ```
3. Organization of Dataset:
      * The raw images are organized as:
        ``` datasets\dataset_name\raw\subject_name ```
        ``` 
        gtfd\raw\person01
              ..\person02
        ```
         
     * Each subject_name folder contains the images.
        ``` 
        gtfd\raw\person1\image0
                      ..\image1
        ```
4. Download and save facenet pre-trained models as:
    ``` ..\models\facenet\20180408-102900\ ```
    * [20180408-102900](https://drive.google.com/open?id=1R77HmFADxe87GmoLwzfgMu_HY0IhcyBz)
    * [20180402-114759](https://drive.google.com/open?id=1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-)

5. Alignment of dataset using MTCNN:
    * ```SET PYTHONPATH=%PYTHONPATH%;'..\facenet\src' ```
    * Run *align_dataset_mtcnn.py*
      ``` 
        python align\align_dataset_mtcnn.py 
        ..\facenet\datasets\dataset_name\raw        
        ..\facenet\datasets\dataset_name\aligned
        --image_size 182  
        --margin 44 
        --random_order
        --gpu_memory_fraction 0.85 
        ```
6. Split Dataset to Test and Train:
    * Run *test_train_split.py*
    * Set the ratio of test data and directory:
        ```python 
        percent = 0.8
        add = ‘directory of aligned dataset’
        ```
7. Train a classifier on the training data:
    * ```SET PYTHONPATH=%PYTHONPATH%;'..\facenet'```
    * Run *classifier_svm.py* or *classifier_mlp.py*
        ```
        python src\classifier_mlp.py
        TRAIN
        datasets\dataset_name\aligned\Train models\facenet\20180408-102900\20180408-102900.pb       
        models\my_classifier.pkl 
        --batch_size 500
        ```
8. Test the classifier on testing data:
    * ```SET PYTHONPATH=%PYTHONPATH%;'..\facenet'```
    * Run *classifier_svm.py* or *classifier_mlp.py*
        ```
        python src\classifier_mlp.py
        CLASSIFY
        datasets\dataset_name\aligned\Test models\facenet\20180408-102900\20180408-102900.pb       
        models\my_classifier.pkl 
        --batch_size 500
        ```
9. To run Real-time face recognition instead:
    * ```SET PYTHONPATH=%PYTHONPATH%;'..\facenet\realtime_facenet'```
    * Run *start_realtime.py*
    * Update classifier file name, model directory and dataset directory in *start_realtime.py*
