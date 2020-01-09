#!/usr/bin/env python
import random
import json
import numpy as np
import argparse
import base64

import aicrowd_helpers
import time
import traceback

import glob
import os
import json


"""
Expected ENVIRONMENT Variables

* AICROWD_TEST_IMAGES_PATH : abs path to  folder containing all the test images
* AICROWD_PREDICTIONS_OUTPUT_PATH : path where you are supposed to write the output predictions.csv
"""

#AICROWD_TEST_IMAGES_PATH = os.getenv('AICROWD_TEST_IMAGES_PATH', 'data/round1')
#AICROWD_PREDICTIONS_OUTPUT_PATH = os.getenv('AICROWD_PREDICTIONS_OUTPUT_PATH', 'random_prediction.csv')
#aicrowd_helpers.execution_error("ee")

def gather_images(test_images_path):
    images = glob.glob(os.path.join(
        test_images_path, "*.jpg"
    ))
    return images

def gather_image_names(test_images_path):
    images = gather_images(test_images_path)
    image_names = [os.path.basename(image_path) for image_path in images]
    return image_names

def get_image_path(image_name):
    test_images_path = os.getenv("AICROWD_TEST_IMAGES_PATH", False)
    return os.path.join(test_images_path, image_name)

def gather_input_output_path():
    test_images_path = os.getenv("AICROWD_TEST_IMAGES_PATH", False)
    assert test_images_path != False, "Please provide the path to the test images using the environment variable : AICROWD_TEST_IMAGES_PATH"

    predictions_output_path = os.getenv("AICROWD_PREDICTIONS_OUTPUT_PATH", False)
    assert predictions_output_path != False, "Please provide the output path (for writing the predictions.csv) using the environment variable : AICROWD_PREDICTIONS_OUTPUT_PATH"

    return test_images_path, predictions_output_path

def get_snake_classes():
    with open('data/class_idx_mapping.csv') as f:
        classes = []
        for line in f.readlines()[1:]:
            class_name = line.split(",")[0]
            classes.append(class_name)
    return classes


def run():
    ########################################################################
    # Register Prediction Start
    ########################################################################
    aicrowd_helpers.execution_start()
    
    os.environ['AICROWD_TEST_IMAGES_PATH']='data/round1'
    os.environ['AICROWD_PREDICTIONS_OUTPUT_PATH'] = 'random_prediction.csv'
    
    ########################################################################
    # Gather Input and Output paths from environment variables
    ########################################################################
    test_images_path, predictions_output_path = gather_input_output_path()

    ########################################################################
    # Gather Image Names
    ########################################################################
    image_names = gather_image_names(test_images_path)

    ########################################################################
    # Do your magic here to train the model
    ########################################################################
	
		
    import numpy as np
    import os
    import glob
    

    from keras.models import load_model
    import efficientnet.keras as efn

    from efficientnet.keras import EfficientNetB5
    model=load_model('model-224-123-16-1.178489-1.705530.hdf5')
    
    counter2 = 0
    df_data = []
    df_count = []
    lim=100000
    lim2=100000
    IMAGE_folder='/home/amaury2_pichat/data/snake/train'
    for folder in os.listdir(IMAGE_folder):
        counter=0
        counter2 = 0
        if len(df_data)<lim:
            for file in os.listdir(IMAGE_folder + '/' + folder):
                if counter2 < lim2:
                    statinfo = os.stat(IMAGE_folder + '/' + folder + '/' + file)
                    if statinfo.st_size!=0 and counter2<lim2:
                        df_data.append((folder, file))
                        counter += 1
                        counter2 += 1
                        
    import pandas as pd
    df_data = pd.DataFrame(df_data, columns=['Folder', 'id'])
    # df_data['Folder']='' + df_data['Folder']
    df_data['Classes'] = df_data['Folder']
    df_data['Class']=df_data['Classes']
    df_data['Path']=IMAGE_folder + '/' + df_data['Folder'] + '/' + df_data['id']
    df_data.drop(['Classes','Folder'],axis=1,inplace=True)

    from keras.preprocessing.image import ImageDataGenerator
    
    valAug = ImageDataGenerator(rescale=1 / 255.0)
    pred_gen = valAug.flow_from_dataframe(df_data,
                                           x_col="Path",
                                           y_col="Class",
                                           shuffle=False,
                                           target_size=(IMAGE_SIZE, IMAGE_SIZE),
                                           batch_size=1,
                                           class_mode='categorical')
                                           
    LINES = []
    
    with open('data/class_idx_mapping.csv') as f:
    	classes = ['filename']
    	for line in f.readlines()[1:]:
    		class_name = line.split(",")[0]
    		classes.append(class_name)
    
    LINES.append(','.join(classes))
    
    predictions = model.predict_generator(pred_gen, verbose=1,steps=len(pred_gen.filenames))
    compteur=0
    for _file_path in pred_gen.filenames:

        file_and_pred=_file_path[_file_path.rfind('\\')+2:]
        LINES.append(_file_path[_file_path.rfind('\\')+2:] + "," + ",".join(str(x) for x in predictions[compteur]))
        compteur += 1
        
    #images_path = AICROWD_TEST_IMAGES_PATH + '/*.jpg'
    #for _file_path in glob.glob(images_path):
    #	probs = softmax(np.random.rand(45))
    #	probs = list(map(str, probs))
    #	LINES.append(",".join([os.path.basename(_file_path)] + probs))
    
    fp = open(AICROWD_PREDICTIONS_OUTPUT_PATH, "w")
    fp.write("\n".join(LINES))
    fp.close()

    ########################################################################
    # Register Prediction Complete
    ########################################################################
    aicrowd_helpers.execution_success({
        "predictions_output_path" : predictions_output_path
    })
    
    


if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        error = traceback.format_exc()
        aicrowd_helpers.execution_error(error)
