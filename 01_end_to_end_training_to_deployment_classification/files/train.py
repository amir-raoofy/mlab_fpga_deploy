import os
import sys
import argparse
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_hub as hub
from pathlib import Path
import pandas as pd
import glob
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.utils import class_weight
import multiprocessing
import numpy as np
import json as json
from tensorflow.python.tools import freeze_graph
from tensorflow.keras.utils import plot_model, to_categorical

#import keras
#config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 56} ) 
#sess = tf.Session(config=config) 
#keras.backend.set_session(sess)

# Silence TensorFlow messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# workaround for TF1.15 bug "Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


class cfg:
    weights = {
        "Agriculture":                                               67256 ,
        "Airports": 0,
        "Annual crops associated with permanent crops" : 0,
        "Bare rock": 0,
        "Beaches, dunes, sands":0,
        "Burnt areas":0,
        "Coastal lagoons":0,
        "Construction sites": 0,
        "Continuous urban fabric": 0,
        "Discontinuous urban fabric":0,
        "Estuaries":0,
        "Fruit trees and berry plantations":0,
        "Green urban areas":0,
        "Land principally occupied by agriculture, with significant areas of natural vegetation":0,
        "Mineral extraction sites":0,
        "Inland marshes":0,
        "Agro-forestry areas":					  15790 ,
        "Arable land":						 100394 ,
        "Beaches,dunes,sands":					   1193 ,
        "Broad-leaved forest":					  73407 ,
        "Coastal wetlands":						   1033 ,
        "Complex cultivation patterns":				  53530 ,
        "Coniferous forest":					  86569 ,
        "Dump sites":0,
        "Industrial or commercial units":				   6182 ,
        "Inland waters":						  35349 ,
        "Inland wetlands":						  11620 ,
        "Intertidal flats":0,
        "Marine waters":						  39110 ,
        "Mixed forest":						  91926 ,
        "Moors and heathland":0,
        "Natural grassland":0,
        "Moors,heathlandandsclerophyllousvegetation":		   8434 ,
        "Naturalgrasslandandsparselyvegetatedareas":		   6663 ,
        "Non-irrigated arable land":0,
        "Olive groves":0,
        "Peatbogs":0,
        "Pastures":						  50977 ,
        "Permanent crops":				  15862 ,
        "Port areas":0,
        "Rice fields":0,
        "Road and rail networks and associated land":0,
        "Permanently irrigated land":0,
        "Salt marshes":0,
        "Salines":0,
        "Transitional woodland/shrub":				  77589 ,
        "Sclerophyllous vegetation":0,
        "Sea and ocean":0,
        "Urbanfabric":					  38779,
        "Sport and leisure facilities":0,
        "Sparsely vegetated areas":0,
        "Vineyards":0,
        "Water bodies":0,
        "Water courses":0
}



def train(input_height,input_width,input_chan,epochs,learnrate, \
          batchsize,output_ckpt_path,infer_graph_path,tboard_path):

    # Set up directories and files
    INFER_GRAPH_DIR = os.path.dirname(infer_graph_path)
    INFER_GRAPH_FILENAME =os.path.basename(infer_graph_path)

    def get_data(path_images):
        classes_arr = []
        files = glob.glob(str(path_images) + "/*/" + '*.tif')

        for filename in files:
            out = Path(filename)
            parent = out.resolve().parent
            label_file = glob.glob(str(parent) + '/*.json')[0]

            with open(label_file) as json_file:
                data = json.load(json_file)
            name = data['labels']
            classes_arr.append([str(out), name])
        
        return pd.DataFrame(classes_arr, columns=['files', 'labels'])

    # load data paths to dictionaries.
    #root_path = "/root/tensorflow_datasets/"
    root_path = "/workspace/data/"
    dict_train = get_data(root_path+"train/")
    dict_test = get_data(root_path+"test/")
    dict_val = get_data(root_path+"val/")

    mlb = MultiLabelBinarizer()
    mlb.fit(dict_train['labels'])
    print("Labels:")
    for (i, label) in enumerate(mlb.classes_):
        print("{}. {}".format(i, label))
    

    # Create Data Iterators
    #Data generator for training data
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_dataframe(
        dict_train,
        x_col='files',
        y_col='labels',
        target_size=(input_height,input_width),
        batch_size=batchsize,
        class_mode='categorical',
    )
    ### Now compute class weights from this

    import sys
    weight_sum = sum([cfg.weights[x] for x in cfg.weights])
    print("Total Weight: %s" % weight_sum)
    weights = {y:(cfg.weights[x] / weight_sum) for y,x in enumerate(mlb.classes_)}

    #Data generator for test data
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    test_generator = test_datagen.flow_from_dataframe(
        dict_test,
        x_col='files',
        y_col='labels',
        target_size=(input_height,input_width),
        batch_size=batchsize,
        class_mode='categorical',
        shuffle=False
    )

    #Data generator for validation data
    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_generator = validation_datagen.flow_from_dataframe(
        dict_val,
        x_col='files',
        y_col='labels',
        target_size=(input_height,input_width),
        batch_size=batchsize,
        class_mode='categorical',
        shuffle=False
    )

    # get number of cores
    num_cpus = multiprocessing.cpu_count()
    print("Number of cores: %s" %num_cpus)


    # Create a callback for storing the model. This will be used for quantization later
    fname = os.path.sep.join([INFER_GRAPH_DIR, "best-model.ckpt"])
    checkpoint = tf.keras.callbacks.ModelCheckpoint(fname,
                                monitor="val_loss", mode="min",
                                #monitor="val_acc", mode="max",
                                save_best_only=True, verbose=1)

    # Load Model from Google
    hub_layer = hub.KerasLayer("https://tfhub.dev/google/remote_sensing/bigearthnet-resnet50/1", trainable=False)
    # Add Layers to predict our dataset
    model = tf.keras.Sequential()
    model.add(hub_layer)
    model.add(tf.keras.layers.Dense(43, activation='sigmoid'))
    # Compile new model
    model.compile(
            loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.SGD(learning_rate=learnrate, momentum=0.9),
            metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()] 
    )

    # Training phase with training data
    print("\n----------------------------",flush=True)
    print(" TRAINING STARTED...",flush=True)
    print("----------------------------",flush=True)

    history = model.fit(
            train_generator,
            epochs=epochs,
            validation_steps=validation_generator.samples/validation_generator.batch_size,
            steps_per_epoch=train_generator.samples/train_generator.batch_size,
            validation_data=validation_generator,
            callbacks=[checkpoint], #checkpoint = ModelCheckpoint(fname
            workers=num_cpus,
            #class_weight=weights,
            verbose=1
    )

    # save post-training checkpoint, this saves all the parameters of the trained network
    print("\n----------------------------",flush=True)
    print(" SAVING CHECKPOINT & GRAPH...",flush=True)
    print("----------------------------",flush=True)

    #print the CNN structure
    model.summary()
    plot_model(model, to_file=INFER_GRAPH_DIR+"/network.png", show_shapes=True)

    # Check the input and output name
    print ("\n TF input node name:")
    print(model.inputs)
    print ("\n TF output node name:")
    print(model.outputs)

    fname = os.path.sep.join([INFER_GRAPH_DIR, "final-model.ckpt"])
    tf.keras.experimental.export_saved_model(model, fname)

    freeze_graph.freeze_graph(None,
                          None,
                          None,
                          None,
                          model.outputs[0].op.name,
                          None,
                          None,
                          os.path.join(INFER_GRAPH_DIR, "frozen_model.pb"),
                          False,
                          "",
                          input_saved_model_dir='build/chkpts/final-model.ckpt/')


    #tf.keras.backend.set_learning_phase(0)

    # fetch the tensorflow session using the Keras backend
    #tf_session = tf.keras.backend.get_session()
    # write out tensorflow checkpoint & meta graph
    #saver = tf.compat.v1.train.Saver()
    #save_path = saver.save(tf_session, output_ckpt_path)
    #print (' Checkpoint created :', output_ckpt_path)

    # set up tensorflow saver object
    #saver = tf.compat.v1.train.Saver()
    #sess = tf.compat.v1.keras.backend.get_session()
    #saver.save(sess, output_ckpt_path)
    #model.save(filepath= output_ckpt_path)
    #tf.keras.models.save_model(model, filepath=INFER_GRAPH_DIR, save_format='tf')
    #with tf.compat.v1.Graph().as_default():
        # define placeholders for the input data
        #x_1 = tf.compat.v1.placeholder(tf.float32, shape=[None,input_height,input_width,input_chan], name='input_1')

        # call the CNN function with is_training=False
        #customcnn = tf.keras.Model(inputs=model.inputs,outputs=model.get_layer(name="dense").output)
        #logits_1 = customcnn(cnn_in=x_1, is_training=False)
        
    #    tf.io.write_graph(tf.compat.v1.get_default_graph().as_graph_def(), INFER_GRAPH_DIR, INFER_GRAPH_FILENAME, as_text=False)
    #    print(' Saved binary inference graph to %s' % infer_graph_path)

    #sess = tf.compat.v1.keras.backend.get_session()
    #tf.compat.v1.train.Saver().save(sess=sess, save_path=INFER_GRAPH_DIR)
    #tf.compat.v1.train.write_graph(graph_or_graph_def=sess.graph_def, logdir=INFER_GRAPH_DIR, name=infer_graph_path, as_text=False)
    # Method 1
    #freeze_graph.freeze_graph(input_graph=pbtxt_filepath, input_saver='', input_binary=False, input_checkpoint=ckpt_filepath, output_node_names='cnn/output', restore_op_name='save/restore_all', filename_tensor_name='save/Const:0', output_graph=pb_filepath, clear_devices=True, initializer_nodes='')
    
    '''

    from sklearn import metrics 
    predictions = model.predict(test_generator, verbose=1)
    ground_truth = test_generator.classes

    y_true = MultiLabelBinarizer().fit_transform(ground_truth)
    y_pred = np.rint(predictions)
    y_score = predictions


    f = metrics.f1_score(y_true, y_pred, average='samples')
    p = metrics.precision_score(y_true, y_pred, average='samples')
    r = metrics.recall_score(y_true, y_pred, average='samples')
    AP = metrics.average_precision_score(y_true, y_score)
    lrap = metrics.label_ranking_average_precision_score(y_true, y_score)
    ji = metrics.jaccard_score(y_true, y_pred, average="samples")
    #mAP = mAP(y_true, y_score)
    hl = metrics.hamming_loss(y_true, y_pred)
    rl = metrics.label_ranking_loss(y_true, y_score)
    cov = metrics.coverage_error(y_true, y_score)

    print("------------------------------")
    print("F1: ", f)
    print("presision: ",p)
    print("Recall: ", r)
    print("AP: ", AP) # edit to mAP
    print("lrap: ", lrap)
    print("Jaccard Index: ", ji)
    #print("mAP", mAP)
    print("Hamming Loss: ", hl)
    print("Ranking Loss: ", rl)
    print("Coverage Error: ", cov)

    '''
    return


def main():
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()

    ap.add_argument('-ih', '--input_height',
                    type=int,
                    default=28,
                    help='Input data height. Default is 28')                  
    ap.add_argument('-iw', '--input_width',
                    type=int,
                    default=28,
                    help='Input data width. Default is 28')                  
    ap.add_argument('-ic', '--input_chan',
                    type=int,
                    default=1,
                    help='Input data channels. Default is 1')                  
    ap.add_argument('-e', '--epochs',
                    type=int,
                    default=1,
                    help='Number of training epochs. Default is 100')                  
    ap.add_argument('-l', '--learnrate',
                    type=float,
                    default=0.0001,
                    help='Learning rate. Default is 0.0001')
    ap.add_argument('-b', '--batchsize',
                    type=int,
                    default=50,
                    help='Training batchsize. Default is 50')  
    ap.add_argument('-o', '--output_ckpt_path',
                    type=str,
                    default='./chkpt/float_model.ckpt',
                    help='Path and filename of trained checkpoint. Default is ./chkpt/float_model.ckpt')
    ap.add_argument('-ig', '--infer_graph_path',
                    type=str,
                    default='./chkpt/inference_graph.pb',
                    help='Path and filename of inference graph. Default is ./chkpt/inference_graph.pb')        
    ap.add_argument('-t', '--tboard_path',
                    type=str,
                    default='./tb_log',
                    help='Path of TensorBoard logs. Default is ./tb_log')
    ap.add_argument('-g', '--gpu',
                    type=str,
                    default='0',
                    help='IDs of GPU cards to be used. Default is 0')                  
    args = ap.parse_args() 


    print('\n------------------------------------')
    print('Keras version      :',tf.keras.__version__)
    print('TensorFlow version :',tf.__version__)
    print('Python version     :',(sys.version))
    print('------------------------------------')
    print ('Command line options:')
    print (' --input_height    : ', args.input_height)
    print (' --input_width     : ', args.input_width)
    print (' --input_chan      : ', args.input_chan)
    print (' --epochs          : ', args.epochs)
    print (' --batchsize       : ', args.batchsize)
    print (' --learnrate       : ', args.learnrate)
    print (' --output_ckpt_path: ', args.output_ckpt_path)
    print (' --infer_graph_path: ', args.infer_graph_path)
    print (' --tboard_path     : ', args.tboard_path)
    print (' --gpu             : ', args.gpu)
    print('------------------------------------\n')

    os.environ["CUDA_DEVICE_ORDER"]='PCI_BUS_ID'
    
    # indicate which GPU to use
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

    train(args.input_height,args.input_width,args.input_chan,args.epochs,args.learnrate, \
          args.batchsize,args.output_ckpt_path,args.infer_graph_path,args.tboard_path)


if __name__ == '__main__':
  main()