import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow_hub as hub;
from pathlib import Path
import pandas as pd;
import glob
from sklearn.preprocessing import MultiLabelBinarizer
import multiprocessing
import numpy as np

class cfg:
    img_size_dest = (224,224)
    train_batchsize = 100
    test_batchsize = 100
    val_batchsize = 256
    num_epochs = 500
    weights = {
        "Agriculture":                                               67256 ,
"Agro-forestryareas":					  15790 ,
"Arableland":						 100394 ,
"Beaches,dunes,sands":					   1193 ,
"Broad-leavedforest":					  73407 ,
"Coastalwetlands":						   1033 ,
"Complexcultivationpatterns":				  53530 ,
"Coniferousforest":					  86569 ,
"Industrialorcommercialunits":				   6182 ,
"Inlandwaters":						  35349 ,
"Inlandwetlands":						  11620 ,
"Marinewaters":						  39110 ,
"Mixedforest":						  91926 ,
"Moors,heathlandandsclerophyllousvegetation":		   8434 ,
"Naturalgrasslandandsparselyvegetatedareas":		   6663 ,
"Pastures":						  50977 ,
"Permanentcrops":				  15862 ,
"Transitionalwoodland,shrub":				  77589 ,
"Urbanfabric":					  38779 ,
}
#    steps_per_epoch = 256
    
    

def get_data(path_images):
    classes_arr = []
    files = glob.glob(str(path_images) + "/" + '*.tif')

    for filename in files:
        out = Path(filename)
        name = str(out.name).split("__")[1].split("--")
        classes_arr.append([str(out), name])
    return pd.DataFrame(classes_arr, columns=['files', 'labels'])

path_data = "data"
# load data paths to dictionaries.
dict_train = get_data("truecolor/train/")
dict_test = get_data("truecolor/test/")
dict_val = get_data("truecolor/val/")


mlb = MultiLabelBinarizer()
mlb.fit(dict_train['labels'])
print("Labels:")
for (i, label) in enumerate(mlb.classes_):
    print("{}. {}".format(i, label))

# Create Data Iterators
''' Data generator for training data'''
train_datagen = ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_dataframe(
    dict_train,
    x_col='files',
    y_col='labels',
    target_size=cfg.img_size_dest,
    batch_size=cfg.train_batchsize,
    class_mode='categorical',
)

### Now compute class weights from this

from sklearn.utils import class_weight
import sys
weight_sum = sum([cfg.weights[x] for x in cfg.weights])
print("Total Weight: %s" % weight_sum)
weights = {y:(cfg.weights[x] / weight_sum) for y,x in enumerate(mlb.classes_)}




''' Data generator for test data'''
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_dataframe(
    dict_test,
    x_col='files',
    y_col='labels',
    target_size=cfg.img_size_dest,
    batch_size=cfg.test_batchsize,
    class_mode='categorical',
    shuffle=False
)

''' Data generator for validation data'''
validation_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = validation_datagen.flow_from_dataframe(
    dict_val,
    x_col='files',
    y_col='labels',
    target_size=cfg.img_size_dest,
    batch_size=cfg.val_batchsize,
    class_mode='categorical',
    shuffle=False
)

# get number of cores
num_cpus = multiprocessing.cpu_count()
print("Number of cores: %s" %num_cpus)



# Load Model from Google
hub_layer = hub.KerasLayer("https://tfhub.dev/google/remote_sensing/bigearthnet-resnet50/1", trainable=False)

# Add Layers to predict our dataset
model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(19, activation='sigmoid'))



# Compile new model
model.compile(
        loss='binary_crossentropy',
        optimizer=tf.keras.optimizers.SGD(learning_rate=10e-3, momentum=0.9),
        metrics=[tf.keras.metrics.Precision(), tf.keras.metrics.Recall()] 
)


history = model.fit(
        train_generator,
        epochs=cfg.num_epochs,
        validation_steps=validation_generator.samples/validation_generator.batch_size,
        steps_per_epoch=train_generator.samples/train_generator.batch_size,
        validation_data=validation_generator,
#       callbacks=[batchMetric],
        workers=num_cpus,
        class_weight=weights,
        verbose=1
)


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
mAP = mAP(y_true, y_score)
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
print("mAP", mAP)
print("Hamming Loss: ", hl)
print("Ranking Loss: ", rl)
print("Coverage Error: ", cov)
