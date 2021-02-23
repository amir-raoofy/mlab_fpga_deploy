import tensorflow as tf
import tensorflow.keras.backend as K
import os
import datetime
from tensorflow.python.tools import freeze_graph
from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input, Activation, BatchNormalization, Lambda
from tensorflow.keras.models import Model


def create_model():
    inputs = Input(shape=(128, 128, 1))
    x = Conv2D(4, (3, 3))(inputs)
    x = BatchNormalization()(x)
    # x = Lambda((lambda x: tf.layers.batch_normalization(x)))(x)
    x = Activation('relu')(x)
    x = Flatten()(x)
    x = Dense(5, activation='softmax')(x)
    model = Model(inputs, x, name='test')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


model = create_model()

# Training goes here...

model.save_weights("weights.h5")

K.clear_session()
K.set_learning_phase(0)

model = create_model()
model.load_weights("weights.h5")

save_dir = "./tmp_{:%Y-%m-%d_%H%M%S}".format(datetime.datetime.now())
tf.saved_model.simple_save(K.get_session(),
                           save_dir,
                           inputs={"input": model.inputs[0]},
                           outputs={"output": model.outputs[0]})

freeze_graph.freeze_graph(None,
                          None,
                          None,
                          None,
                          model.outputs[0].op.name,
                          None,
                          None,
                          os.path.join(save_dir, "frozen_model.pb"),
                          False,
                          "",
                          input_saved_model_dir=save_dir)