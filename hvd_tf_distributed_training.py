import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models
import horovod.tensorflow.keras as hvd
import numpy as np
import tensorflow_datasets as tfds

if __name__=='__main__':

    hvd.init()

    batch_size = 32
    epochs = 100

    gpus = tf.config.experimental.list_physical_devices('GPU')

    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

    train_ds, test_ds = tfds.load('cifar10', 
                                split=['train','test'], 
                                as_supervised = True, 
                                batch_size = batch_size)

    model = tf.keras.applications.mobilenet_v2.MobileNetV2(include_top=True, 
                                                        weights=None,
                                                        input_shape=(32, 32, 3),
                                                        classes=10)

    if hvd.rank() == 0:
        print(model.summary())

    opt = tf.keras.optimizers.SGD(0.0005 * hvd.size())
    opt = hvd.DistributedOptimizer(opt)

    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy'],
        experimental_run_tf_function=False)

    callbacks = [hvd.callbacks.BroadcastGlobalVariablesCallback(0)]

    if hvd.rank() == 0:
        verbose = 1
    else:
        verbose=0

    model.fit(train_ds, epochs=epochs, 
            verbose=verbose, callbacks=callbacks)

