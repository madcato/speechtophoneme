#!/usr/bin/env python

import argparse
import os
import sys
import datetime
import keras
from keras.optimizers import SGD, RMSprop, Adam, Nadam
from data import combine_all_wavs_and_trans_from_csvs
from model import *
from char_map import get_number_of_char_classes
from generator import *
from memory_generator import *
from model import create_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
#######################################################
# Prevent pool_allocator message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
#######################################################
num_classes = get_number_of_char_classes()

def main(args, output_dir):
    '''
    5 steps
    '''
    #0. Check use of GPU
    from keras import backend as K
    gpus = K.tensorflow_backend._get_available_gpus()
    print("GPU available: {}".format(gpus))
    #1. combine data
    print("Getting data")
    train_dataprops, df_train = combine_all_wavs_and_trans_from_csvs(args.train_files)
    valid_dataprops, df_valid = combine_all_wavs_and_trans_from_csvs(args.valid_files)
    # Create generators
    if args.memory:
        GENE = PhonemeMemoryDataGenerator
    else:
        GENE = PhonemeDataGenerator
    training_generator = GENE(dataframe=df_train, batch_size=args.batchsize)
    validation_generator = GENE(dataframe=df_valid, batch_size=args.batchsize)
    print("Configuring project")
    # Create model
    model = create_model(fc_size=args.fc_size)
    # Create callbacks
    checkpointer = ModelCheckpoint(filepath='results/'+args.name, verbose=1, monitor='val_acc', save_best_only=True, mode='max')
    earlystopping = EarlyStopping(monitor='val_acc', mode='max', verbose=1, patience=25, min_delta=0.01)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=args.batchsize, write_graph=True, write_grads=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
    report = K.function([input_data, K.learning_phase()], [y_pred])
    report_cb = ReportCallback(report, validdata, model, args.name, save=True)
    callbacks = [checkpointer, earlystopping, reduce_lr, tensorboard, report_cb]
    # Save model
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    save_model(model, output_dir)
    # Optimizer
    if (args.opt.lower() == 'sgd'):
        optimizer = SGD(lr=args.learning_rate, decay=1e-6, momentum=0.9, nesterov=True, clipnorm=5)
    elif (args.opt.lower() == 'adam'):
        optimizer = Adam(lr=args.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8, clipnorm=5)
    elif (args.opt.lower() == 'nadam'):
        optimizer = Nadam(lr=args.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8, clipnorm=5)
    # Compile
    # CTC loss is implemented elsewhere, so use a dummy lambda function for the loss
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    # Train model on dataset
    model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        epochs=args.epochs,
                        callbacks=callbacks,
                        use_multiprocessing=False,
                        workers=6,
                        verbose=1)

if __name__ == '__main__':
    print("Getting args")
    parser = argparse.ArgumentParser()
    parser.add_argument('--tensorboard', type=bool, default=True,
                       help='True/False to use tensorboard')
    parser.add_argument('--memcheck', type=bool, default=False,
                       help='print out memory details for each epoch')
    parser.add_argument('--name', type=str, default='',
                       help='name of run, used to set checkpoint save name. Default uses timestamp')
    parser.add_argument('--train_files', type=str, default='',
                       help='list of all train files, seperated by a comma if multiple')
    parser.add_argument('--valid_files', type=str, default='',
                       help='list of all validation files, seperate by a comma if multiple')
    parser.add_argument('--train_steps', type=int, default=0,
                        help='number of steps for each epoch. Use 0 for automatic')
    parser.add_argument('--valid_steps', type=int, default=0,
                        help='number of validsteps for each epoch. Use 0 for automatic')
    parser.add_argument('--fc_size', type=int, default=512,
                       help='fully connected size for model')
    parser.add_argument('--loadcheckpointpath', type=str, default='',
                       help='If value set, load the checkpoint in a folder minus name minus the extension '
                            '(weights assumed as same name diff ext) '
                            ' e.g. --loadcheckpointpath ./checkpoints/'
                            'TRIMMED_ds_ctc_model/')
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='the learning rate used by the optimiser')
    parser.add_argument('--opt', type=str, default='sgd',
                        help='the optimiser to use, default is SGD, ')
    parser.add_argument('--sortagrad', type=bool, default=True,
                       help='If true, we sort utterances by their length in the first epoch')
    parser.add_argument('--epochs', type=int, default=1,
                       help='Number of epochs to train the model')
    parser.add_argument('--batchsize', type=int, default=32,
                       help='batch_size used to train the model')
    parser.add_argument('--memory', type=bool, default=False,
                       help='If true, all the data to train/eval is stored in ram, else it\'s stored in disk')
    args = parser.parse_args()
    runtime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    if args.name == "":
        args.name = "DS_model_"+runtime
    #required to save the JSON
    output_dir = os.path.join('results',
                                  'model_%s' % (runtime))
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    assert(keras.__version__ == "2.2.5") ## CoreML is strict
    print(args)
    main(args, output_dir)
