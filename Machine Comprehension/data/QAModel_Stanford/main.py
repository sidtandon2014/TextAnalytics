# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file contains the entrypoint to the rest of the code"""

from __future__ import absolute_import
from __future__ import division

import os
import io
import json
import sys
import logging

import tensorflow as tf

from qa_model import QAModel
from vocab import get_glove
from official_eval_helper import get_json_data, generate_answers, generate_answers_prob


logging.basicConfig(level=logging.INFO)

MAIN_DIR = "F:\Sid\Learnings\Data Scientist\Text Analysis\Machine Comprehension\data"
DEFAULT_DATA_DIR = os.path.join(MAIN_DIR, "TrainData") # relative path of data dir
EXPERIMENTS_DIR = os.path.join(MAIN_DIR, "ValidationData") # relative path of experiments dir


# High-level options
tf.app.flags.DEFINE_integer("gpu", 0, "Which GPU to use, if you have multiple.")
tf.app.flags.DEFINE_string("mode", "train", "Available modes: train / show_examples / official_eval")
tf.app.flags.DEFINE_string("experiment_name", "", "Unique name for your experiment. This will create a directory by this name in the experiments/ directory, which will hold all data related to this experiment")
tf.app.flags.DEFINE_integer("num_epochs", 0, "Number of epochs to train. 0 means train indefinitely")

# Hyperparameters
tf.app.flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
tf.app.flags.DEFINE_float("max_gradient_norm", 5.0, "Clip gradients to this norm.")
tf.app.flags.DEFINE_float("dropout", 0.15, "Fraction of units randomly dropped on non-recurrent connections.")
tf.app.flags.DEFINE_integer("batch_size", 100, "Batch size to use")
tf.app.flags.DEFINE_integer("hidden_size_encoder", 150, "Size of the hidden states") #150 for bidaf ; #200 otherwise
tf.app.flags.DEFINE_integer("hidden_size_qp_matching", 150, "Size of the hidden states")
tf.app.flags.DEFINE_integer("hidden_size_sm_matching", 50, "Size of the hidden states")
tf.app.flags.DEFINE_integer("hidden_size_fully_connected", 200, "Size of the hidden states")
tf.app.flags.DEFINE_integer("context_len", 50, "The maximum context length of your model")
tf.app.flags.DEFINE_integer("question_len", 12, "The maximum question length of your model")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Size of the pretrained word vectors. This needs to be one of the available GloVe dimensions: 50/100/200/300")


## Bool flags to select different models
tf.app.flags.DEFINE_bool("do_char_embed", False, "Include char embedding -True/False")
tf.app.flags.DEFINE_bool("add_highway_layer", True, "Add highway layer to concatenated embeddings -True/False")
tf.app.flags.DEFINE_bool("cnn_encoder", False, "Add CNN Encoder Layer -True/False")
tf.app.flags.DEFINE_bool("rnet_attention", False, "Perform RNET QP and SM attention-True/False")
tf.app.flags.DEFINE_bool("bidaf_attention", True, "Use BIDAF Attention-True/False")
tf.app.flags.DEFINE_bool("answer_pointer_RNET", False, "Use Answer Pointer from RNET-True/False")
tf.app.flags.DEFINE_bool("smart_span", True, "Select start and end idx based on smart conditions-True/False")

## Hyperparameters for Char CNN
tf.app.flags.DEFINE_integer("char_embedding_size", 8, "Size of char embedding")  #as suggested in handout
tf.app.flags.DEFINE_integer("word_max_len", 16, "max length for each word") # 99th percentile from Jupyter notebook
tf.app.flags.DEFINE_integer("char_out_size", 100, "num filters char CNN/out size") # same as filer size; as suggested in handout
tf.app.flags.DEFINE_integer("window_width", 5, "Kernel size for char cnn") #as suggested in handout


## Hyperparameters for CNN Encoder
tf.app.flags.DEFINE_integer("filter_size_encoder", 50, "Size of filter for cnn encoder")

## Hyperparameters for BIDAF
tf.app.flags.DEFINE_integer("hidden_size_modeling", 150, "Size of modeling layer")  #

# How often to print, save, eval
tf.app.flags.DEFINE_integer("print_every", 1, "How many iterations to do per print.")
tf.app.flags.DEFINE_integer("save_every", 500, "How many iterations to do per save.")
tf.app.flags.DEFINE_integer("eval_every", 500, "How many iterations to do per calculating loss/f1/em on dev set. Warning: this is fairly time-consuming so don't do it too often.")
tf.app.flags.DEFINE_integer("keep", 1, "How many checkpoints to keep. 0 indicates keep all (you shouldn't need to do keep all though - it's very storage intensive).")

# Reading and saving data
tf.app.flags.DEFINE_string("train_dir", "", "Training directory to save the model parameters and other info. Defaults to experiments/{experiment_name}")
tf.app.flags.DEFINE_string("glove_path", "", "Path to glove .txt file. Defaults to data/glove.6B.{embedding_size}d.txt")
tf.app.flags.DEFINE_string("data_dir", DEFAULT_DATA_DIR, "Where to find preprocessed SQuAD data for training. Defaults to data/")
tf.app.flags.DEFINE_string("ckpt_load_dir", "", "For official_eval mode, which directory to load the checkpoint fron. You need to specify this for official_eval mode.")
tf.app.flags.DEFINE_string("json_in_path", "", "For official_eval mode, path to JSON input file. You need to specify this for official_eval_mode.")
tf.app.flags.DEFINE_string("json_out_path", "predictions.json", "Output path for official_eval mode. Defaults to predictions.json")

FLAGS = tf.app.flags.FLAGS
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)


def main(unused_argv):
    # Print an error message if you've entered flags incorrectly
    if len(unused_argv) != 1:
        raise Exception("There is a problem with how you entered flags: %s" % unused_argv)

    # Check for Python 2
    if sys.version_info[0] != 2:
        raise Exception("ERROR: You must use Python 2 but you are running Python %i" % sys.version_info[0])

    # Print out Tensorflow version
    print "This code was developed and tested on TensorFlow 1.4.1. Your TensorFlow version: %s" % tf.__version__

    
    # Initialize bestmodel directory
    bestmodel_dir = os.path.join("F:\Sid\Learnings\Data Scientist\Text Analysis\Machine Comprehension\data\ModelLogs\best_weights", "best_checkpoint")


    # Initialize model
    qa_model = QAModel(FLAGS, id2word, word2id, emb_matrix)

    # Some GPU settings
    config=tf.ConfigProto()
    config.gpu_options.allow_growth = True

    # Split by mode
    if FLAGS.mode == "train":

        # Setup train dir and logfile
        if not os.path.exists(FLAGS.train_dir):
            os.makedirs(FLAGS.train_dir)
        file_handler = logging.FileHandler(os.path.join(FLAGS.train_dir, "log.txt"))
        logging.getLogger().addHandler(file_handler)

        # Save a record of flags as a .json file in train_dir
        with open(os.path.join(FLAGS.train_dir, "flags.json"), 'w') as fout:
            json.dump(FLAGS.__flags, fout)

        # Make bestmodel dir if necessary
        if not os.path.exists(bestmodel_dir):
            os.makedirs(bestmodel_dir)

        with tf.Session(config=config) as sess:

            # Load most recent model
            initialize_model(sess, qa_model, FLAGS.train_dir, expect_exists=False)

            # Train
            qa_model.train(sess, train_context_path, train_qn_path, train_ans_path, dev_qn_path, dev_context_path, dev_ans_path)


    elif FLAGS.mode == "show_examples":
        with tf.Session(config=config) as sess:

            # Load best model
            initialize_model(sess, qa_model, bestmodel_dir, expect_exists=True)

            # Show examples with F1/EM scores
            _, _ = qa_model.check_f1_em(sess, dev_context_path, dev_qn_path, dev_ans_path, "dev", num_samples=10, print_to_screen=True)


    elif FLAGS.mode == "official_eval":
        if FLAGS.json_in_path == "":
            raise Exception("For official_eval mode, you need to specify --json_in_path")
        if FLAGS.ckpt_load_dir == "":
            raise Exception("For official_eval mode, you need to specify --ckpt_load_dir")

        # Read the JSON data from file
        qn_uuid_data, context_token_data, qn_token_data = get_json_data(FLAGS.json_in_path)

        with tf.Session(config=config) as sess:

            # Load model from ckpt_load_dir
            initialize_model(sess, qa_model, FLAGS.ckpt_load_dir, expect_exists=True)

            # Get a predicted answer for each example in the data
            # Return a mapping answers_dict from uuid to answer
            answers_dict = generate_answers(sess, qa_model, word2id, qn_uuid_data, context_token_data, qn_token_data)

            # Write the uuid->answer mapping a to json file in root dir
            print "Writing predictions to %s..." % FLAGS.json_out_path
            with io.open(FLAGS.json_out_path, 'w', encoding='utf-8') as f:
                f.write(unicode(json.dumps(answers_dict, ensure_ascii=False)))
                print "Wrote predictions to %s" % FLAGS.json_out_path


    else:
        raise Exception("Unexpected value of FLAGS.mode: %s" % FLAGS.mode)

if __name__ == "__main__":
    tf.app.run()
