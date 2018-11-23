
import tensorflow as tf
import sys

#flags = tf.app.flags

flags = tf.app.flags 
FLAGS = flags.FLAGS

flags.DEFINE_integer('VOCAB_SIZE', 10002, '')
flags.DEFINE_integer('BATCH_SIZE', 32, '')
flags.DEFINE_integer('TEST_BATCH_SIZE', 1, '')
flags.DEFINE_integer('SEQ_LEN', 60, '')
flags.DEFINE_integer('EPOCH', 40, '')
flags.DEFINE_integer('BATCHES_PER_EPOCH', 1000, '')

flags.DEFINE_float('DROPWORD_KEEP', 0.62, '')
flags.DEFINE_float('ENCODER_DROPOUT_KEEP', 1.0, '')
flags.DEFINE_float('DECODER_DROPOUT_KEEP', 1.0, '')
flags.DEFINE_float('LEARNING_RATE', 0.001, '')
flags.DEFINE_float('LR_DECAY_START', 10, '')
flags.DEFINE_float('MAX_GRAD', 5.0, '')

flags.DEFINE_integer('EMBED_SIZE', 353, '')
flags.DEFINE_integer('LATENT_VARIABLE_SIZE', 13, '')

flags.DEFINE_integer('RNN_NUM', 1, '')
flags.DEFINE_integer('RNN_SIZE', 191, '')

FLAGS = flags.FLAGS
