import sys

import tensorflow as tf
import numpy as np

sys.path.append("../")

from config import FLAGS

class VAE():
    def __init__(self, sess, batchloader, learning_rate, training=True, ru=False):

        self.sess = sess
        self.batchloader = batchloader
        self.training = training
        self.ru = ru
        self.learning_rate = learning_rate
        self.word_embedding = self.batchloader.word_embedding

        self.parameter_init()
        self.embedding_init()
        self.encoder_init()
        self.decoder_init()
        self.loss_init()
        self.train_init()

        '''with tf.name_scope("Summary"):
            if self.training:
                loss_summary = tf.summary.scalar("loss", self.loss, family="train_loss")
                reconst_loss_summary = tf.summary.scalar("reconst_loss", self.reconst_loss, family="train_loss")
                kld_summary = tf.summary.scalar("kld", self.KL_d, family="kld")
                kld_weight_summary = tf.summary.scalar("kld_weight", self.KL_d_weight, family="parameters")
                mu_summary = tf.summary.histogram("mu", tf.reduce_mean(self.mu, 0))
                var_summary = tf.summary.histogram("var", tf.reduce_mean(tf.exp(self.logvar), 0))
                lr_summary = tf.summary.scalar("lr", self.learning_rate, family="parameters")

                self.merged_summary = tf.summary.merge([loss_summary, reconst_loss_summary, kld_summary, kld_weight_summary, mu_summary, var_summary, lr_summary])

            else:
                valid_reconst_loss_summary = tf.summary.scalar("valid_reconst_loss", self.reconst_loss, family="valid_loss")

                self.merged_summary = tf.summary.merge([valid_reconst_loss_summary])'''
            
    def parameter_init(self):
        with tf.variable_scope("parameter"):
            
            self.encoder_input = tf.placeholder(tf.int64, shape=[FLAGS.BATCH_SIZE, FLAGS.SEQ_LEN], name='encoder_input')                        
            self.decoder_input = tf.placeholder(tf.int64, shape=[FLAGS.BATCH_SIZE, FLAGS.SEQ_LEN], name='decoder_input')
            self.target = tf.placeholder(tf.int64, shape=[FLAGS.BATCH_SIZE, FLAGS.SEQ_LEN], name='target')
        
            
            
            self.step = tf.placeholder(tf.float32, shape=(), name="step")
            
            self.pad_id = tf.convert_to_tensor(10000, dtype=tf.int64, name='pad')
            self.encoder_length = tf.reduce_sum(tf.to_int64(tf.not_equal(self.encoder_input, self.pad_id)), 1, name='encoder_input_lengths')            
            self.decoder_length = tf.reduce_sum(tf.to_int64(tf.not_equal(self.decoder_input, self.pad_id)), 1, name='decoder_input_lengths')
            
            #print('[encoder_length] {}\t[decoder_length] {}'.format(self.encoder_length, self.decoder_length))
            
    def embedding_init(self):
        with tf.variable_scope("embedding_layer"):
            self.embedding = tf.get_variable(name='embedding', 
                                             shape=[FLAGS.VOCAB_SIZE, FLAGS.EMBED_SIZE], 
                                             dtype=tf.float32, 
                                             initializer=tf.constant_initializer(self.word_embedding))
            # start, unk, pad, eos = 0 인 부분 만들어줘야함
    def encoder_init(self):
        
        with tf.variable_scope("encoder_rnn_layer"):
            self.encoder_input_embedding = tf.nn.embedding_lookup(self.embedding, self.encoder_input)
            
            print(self.encoder_input_embedding.shape)
            
            
            cell = tf.contrib.rnn.LSTMCell(FLAGS.RNN_SIZE, state_is_tuple=True)
  
            if self.training:
                cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=FLAGS.ENCODER_DROPOUT_KEEP)
            
  
            #rnn_layers = tf.contrib.rnn.MultiRNNCell([cell] * FLAGS.RNN_NUM)
            self.init_state = cell.zero_state(FLAGS.BATCH_SIZE, dtype=tf.float32)
            #self.init_state = [rnn_layers.zero_state(FLAGS.BATCH_SIZE, dtype=tf.float32) for _ in range(FLAGS.RNN_NUM)]
            #self.state = [tf.placeholder(tf.float32, (FLAGS.BATCH_SIZE), "state") for _ in range(FLAGS.RNN_NUM)]

            self.outputs , self.final_state = tf.nn.dynamic_rnn(cell, self.encoder_input_embedding, initial_state=self.init_state, time_major=False, sequence_length=self.encoder_length, dtype=tf.float32)

            #self.reshape_state = tf.reshape(self.final_state, [-1, FLAGS.RNN_SIZE])

        with tf.variable_scope("encoder_rnn_to_linear1"):
            context_to_hidden_W = tf.get_variable(name="rnn_to_linear_W",
                                                  shape=[FLAGS.RNN_SIZE, 100],
                                                  dtype=tf.float32,
                                                  initializer=tf.random_normal_initializer(stddev=0.1))
            
            context_to_hidden_b = tf.get_variable(name="rnn_to_linear_b",
                                                  shape=[100],
                                                  dtype=tf.float32)
        
        with tf.variable_scope("encoder_rnn_linear2"):
            context_to_mu_W = tf.get_variable(name="linear_to_mu_W",
                                              shape=[100, FLAGS.LATENT_VARIABLE_SIZE],
                                              dtype=tf.float32,
                                              initializer=tf.random_normal_initializer(stddev=0.1))
            
            context_to_mu_b = tf.get_variable(name="linear_to_mu_b",
                                              shape=[FLAGS.LATENT_VARIABLE_SIZE],
                                              dtype=tf.float32)
            
            context_to_logvar_W = tf.get_variable(name="linear_to_logvar_W",
                                                  shape=[100, FLAGS.LATENT_VARIABLE_SIZE],
                                                  dtype=tf.float32,
                                                  initializer=tf.random_normal_initializer(stddev=0.1))
            
            context_to_logvar_b = tf.get_variable(name="linear_to_logvar_b",
                                                  shape=[FLAGS.LATENT_VARIABLE_SIZE],
                                                  dtype=tf.float32)
        
        with tf.name_scope("hidden_state"):
            hidden = tf.nn.relu(tf.matmul(self.final_state[-1], context_to_hidden_W) + context_to_hidden_b)
            #self.encoder = Encoder[FLAGS.ENCODER_NAME](self.embedding, self.encoder_input, training=self.training, ru=self.ru)
    
        with tf.name_scope("mu"):
            self.mu = tf.matmul(hidden, context_to_mu_W) + context_to_mu_b
        
        with tf.name_scope("log_var"):
            self.logvar = tf.matmul(hidden, context_to_logvar_W) + context_to_logvar_b
        
        with tf.name_scope("z"):
            z = tf.random_normal((FLAGS.BATCH_SIZE, FLAGS.LATENT_VARIABLE_SIZE), stddev=1.0)
        
        with tf.name_scope("latent_variables"):
            #if self.training:
            self.latent_variables = self.mu + tf.exp(0.5 * self.logvar) * z
            #else:
            #    self.latent_variables = tf.placeholder(tf.float32, shape=[FLAGS.BATCH_SIZE, FLAGS.LATENT_VARIABLE_SIZE], name="latent_variables_input")
                
    
    def decoder_init(self):
        with tf.variable_scope("decoder_rnn2vocab"):
            self.rnn2vocab_W = tf.get_variable(name="rnn2vocab_W",
                                               shape=(FLAGS.RNN_SIZE, FLAGS.VOCAB_SIZE),
                                               dtype=tf.float32,
                                               initializer=tf.random_normal_initializer(stddev=0.1))
            self.rnn2vocab_b = tf.get_variable(name="rnn2vocab_b",
                                               shape=(FLAGS.VOCAB_SIZE),
                                               dtype=tf.float32)
            
        with tf.variable_scope("docoder_rnn_layer"):
            decoder_cell = tf.contrib.rnn.BasicLSTMCell(FLAGS.RNN_SIZE)
            decoder_cell = tf.nn.rnn_cell.DropoutWrapper(decoder_cell, output_keep_prob=FLAGS.DECODER_DROPOUT_KEEP)
            #self.decoder_rnn_layers = tf.contrib.rnn.MultiRNNCell([decoder_cell] * FLAGS.RNN_NUM)

            self.decoder_init_states = decoder_cell.zero_state(FLAGS.BATCH_SIZE, dtype=tf.float32)
            
            if self.training:
                
                self.decoder_input_embedding = tf.nn.embedding_lookup(self.embedding, self.decoder_input)
                print('[decoder_input] {}'.format(self.decoder_input_embedding.shape))
                assert self.decoder_input_embedding.shape == (FLAGS.BATCH_SIZE, FLAGS.SEQ_LEN, FLAGS.EMBED_SIZE)
                
                self.mod_latent_variables = tf.stack([self.latent_variables] * FLAGS.SEQ_LEN, axis=1)
                print('[latent] {}'.format(self.mod_latent_variables.shape))
                assert self.mod_latent_variables.shape == (FLAGS.BATCH_SIZE, FLAGS.SEQ_LEN, FLAGS.LATENT_VARIABLE_SIZE)
                
                self.decoder_rnn_input = tf.concat([self.mod_latent_variables, self.decoder_input_embedding], axis=2)
                assert self.decoder_rnn_input.shape == (FLAGS.BATCH_SIZE, FLAGS.SEQ_LEN, FLAGS.LATENT_VARIABLE_SIZE + FLAGS.EMBED_SIZE)
                
                self.decoder_outputs, self.decoder_state = tf.nn.dynamic_rnn(decoder_cell, self.decoder_rnn_input, initial_state=self.decoder_init_states, sequence_length=self.decoder_length, time_major=False, dtype=tf.float32)
                
                #print('[decoder_output] {}\t[decoder_state] {}'.format(self.decoder_outputs, self.decoder_state))
                self.decoder_outputs = tf.reshape(self.decoder_outputs, [-1, FLAGS.RNN_SIZE])
                
                with tf.name_scope("decoder_train_logits"):
                    self.decoder_logits = tf.matmul(self.decoder_outputs, self.rnn2vocab_W) + self.rnn2vocab_b

                print('[decoder_logit] {}'.format(self.decoder_logits.shape))
                
                #self.decoder_prediction = tf.argmax(self.decoder_logits, 2)
                '''decoder_input_t = tf.transpose(self.decoder_input, perm=[1, 0])
                
                self.decoder_input_list = []
                self.decoder_logits = []
                
                for i in range(FLAGS.SEQ_LEN):
                    self.decoder_input_list.append(decoder_input_t[i])
                    assert self.decoder_input_list[i].shape == (FLAGS.BATCH_SIZE)
                    
                for i in range(FLAGS.SEQ_LEN):
                    self.decoder_input_embedding = tf.nn.embedding_lookup(self.embedding, self.decoder_input_list[i])
                    #print('decoder input embedding : {}'.format(self.decoder_input_embedding))
                                    
                    print('[latent] type: {}\tshape: {}'.format(type(self.latent_variables), np.shape(self.latent_variables)))
                    print('[decoder_input] type: {}\tshape: {}'.format(type(self.decoder_input_embedding), np.shape(self.decoder_input_embedding)))
                    self.decoder_rnn_input = tf.concat([self.latent_variables, self.decoder_input_embedding], axis=1)
                    #print('decoder_rnn_input : {}'.format(self.decoder_rnn_input.shape))
                    
                    self.decoder_outputs, self.decoder_state = decoder_cell(self.decoder_rnn_input, self.decoder_init_states)
                    assert self.decoder_outputs.shape == (FLAGS.BATCH_SIZE, FLAGS.RNN_SIZE)
                    
                    self.decoder_logit = tf.contrib.layers.linear(self.decoder_outputs, FLAGS.VOCAB_SIZE)
                    #print('decoder_logit : {} {}'.format(self.decoder_logit.shape, self.decoder_logit))
                    assert self.decoder_logit.shape == (FLAGS.BATCH_SIZE, FLAGS.VOCAB_SIZE)
                    
                    self.decoder_logits.append(self.decoder_logit)
                    
                self.decoder_prediction = tf.argmax(self.decoder_logits, 2)'''
            else:               

                word2idx = self.batchloader.word2idx
                decoder_start_input = np.array([word2idx['<start>'] for _ in range(FLAGS.BATCH_SIZE)])
                self.decoder_start_input = tf.constant(decoder_start_input, dtype=tf.int32)
                next_input = tf.nn.embedding_lookup(self.embedding, self.decoder_start_input)
                
                print('[start] {}'.format(next_input.shape))

                self.decoder_state = self.decoder_init_states
                self.decoder_prediction = []
                self.decoder_logits = []
                
                for i in range(FLAGS.SEQ_LEN):
                    
                    self.decoder_rnn_input = tf.concat([self.latent_variables, next_input], axis=1)
                    assert self.decoder_rnn_input.shape == (FLAGS.BATCH_SIZE, FLAGS.LATENT_VARIABLE_SIZE + FLAGS.EMBED_SIZE)

                    self.decoder_rnn_input = tf.expand_dims(self.decoder_rnn_input, axis=1)
                    #print('[decoder_rnn_input] {}'.format(self.decoder_rnn_input.shape))
                                        
                    self.decoder_outputs, self.decoder_state = tf.nn.dynamic_rnn(decoder_cell, self.decoder_rnn_input, initial_state=self.decoder_state, time_major=False, dtype=tf.float32)
                    #print('[decoder_output] {}\t[decoder_state] {}'.format(self.decoder_outputs, self.decoder_state))
                    
                    self.decoder_outputs = tf.reshape(self.decoder_outputs, [-1, FLAGS.RNN_SIZE])
                    self.decoder_logit = tf.matmul(self.decoder_outputs, self.rnn2vocab_W) + self.rnn2vocab_b
                    self.decoder_logits.append(self.decoder_logit)
                    #print('[decoder_logit] {}'.format(self.decoder_logits))
                    
                    next_token = tf.stop_gradient(tf.argmax(self.decoder_logit, 1))
                    self.decoder_prediction.append(next_token)
                    next_input = tf.nn.embedding_lookup(self.embedding, next_token)
                
                '''for i in range(FLAGS.SEQ_LEN):
                    self.decoder_rnn_input = tf.concat([self.latent_variables, next_input], axis=1)
                    assert self.decoder_rnn_input.shape == (FLAGS.BATCH_SIZE, FLAGS.LATENT_VARIABLE_SIZE + FLAGS.EMBED_SIZE)
                    
                    decoder_output, decoder_state = decoder_cell(self.decoder_rnn_input, state)
                    
                    logit = tf.contrib.layers.linear(decoder_output, FLAGS.VOCAB_SIZE)
                    assert logit.shape == (FLAGS.BATCH_SIZE, FLAGS.VOCAB_SIZE)
                    
                    self.decoder_logits.append(logit)
                    
                    next_symbol = tf.stop_gradient(tf.argmax(logit, 1))
                    self.decoder_prediction.append(next_symbol)
                    next_input = tf.nn.embedding_lookup(self.embedding, next_symbol)'''
                    
                
                #print('next symbol: {}'.format(np.shape(next_symbol)))
                #print('decoder logits : {}'.format(np.shape(self.decoder_logits)))
                #print('decoder prediction : {}'.format(np.shape(self.decoder_prediction)))
    
            #self.decoder = Decoder[FLAGS.DECODER_NAME](self.embedding, self.decoder_input_list, self.latent_variables, self.batchloader, training=self.training, ru=self.ru)
    
    def loss_init(self):
        with tf.name_scope("Loss"):
            self.logits = self.decoder_logits

            self.KL_d = tf.reduce_mean(-0.5 * tf.reduce_sum(2 * self.logvar - tf.square(self.mu) - tf.exp(2 * self.logvar) + 1, axis=1))

            self.KL_d_weight = tf.placeholder(tf.float32, shape=[], name="KL_d_weight")

            label = tf.one_hot(self.target, depth=FLAGS.VOCAB_SIZE, dtype=tf.float32)
            #print('[label] {}'.format(label.shape))
            #print('[logit] {}'.format(self.logits.shape))

            reconstruction_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=tf.one_hot(self.target, depth=FLAGS.VOCAB_SIZE, dtype=tf.float32))                    
            print('[reconstruction_loss] {}'.format(reconstruction_loss.shape))
            self.reconst_loss = tf.reduce_mean(reconstruction_loss) * FLAGS.SEQ_LEN

            print('[reconst_loss] {}'.format(self.reconst_loss.shape))

            self.loss = self.reconst_loss + self.KL_d_weight * self.KL_d

    def train_init(self):
        if self.training:
            with tf.name_scope("Optimizer"):
          
                grad, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tf.trainable_variables()), FLAGS.MAX_GRAD)
                optimizer = tf.train.AdamOptimizer(self.learning_rate, beta1=0.5)

                self.train_op = optimizer.apply_gradients(zip(grad, tf.trainable_variables()))
