import sys
import chainer
import tensorflow as tf
import numpy as np

sys.path.append("../")

from config import FLAGS

class BatchLoader:
    def __init__(self):
        
        self.load_ptb_data()
        
    def load_ptb_data(self):
        train, val, test = chainer.datasets.get_ptb_words()
        
        word2idx = chainer.datasets.get_ptb_words_vocabulary()
        word2idx['<pad>'] = 10000
        word2idx['<start>'] = 10001
        word2idx['<unk>'] = 10002
        
        idx2word = dict((idx, word) for word, idx in word2idx.items())
        
        self.train_data = train
        self.val_data = val
        self.test_data = test
        self.word2idx = word2idx
        self.idx2word = idx2word
    
    def make_batch(self, batch_size, is_training):
        document = []
        sentence = []
        
        if is_training:
            for ele in self.train_data:
                sentence.append(ele)
                if ele == 24:
                    document.append(sentence)
                    sentence = []
        else:
            for ele in self.val_data:
                sentence.append(ele)
                if ele == 24:
                    document.append(sentence)
                    sentence = []
                    
        return document
        
    def prepro_minibatch(self, minibatch, dropword):
        
        encoder_input_batch = []
        decoder_input_batch = []
        target_batch = []

        for sentence in minibatch:
            
            encoder_input = [word for word in sentence[:-1]]
            if len(encoder_input) < FLAGS.SEQ_LEN:
                for i in range(FLAGS.SEQ_LEN - len(encoder_input)):
                    encoder_input.append(10000)
            
            decoder_input = [word for word in sentence[:-1]]
            decoder_input.insert(0, word2idx['<start>'])
            
                        
            if len(decoder_input) < FLAGS.SEQ_LEN:
                for i in range(FLAGS.SEQ_LEN - len(decoder_input)):
                    decoder_input.append(10000)
                    
            target = sentence
            if len(target) < FLAGS.SEQ_LEN:
                for i in range(FLAGS.SEQ_LEN - len(target)):
                    target.append(10000)
                    
            encoder_input_batch.append(encoder_input)
            decoder_input_batch.append(decoder_input)
            target_batch.append(target)
        
        if dropword:
            r = np.random.rand(FLAGS.BATCH_SIZE, FLAGS.SEQ_LEN)
            
            for i in range(FLAGS.BATCH_SIZE):
                for j in range(FLAGS.SEQ_LEN):
                    if(r[i][j] > FLAGS.DROPWORD_KEEP and decoder_input_batch[i][j] not in [self.word2idx['<start>'], self.word2idx['pad']]):
                        decoder_input[i][j] = self.word2idx['<unk>']
        
        return np.array(encoder_input_batch), np.array(decoder_input_batch), np.array(target_batch)

    def logits2str(self, logits, sample_num, onehot=True):
        
        assert sample_num <= FLAGS.BATCH_SIZE
        
        generated_texts = []
        
        if onehot:
            indices = [np.argmax(logit, 1) for logit in logits]
        else:
            indices = logits
            
        seq_len = len(indices)
        assert seq_len == FLAGS.SEQ_LEN
        
        for i in range(sample_num):
            temp = ''
            
            for j in range(seq_len):
                temp += self.idx2word[indices[j][i]]
            
            generated_texts.append(temp)
        
        return generated_texts
        