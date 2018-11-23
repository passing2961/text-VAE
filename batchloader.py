import sys
import chainer
import tensorflow as tf
import numpy as np

sys.path.append("../")

from config import FLAGS

class BatchLoader:
    def __init__(self):
        
        self.load_ptb_data()
        self.make_embedding_dict()
        
    def load_ptb_data(self):
        train, val, test = chainer.datasets.get_ptb_words()
        
        word2idx = chainer.datasets.get_ptb_words_vocabulary()
        word2idx['<pad>'] = 10000
        word2idx['<start>'] = 10001
        
        idx2word = {}
        
        for word, idx in word2idx.items():
            idx2word[idx] = word
            
        self.train_data = train
        self.val_data = val
        self.test_data = test
        self.word2idx = word2idx
        self.idx2word = idx2word
            
    def make_embedding_dict(self):
        embedding_dict = {}
        special_word_list = ['<start>','<pad>','<unk>','<eos>']
        
        for word, idx in self.word2idx.items():
            if word in special_word_list:
                embedding_dict[idx] = [0.0 for _ in range(FLAGS.EMBED_SIZE)]
            else:
                embedding_dict[idx] = [np.random.normal(scale=0.1) for _ in range(FLAGS.EMBED_SIZE)]

        print(len(embedding_dict))
        embedding_list = []
        for idx, emb in embedding_dict.items():
            embedding_list.append(emb)


        self.word_embedding = embedding_list
        
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

        #print('minibatch: {}'.format(minibatch))
        for sentence in minibatch:
            #print('sentence: {}'.format(sentence))
            
            encoder_input = [word for word in sentence[:-1]]
            #print('encoder_input: {}'.format(encoder_input))
            if len(encoder_input) < FLAGS.SEQ_LEN:
                for i in range(FLAGS.SEQ_LEN - len(encoder_input)):
                    encoder_input.append(10000)
            else:
                encoder_input = encoder_input[:FLAGS.SEQ_LEN]
            
            decoder_input = [word for word in sentence[:-1]]
            decoder_input.insert(0, self.word2idx['<start>'])
            
                        
            if len(decoder_input) < FLAGS.SEQ_LEN:
                for i in range(FLAGS.SEQ_LEN - len(decoder_input)):
                    decoder_input.append(10000)
            else:
                decoder_input = decoder_input[:FLAGS.SEQ_LEN]
                
            target = sentence
            if len(target) < FLAGS.SEQ_LEN:
                for i in range(FLAGS.SEQ_LEN - len(target)):
                    target.append(10000)
            else:
                target = target[:FLAGS.SEQ_LEN]
                target[-1] = 24
                
            encoder_input_batch.append(encoder_input)
            decoder_input_batch.append(decoder_input)
            target_batch.append(target)
        
        #print('[Before] encoder_input_batch: {}\ttype: {}, {}'.format(encoder_input_batch, type(encoder_input_batch), type(encoder_input_batch[0])))
        #print('[Before] {}'.format(type(encoder_input_batch[0][0])))
        if dropword:
            r = np.random.rand(FLAGS.BATCH_SIZE, FLAGS.SEQ_LEN)
            
            for i in range(FLAGS.BATCH_SIZE):
                for j in range(FLAGS.SEQ_LEN):
                    if(r[i][j] > FLAGS.DROPWORD_KEEP and decoder_input_batch[i][j] not in [self.word2idx['<start>'], self.word2idx['<pad>']]):
                        decoder_input_batch[i][j] = self.word2idx['<unk>']
        
        #print('[Check] {}, {}'.format(np.shape(encoder_input_batch), np.shape(encoder_input_batch[0])))

        encoder_input_batch = np.array(encoder_input_batch)
        decoder_input_batch = np.array(decoder_input_batch)
        target_batch = np.array(target_batch)
        
        #print('[After] encoder_input_batch: {}\ttype: {}, {}'.format(encoder_input_batch, type(encoder_input_batch), type(encoder_input_batch[0])))
        #print('[After] {}'.format(type(encoder_input_batch[0][0])))
        
        return encoder_input_batch, decoder_input_batch, target_batch

    def input2str(self, input_txt):
        except_list = ['<eos>', '<pad>']
        
        final_result = []
        
        for i in range(FLAGS.BATCH_SIZE):
            temp = []
            for j in range(FLAGS.SEQ_LEN):
                token = input_txt[i][j]
                if self.idx2word[token] not in except_list:
                    temp.append(self.idx2word[token])

            final_result.append(' '.join(temp))
            
        return final_result
    
    def pred2str(self, pred):
        except_list = ['<eos>', '<pad>', '<unk>']
        final_result = []
        
        for i in range(FLAGS.BATCH_SIZE):
            temp = []
            
            for j in range(FLAGS.SEQ_LEN):
                token = pred[j][i]

                if self.idx2word[token] not in except_list:
                    temp.append(self.idx2word[token])

            final_result.append(' '.join(temp))
            
        return final_result
    
    def logits2str(self, logits, sample_num, onehot=True):
        
        assert sample_num <= FLAGS.BATCH_SIZE
        
        generated_texts = []

        #print('[logits] {}: {}'.format(type(logits), np.shape(logits)))
        
        
        print('[argmax] {}'.format(np.argmax(logits[0],1)))
        if onehot:
            indices = [np.argmax(logit, 1) for logit in logits]
        else:
            indices = logits
            
        print('[indice] {}: {}'.format(type(indices), np.shape(indices)))

        temp_str = ''
        temp = []
        
        if onehot:
            for i in range(FLAGS.BATCH_SIZE):
                for j in range(FLAGS.SEQ_LEN):
                    if self.idx2word[indices[j][i]] == '<eos>' or self.idx2word[indices[j][i]] == '<pad>':
                        temp_str = ' '.join(temp)
                        break

                    temp.append(self.idx2word[indices[j][i]])

                generated_texts.append(temp_str)

        else:
            for i in range(FLAGS.BATCH_SIZE):
                for j in range(FLAGS.SEQ_LEN):
                    if self.idx2word[indices[i][j]] == '<eos>' or self.idx2word[indices[i][j]] == '<pad>':
                        temp_str = ' '.join(temp)
                        break

                    temp.append(self.idx2word[indices[i][j]])

                generated_texts.append(temp_str)

        return generated_texts
        
