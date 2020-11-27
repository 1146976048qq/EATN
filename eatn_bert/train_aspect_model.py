import argparse
import logging
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import functional as F

import sys
sys.path.append('../')
from pre_models import TextNormalization, MultiLabel_Vectorizer, DocProcessor
from pre_models import BertForAspect
from pre_models import BertTrainer, BertPredictor, TextPhraseSimProcessor, get_batch, accuracy
from pre_models import BertInputFeatures
from pytorch_pretrained_bert import BertTokenizer

def fill_token(sent, term):
  #print(sent, term)
  return sent.replace('$T$', term)

def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--model_name', default='bert_aspect', type=str)
    
    logging.info('Loading data file')
    prefix_train = '../data/Experiment/aspect/2014_laptop/'
    #prefix_test = '../data/Experiment/aspect/2014_resturant/'
    prefix_test = '../data/Experiment/aspect/2014_laptop/'
  
    laptop_train = prefix_train + 'train.csv'
    laptop_test = prefix_test + 'test.csv'

    laptop_train_df = pd.read_csv(laptop_train)
    laptop_test_df = pd.read_csv(laptop_test)
    laptop_train_df['sent'] = laptop_train_df.apply(lambda x: fill_token(x[1], x[2]), axis=1)
    laptop_test_df['sent'] = laptop_test_df.apply(lambda x: fill_token(x[1], x[2]), axis=1)
    logging.info('finish loading data file')
    
    logging.info('Text Processor starting')
    ps_proc = TextPhraseSimProcessor(labels=None, feature_columns=['sent', 'term'], label_columns='polarity')

    train_examples = ps_proc.get_train_examples(df=laptop_train_df, size=-1, labels_available=True)
    dev_examples = ps_proc.get_dev_examples(df=laptop_test_df, size=-1, labels_available=True)
    logging.info('train_examples {}, dev_examples {}'.format(len(train_examples), len(dev_examples)))
    print('train_examples 0 ',train_examples[1])
    print('dev examples 1 ', dev_examples[1])
  
    logging.info('Loading tokenizer')
    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased-vocab.txt', do_lower_case=True)
    logging.info('tokenizer {}'.format(tokenizer))
    logging.info('initialize label proc')
    y_proc = MultiLabel_Vectorizer(onehot=False)
    y_proc.fit(laptop_train_df['polarity'])
  
    n_labels = y_proc.n_labels
    max_seq_len = 300    
    logging.info('n_labels {}, max_seq_len {}'.format(n_labels, max_seq_len))
    logging.info('labels {}'.format(y_proc.label2int_dict))
  
    #label_map = {-1:0, 0:1, 1:2}
    #inverse_label_map = {val:key for key, val in label_map.items()}
    #n_labels = 3
    
    params = {'n_labels':n_labels, 'batch_size':16, 'n_epochs':15, 
            'seq_len':max_seq_len, 'n_workers':-1, 'lr':0.0001, 'keep_prob':0.1, 'heads':12}
    
    #train_data = get_batch(train_examples, label_map, max_seq_len, tokenizer, output_mode="classification", label_available=True, batch_size=params['batch_size'], num_workers=-1)
    #val_data = get_batch(dev_examples, label_map, max_seq_len, tokenizer, output_mode="classification", label_available=True, batch_size=params['batch_size'], num_workers=-1)
    bert_input_proc = BertInputFeatures(tokenizer,max_seq_len,y_proc)
    train_data = get_batch(train_examples, bert_input_proc, label_available=True, batch_size=params['batch_size'], num_workers=-1)
    dev_data = get_batch(dev_examples, bert_input_proc, label_available=True, batch_size=params['batch_size'], num_workers=-1)
    
    bert_aspect_model = BertForAspect.from_pretrained('/home/kkzhang/bert_pytorch_model/bert_base', params)
   # bert_aspect_model = BertForAspect.from_pretrained('bert-base-uncased', params)

    device = torch.device('cpu')
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    #if torch.cuda.device_count()> 1:
    #    logging.info('has {} gpus'.format(torch.cuda.device_count()))
    #    bert_aspect_model = nn.DataParallel(bert_aspect_model)
            
    bert_aspect_model.to(device)
    
    loss_fn = F.cross_entropy

    # maintain all metrics required in this dictionary- these are used in the training and evaluation loops
    metrics = {
        'accuracy': accuracy,
        # could add more metrics such as accuracy for each token type
    }

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, bert_aspect_model.parameters()), lr=params['lr'])

    bert_aspect_trainer = BertTrainer(device, batch_size=params['batch_size'], n_epochs=params['n_epochs'], min_clip_val=-1.0, max_clip_val=1.0, n_labels=params['n_labels'])
    bert_aspect_trainer.train_and_evaluate(bert_aspect_model, train_data, dev_data, optimizer, metrics, loss_fn=loss_fn, model_dir='./results/')

#    bert_aspect_pred = BertPredictor(device, model=bert_aspect_model, max_seq_length=max_seq_len, tokenizer=tokenizer, X_proc=ps_proc, target_int2label_dict=inverse_label_map,
 #                        target_label2int_dict=label_map)


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  main()
