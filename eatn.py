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
from pre_models import BertForCrossAspect
from pre_models import BertMultiTrainer, BertPredictor
from pre_models import CrossAspectProcessor,get_cross_aspect_batch,accuracy
from pre_models import BertCrossFeatures
from pytorch_pretrained_bert import BertTokenizer


def fill_token(sent, term):
  #print(sent, term)
  return sent.replace('$T$', term)

def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--model_name', default='bert_aspect', type=str)
    
    # aspect-term dataset
    prefix_train = '../data_after_process/aspect_term/2014_laptop/'
    # prefix_train = '../data_after_process/aspect_term/2014_resturant/'
    # prefix_train = '../data_after_process/aspect_term/2014_twitter/'

    prefix_test = '../data_after_process/aspect_term/2014_resturant/'
    # prefix_train = '../data_after_process/aspect_term/2014_laptop/'
    # prefix_train = '../data_after_process/aspect_term/2014_twitter/'


    # aspect-category dataset
    # prefix_train = '../data_after_process/aspect_term/2014_laptop/'
    # prefix_train = '../data_after_process/aspect_term/2014_resturant/'
    # prefix_train = '../data_after_process/aspect_term/2014_twitter/'

    # prefix_train = '../data_after_process/aspect_category/restaurant/'
    # prefix_train = '../data_after_process/aspect_category/hotel/'
    # prefix_train = '../data_after_process/aspect_category/beautyspa/'


    source_train = prefix_train + 'train.csv'
    target_train = prefix_test + 'train.csv'
    source_dev = prefix_train + 'test.csv'
    target_dev = prefix_test + 'test.csv'

    source_train_df = pd.read_csv(source_train)
    target_train_df = pd.read_csv(target_train)
    source_train_df['sent'] = source_train_df.apply(lambda x: fill_token(x[1], x[2]), axis=1)
    target_train_df['sent'] = target_train_df.apply(lambda x: fill_token(x[1], x[2]), axis=1)
    logging.info('source_train_df shape {}, target_train_df shape {}'.format(source_train_df.shape, target_train_df.shape))

    source_dev_df = pd.read_csv(source_dev)
    target_dev_df = pd.read_csv(target_dev)
    source_dev_df['sent'] = source_dev_df.apply(lambda x: fill_token(x[1], x[2]), axis=1)
    target_dev_df['sent'] = target_dev_df.apply(lambda x: fill_token(x[1], x[2]), axis=1)
    logging.info('source_dev_df shape {}, target_dev_df shape {}'.format(source_dev_df.shape,target_dev_df.shape))
    
#    ps_proc = TextPhraseSimProcessor(labels=None, feature_columns=['sent', 'term'], label_columns='polarity')
    ps_proc = CrossAspectProcessor(labels=None, source_feature_columns=['sent','term'], source_aspect_labels='polarity',
                                   target_feature_columns=['sent','term'], target_aspect_labels='polarity')
    
    logging.info('start get examples')
    source_train_examples,target_train_examples = ps_proc.get_train_examples(source_df=source_train_df, target_df=target_train_df, size=-1, labels_available=True)
    source_dev_examples, target_dev_examples = ps_proc.get_dev_examples(source_df=source_dev_df,target_df=target_dev_df, size=-1, labels_available=True)
    logging.info('source_train_examples {}, target_train_examples {}'.format(len(source_train_examples), len(target_train_examples)))
    logging.info('source_dev_examples {}, target_dev_examples {}'.format(len(source_dev_examples), len(target_dev_examples)))    

    
   # source_dev_examples, target_dev_examples = ps_proc.get_dev_examples(source_df=source_train_df,target_df=target_train_df, size=100, labels_available=True)
    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased-vocab.txt', do_lower_case=True)
    
    max_seq_len = 300    
    
    # change this label_map based on datset
    domain_label_map = {'laptop':0, 'restaurant':1}
    sentiment_label_map = {-1:0, 0:1, 1:2}
    #inverse_domain_label_map = {val:key for key, val in domain_label_map.items()}
    domain_n_labels = 2
    sentiment_n_labels = 3

    params = {'aspect_n_labels':aspect_n_labels,'domain_n_labels':domain_n_labels,'sentiment_label_map':sentiment_label_map,'domain_label_map':domain_label_map,
              'batch_size':16, 'n_epochs':25, 'seq_len':max_seq_len, 'n_workers':-1, 'lr':0.0001, 'keep_prob':0.1, 'heads':12}


    #train_data = get_cross_aspect_batch(source_train_examples, domain_label_map, sentiment_label_map, max_seq_len, tokenizer, target_train_examples,
    #                                    output_mode="classification",label_available=True, batch_size=16, num_workers=-1)
    #dev_data = get_cross_aspect_batch(source_dev_examples, domain_label_map, sentiment_label_map, max_seq_len, tokenizer, target_dev_examples,
    #                                    output_mode="classification",label_available=True, batch_size=16, num_workers=-1)
    bert_cross_proc = BertCrossFeatures(tokenizer, max_seq_len, domain_label_map, sentiment_label_map)
    train_data = get_cross_aspect_batch(source_train_examples,target_train_examples,bert_cross_proc,label_available=True, batch_size=params['batch_size'], num_workers=-1)
    logging.info('train_data {}'.format(train_data[0]))
    dev_data = get_cross_aspect_batch(source_dev_examples,target_dev_examples,bert_cross_proc,label_available=True, batch_size=params['batch_size'], num_workers=-1)
    logging.info('dev_data {}'.format(dev_data[0]))
    
    bert_cross_aspect_model = BertForCrossAspect.from_pretrained('/home/kkzhang/bert_pytorch_model/bert_base', params)
   # bert_aspect_model = BertForAspect.from_pretrained('bert-base-uncased', params)

    #device = torch.device('cpu')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    if device!='cpu' and  torch.cuda.device_count()> 1:
      logging.info('has {} gpus'.format(torch.cuda.device_count()))
      bert_cross_aspect_model = nn.DataParallel(bert_cross_aspect_model)
            
    bert_cross_aspect_model.to(device)
    
    #loss_fn = F.cross_entropy
    loss_fn = None
    
    # maintain all metrics required in this dictionary- these are used in the training and evaluation loops
    metrics = {
        'accuracy': accuracy,
        # could add more metrics such as accuracy for each token type
    }

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, bert_cross_aspect_model.parameters()), lr=params['lr'])

    bert_cross_aspect_trainer = BertMultiTrainer(device,batch_size=params['batch_size'],n_epochs=params['n_epochs'],min_clip_val=-1.0, max_clip_val=1.0)
    bert_cross_aspect_trainer.train_and_evaluate(bert_cross_aspect_model,train_data,dev_data,optimizer,metrics,loss_fn=loss_fn,model_dir='./cross_aspect_results/')

    bert_aspect_pred = BertPredictor(device, model=bert_aspect_model, max_seq_length=max_seq_len, tokenizer=tokenizer, X_proc=ps_proc, target_int2label_dict=inverse_label_map,
                         target_label2int_dict=label_map)


if __name__ == '__main__':
  logging.basicConfig(level=logging.INFO)
  main()
