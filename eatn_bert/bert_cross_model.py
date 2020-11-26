import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import functional as F
import logging
from pytorch_pretrained_bert import BertModel, BertForSequenceClassification

from deepqa_models.transformer.transformer import ScaledDotProductAttention, clones, MultiHeadAttention, LayerNorm

class BertForCrossAspect(BertForSequenceClassification):
    """
    Bert For aspect cross domain multi Task
    """
    def __init__(self, config, params):
        super(BertForCrossAspect, self).__init__(config, params['sentiment_n_labels'])
        self.sentiment_n_labels = params['sentiment_n_labels']
        self.domain_n_labels = params['domain_n_labels']
        
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.domain_classifier = nn.Linear(config.hidden_size, self.domain_n_labels)
        self.sentiment_classifier = nn.Linear(config.hidden_size, self.sentiment_n_labels)
        self.apply(self.init_bert_weights)
        self.multiheads = MultiHeadAttention(config.hidden_size, params['heads'], keep_prob=params['keep_prob'])
        
    def get_bert_encoding(self, input_ids, token_type_ids=None, attention_mask=None):
        """
        Args:
          input_ids: (batch, seq_len), word index of text, start with [CLS] and end with [SEP] token ids
          token_type_ids: (batch, seq_len), values from [0,1], indicates whether it's from sentence A(0) or B(1)
          attention_mask: (batch, seq_len), mask for input text, values from [0,1], 1 means word is padded
          labels: (batch), y 
        """
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                     output_all_encoded_layers=False)
        logging.debug('bert for aspect: sequence_output {}'.format(sequence_output.shape))    

        pooled_output = self.dropout(sequence_output)
        pooled_output = self.multiheads(pooled_output)
        logging.debug('bert for aspect: multihead pooled_output shape {}'.format(pooled_output.shape))
        pooled_output = pooled_output[:,0,:]
        logging.debug('bert for aspect: pooled_out {}'.format(pooled_output.shape))
        return pooled_output
        
    def forward(self, source_input_ids, source_token_type_ids=None, source_attention_mask=None, source_domain_labels=None, source_sentiment_labels=None,
               target_input_ids=None, target_token_type_ids=None, target_attention_mask=None, target_domain_labels=None, target_sentiment_labels=None,
               is_training=False):
        """
        Args:
          input_ids: (batch, seq_len), word index of text, start with [CLS] and end with [SEP] token ids
          token_type_ids: (batch, seq_len), values from [0,1], indicates whether it's from sentence A(0) or B(1)
          attention_mask: (batch, seq_len), mask for input text, values from [0,1], 1 means word is padded
          labels: (batch), y 
        """
        # source inputs
        source_output = self.get_bert_encoding(source_input_ids, source_token_type_ids, source_attention_mask)
        logging.debug('source_output {}'.format(source_output))
        
        # target inputs
        if target_input_ids is not None:
          target_output = self.get_bert_encoding(target_input_ids, target_token_type_ids, target_attention_mask)
          logging.debug('target_output {}'.format(target_output))
          target_domain_logits = self.domain_classifier(target_output)
      
        # sentiment classifier
        source_sentiment_logits = self.sentiment_classifier(source_output)
        
        if is_training:
          # domain classifier
          source_domain_logits = self.domain_classifier(source_output)
          logging.debug('source_domain_logits {}'.format(source_domain_logits))
        
          # domain loss
          source_domain_prediction = F.softmax(source_domain_logits, dim=-1)
          target_domain_prediction = F.softmax(target_domain_logits, dim=-1)
          source_sentiment_prediction = F.softmax(source_sentiment_logits, dim=-1)
          
          domain_loss = F.cross_entropy(source_domain_logits, source_domain_labels.view(-1)) + F.cross_entropy(target_domain_logits, target_domain_labels.view(-1))
          sentiment_loss = F.cross_entropy(source_sentiment_logits, source_sentiment_labels.view(-1))
          loss = domain_loss + sentiment_loss
          #return loss, source_domain_logits, target_domain_logits, source_aspect_logits
          return loss, source_domain_prediction, target_domain_prediction, source_sentiment_prediction
          
        else:
          return source_sentiment_logits
