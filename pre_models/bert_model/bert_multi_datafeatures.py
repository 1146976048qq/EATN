import logging
import pandas as pd
import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from .bert_dataloader import BertDataset

class CrossAspectExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, domain_label=None, aspect_label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.domain_label = domain_label
        self.aspect_label = aspect_label


class CrossAspectFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, input_mask, segment_ids, domain_label_ids, aspect_label_ids=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.domain_label_ids = domain_label_ids
        self.aspect_label_ids = aspect_label_ids

        
class BertCrossFeatures(object):
    def __init__(self, tokenizer, max_seq_len=None, domain_label_proc=None, aspect_label_proc=None):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer
        self.domain_label_proc = domain_label_proc
        self.aspect_label_proc = aspect_label_proc
        
    def pad_sentence(self, tokens_a, tokens_b):
        """
        Args:
          token_a: str 
          token_b: str
        """
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= self.max_seq_len-3:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()
        return tokens_a, tokens_b
    
    def convert_single_words2id(self, example):
        """
        Args:
          example: single example
        """
        if example is None:
            return None
          
        tokens_a = self.tokenizer.tokenize(example.text_a)
        tokens_b = None
        if example.text_b:
            tokens_b = self.tokenizer.tokenize(example.text_b)
            tokens_a, tokens_b = self.pad_sentence(tokens_a, tokens_b)
        else:
            if len(tokens_a) > self.max_seq_len - 2:
                tokens_a = tokens_a[:self.max_seq_len-2]

        # process text
        ## add special tokens, [CLS], [SEP]
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        ## The mask has 1 for real tokens and 0 for padding tokens. Only real
        ## tokens are attended to.
        input_mask = [1] * len(input_ids)

        ## Zero-pad up to the sequence length.
        padding = [0] * (self.max_seq_len - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == self.max_seq_len
        assert len(input_mask) == self.max_seq_len
        assert len(segment_ids) == self.max_seq_len
        
        return input_ids, input_mask, segment_ids
      
    def convert_single_label2id(self, example):
        """
        Args:
            example: single example
        """
        if example is None:
            return None
          
        domain_label_id = None
        aspect_label_id = None
        if example.domain_label is not None:
            #domain_label_id = self.domain_label_proc.label2int_dict.get(example.domain_label, -1)
            domain_label_id = self.domain_label_proc.get(example.domain_label, 0)
        if example.aspect_label is not None:
            #aspect_label_id = self.aspect_label_proc.label2int_dict.get(example.aspect_label, -1)
            aspect_label_id = self.aspect_label_proc.get(example.aspect_label, 0)

        return domain_label_id, aspect_label_id
        
    def convert_sentence2id(self, examples):
        """
        Args:
          examples: struct, include text info
        """
        features = []
        for idx, example in enumerate(examples):
            if idx % 1000 == 0:
                logging.info('processing example {}, {}'.format(idx, example))
                
            # process single example sentence
            input_ids, input_mask, segment_ids = self.convert_single_words2id(example)
                
            # process single example label
            domain_label_id,aspect_label_id = self.convert_single_label2id(example)
                
            features.append(CrossAspectFeatures(input_ids=input_ids,
                                                input_mask=input_mask,
                                                segment_ids=segment_ids,
                                                domain_label_ids=domain_label_id,
                                                aspect_label_ids=aspect_label_id))

        print('features in total {}'.format(len(features)))
        return features
                
    def fit(self, examples):
        #if self.label_proc is None:
        #    self.label_proc = self.initialize_label_proc(examples)
        return self.convert_sentence2id(examples)
        
    def transform(self, examples):
        return self.convert_sentence2id(examples)
      
    def initialize_label_proc(self, examples):
        """
        Args:
          example: struct
        """
        pass


class CrossAspectProcessor(object):
    """
    This is processor for cross domain aspect dataset
    """
    def __init__(self, labels=None, source_feature_columns=['sentence','aspect'], source_aspect_labels=['label'],
                target_feature_columns=['sentence','aspect'], target_aspect_labels=['label']):
        #super(CrossAspectProcessor, self).__init__(source_feature_columns, source_aspect_labels)
        self.source_feature_columns = source_feature_columns
        self.source_aspect_labels = source_aspect_labels
        self.target_feature_columns = target_feature_columns
        self.target_aspect_labels = target_aspect_labels

    def load_datafile(self, filename=None, df=None, data_type='train', size=-1, labels_available=True, domain_class='laptop'):
        assert (filename is not None or df is not None), ('Must specify filename or df to read data')
        logging.info('load_datafile: labels_available {}'.format(labels_available))

        if filename is not None:
          logging.info('Loading {} dataset : {}'.format(data_type, filename))
          df = pd.read_csv(filename)
        if size == -1:            
            return self._create_examples(df, data_type, labels_available, domain_class)
        else:
            return self._create_examples(df.sample(size), data_type, labels_available, domain_class)
    
    def get_train_examples(self, source_filename=None, source_df=None, target_filename=None, target_df=None, size=-1, labels_available=True):
        source_examples = self.load_datafile(source_filename, source_df, 'train', size, labels_available)
        target_examples = self.load_datafile(target_filename, target_df, 'train', size, labels_available)
        return source_examples, target_examples

    def get_dev_examples(self, source_filename=None, source_df=None, target_filename=None, target_df=None, size=-1, labels_available=True):
        source_examples = self.load_datafile(source_filename, source_df, 'dev', size, labels_available)
        target_examples = self.load_datafile(target_filename, target_df, 'dev', size, labels_available)
        return source_examples, target_examples
    
    def get_test_examples(self, filename=None, df=None, size=-1, labels_available=False):
        return self.load_datafile(filename, df, 'test', size, labels_available)

    # def get_labels(self):
    #       return self.labels
    
    def _create_examples(self, df, set_type='source', labels_available=True, domain_class='laptop'):
        examples = []
        for index, row in df.iterrows():
            text_a, text_b = '',''
            aspect_label = ''
            if set_type == 'source':
                text_a = row.get(self.source_feature_columns[0], '')
                text_b = row.get(self.source_feature_columns[1], '')
                if labels_available:
                    aspect_label = row.get(self.source_label_columns, '')
            elif set_type == 'target':
                text_a = row.get(self.target_feature_columns[0], '')
                text_b = row.get(self.target_feature_columns[1], '')
                if labels_available:
                    aspect_label = row.get(self.target_label_columns, '')
            domain_label = domain_class

            examples.append(CrossAspectExample(guid=index, text_a=text_a, text_b=text_b, domain_label=domain_label, aspect_label=aspect_label))              
        return examples


def get_cross_aspect_batch(source_examples, target_examples, bert_input_proc,
              label_available=True, batch_size=32, num_workers=-1):
    """
    Args:
        data_examples: examples from DataProcessor get_*_examples
        #label_list: list of all labels
        label_map: dict, {label:label_index}
        max_seq_length: int, fixed length that sentences are converted to
        tokenizer: BertTokenizer
        output_mode: task mode, whether it is classification or regression
        label_availabel: True, whether there is label in dataset
        batch_size: int
        num_workers: int, for distributed training
    return:
        DataLoader
    """
    #source_data = convert_cross_aspect_examples_to_features(source_examples, domain_label_map, aspect_label_map, max_seq_length, tokenizer, output_mode)
    source_data = bert_input_proc.fit(source_examples)
    logging.info('source_data shape {}'.format(len(source_data)))

    # loop over data
    # to do: think an efficient way to process features
    source_input_ids = [f.input_ids for f in source_data]
    source_input_mask = [f.input_mask for f in source_data]
    source_segment_ids = [f.segment_ids for f in source_data]
    
    if target_examples is not None:
        #target_data = convert_cross_aspect_examples_to_features(target_examples, domain_label_map, aspect_label_map, max_seq_length, tokenizer, output_mode)
        target_data = bert_input_proc.fit(target_examples)
        logging.info('target_data shape {}'.format(len(target_data)))
        
        target_input_ids = [f.input_ids for f in target_data]
        target_input_mask = [f.input_mask for f in target_data]
        target_segment_ids = [f.segment_ids for f in target_data]

    if label_available:
        source_domain_label_ids = [f.domain_label_ids for f in source_data]
        source_aspect_label_ids = [f.aspect_label_ids for f in source_data]

        if target_examples is not None:
            target_domain_label_ids = [f.domain_label_ids for f in target_data]
            target_aspect_label_ids = [f.aspect_label_ids for f in target_data]

        # for train and dev dataset
        data_set = BertDataset(source_input_ids, source_input_mask, source_segment_ids, source_domain_label_ids, source_aspect_label_ids, 
                            target_input_ids, target_input_mask, target_segment_ids, target_domain_label_ids, target_aspect_label_ids)
      
        print('data_set shape {}'.format(data_set.shape))

        # use sampler
        if num_workers == -1:
            data_sampler = RandomSampler(data_set)
        else:
            data_sampler = DistributedSampler(data_set)
    
    else:
      # for test dataset
      data_set = BertDataset(source_input_ids, source_input_mask, source_segment_ids, source_domain_label_ids, source_aspect_label_ids, 
                            target_input_ids=None, target_input_mask=None, target_segment_ids=None, target_domain_label_ids=None, target_aspect_label_ids=None)
      #data_set = BertDataset(input_ids, input_mask, segment_ids)
      #data_sampler = SequentialSampler(data_set)
      data_sampler = None
    
    return DataLoader(data_set, sampler=data_sampler, batch_size=batch_size)    
