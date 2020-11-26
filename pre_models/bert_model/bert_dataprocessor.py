import pandas as pd
import os

#from .bert_datafeatures import InputFeatures, InputExample, convert_examples_to_features, _truncate_seq_pair
from .bert_datafeatures import InputExample

class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        import csv
        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                if sys.version_info[0] == 2:
                    line = list(unicode(cell, 'utf-8') for cell in line)
                lines.append(line)
            return lines        
        

class TextClfProcessor(DataProcessor):
    """
    This is the text processor for text classification task dataset.
    """
    def __init__(self, feature_columns='text', label_columns='label'):
        """
        Args:
          #data_dir: prefix of path of data files, 
          feature_columns: str or list, column names of features
          label_columns: str or list, column names of labels
        """
        self.feature_columns = feature_columns
        self.label_columns = label_columns
        self.labels = None
    
    def load_datafile(self, filename=None, df=None, data_type='train', size=-1, labels_available=True):
        assert (filename is not None or df is not None), ('Must specify filename or df to read data')
        
        if filename is not None:
          logging.info('Loading {} dataset : {}'.format(data_type, filename))
          df = pd.read_csv(filename)
        if size == -1:            
            return self._create_examples(df, data_type, labels_available)
        else:
            return self._create_examples(df.sample(size), data_type, labels_available)
    
    def get_train_examples(self, filename=None, df=None, size=-1, labels_available=True):
        return self.load_datafile(filename, df, 'train', size, labels_available)
        
    def get_dev_examples(self, filename=None, df=None, size=-1, labels_available=True):
        return self.load_datafile(filename, df, 'dev', size, labels_available)
    
    def get_test_examples(self, filename=None, df=None, size=-1, labels_available=False):
        return self.load_datafile(filename, df, 'test', size, labels_available)

    def get_labels(self):
        if self.labels == None:
            self.labels = list(pd.read_csv(os.path.join(self.data_dir, "classes.txt"),header=None)[0].values)
        return self.labels
      
    def _create_examples(self, df, set_type, labels_available=True):
        examples = []
        for index, row in df.iterrows():
          text_a = row.get(self.feature_columns, '')
          label = ''
          if labels_available:
            label = row.get(self.label_columns, '')
          
          examples.append(InputExample(guid=index, text_a=text_a, label=label))
        return examples
          

class TextNERProcessor(TextClfProcessor):
    """
    This is the NER classification text processor.
    """
    def __init__(self, labels, feature_columns='text', label_columns='tag'):
      """
      Args:
        feature_columns: str or list, column names of features
        label_columns: str or list, column names of labels
        labels: list of all ner labels
      """
      super(TextNERProcessor, self).__init__(feature_columns, label_columns)
      self.labels = labels
      
    def get_labels(self):
        return self.labels
      
    def _create_examples(self, df, set_type, labels_available=True):
        examples = []
        for index, row in df.iterrows():
            text_a = row.get(self.feature_columns, '')
            label = ''
            if labels_available:
                label = row.get(self.label_columns, '')
            
            examples.append(InputExample(guid=index, text_a=text_a, label=label))
        return examples
      
      
class TextPhraseSimProcessor(TextClfProcessor):
    """
    This is the phrase sim comparision text processor
    """
    def __init__(self, labels=None, feature_columns=['sentence_A', 'sentence_B'], label_columns='label'):
      """
      Args:
        feature_columns: str or list, column names of features
        label_columns: str or list, column names of labels
        labels: list of all ner labels
      """
      super(TextPhraseSimProcessor, self).__init__(feature_columns, label_columns)
      self.labels = labels
      
    def get_labels(self):
      return self.labels
    
    def _create_examples(self, df, set_type, labels_available=True):
        examples = []
        for index, row in df.iterrows():
            text_a = row.get(self.feature_columns[0], '')
            text_b = row.get(self.feature_columns[1], '')
            label = ''
            if labels_available:
                label = row.get(self.label_columns, '')
            examples.append(InputExample(guid=index, text_a=text_a, text_b=text_b, label=label))              
        return examples
      
      