from .bert_dataprocessor import TextClfProcessor

class BertPredictor(object):
    """
    It supports predicting label(int), label(str) and label_proba from a model
    """
    def __init__(self, device, model, max_seq_length, tokenizer, X_proc=None,
                 target_int2label_dict=None, target_label2int_dict=None, batch_size=16, num_workers=-1):
        """
        Args:
          device: cuda or cpu
          model: saved or pretrained model
          max_seq_length: int
          X_proc: DataProcessor
          tokenizer: BertTokenizer
          target_int2label_dict: dict, {label: label_index}
          target_label2int_dict: dict, {label_indx: label}
          batch_size: 16
          num_workers: -1
        """
        #super(Predictor,self).__init__()
        self.device = device        
        self.model = model.to(device)
        self.model.device = device
        self.X_proc = X_proc
        self.target_int2label_dict = target_int2label_dict
        #self.label_list = target_int2label_dict.keys() 
        self.label_map = target_label2int_dict
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.num_workers = num_workers        
        
    def preprocess_sentences(self, filename, df):
      if self.X_proc is None:
        self.X_proc = TextClfProcessor(filename)
      test_examples = self.X_proc.get_test_examples(filename, df, size=-1, labels_available=False)
      
      # Hold input data for returning it 
      #input_data = [{ 'id': input_example.guid, 'comment_text': input_example.text_a } for input_example in test_examples]
      
      test_data = get_batch(test_examples, self.label_map, self.max_seq_length, self.tokenizer, output_mode="classification", 
                            label_available=False, batch_size=self.batch_size, num_workers=self.num_workers)
      return test_data
    
    def predict_proba(self, filename=None, df=None):
        """
        Args:
          filename: str, path to test_data
        
        returns: 
          probability: predicted probability of each label 
        """
        import logging
        from tqdm import trange, tqdm
        
        test_data = self.preprocess_sentences(filename, df)
               
        # set eval mode for the model
        self.model.eval()
        
        with torch.no_grad():
            for i, batch in enumerate(tqdm(test_data, desc='predict_data_iter')):
                batch = tuple(x.to(self.device) for x in batch)
                input_ids, input_mask, segment_ids = batch
                
                logits = self.model(input_ids, input_mask, segment_ids)
                probability = self.model.get_prediction_result() # (batch, n_labels) after activation
                logging.info('logits shape {}, probability shape {}'.format(logits.shape, probability.shape))
                
            return probability
            
    def predict(self, filename=None, df=None, single_label=True, thres=None):
        """
        Args:
            filename: str, path to test_data
            single_label: bool, whether it is a single or multiple label prediction
            thres: float, manually set threshold to select a predicted label
        returns:
            pred_label: actual labels for each data
            pred_label_ids: label index for each data
            probability: float probability of each label 
        """
        # probability gives probability for all labels classes
        probability = self.predict_proba(filename, df)
        pred_label, pred_probability = self.get_prediction_labels(probability, single_label, thres, use_target_int2label_dict=True)
        return pred_label, pred_probability, probability
        
    def get_label_from_dict(self, pred_pos, use_target_int2label_dict=False):
        """
        Args:
            pred_pos: int, index with highest probability
            use_target_int2label_dict: bool
        """
        pred_label = pred_pos
        if not use_target_int2label_dict:
            return pred_label
        
        if self.target_int2label_dict:
            try:
                pred_label = self.target_int2label_dict[pred_pos]
            except:
                pred_label = self.target_int2label_dict[str(pred_pos)]
        return pred_label
    
    def get_prediction_labels(self, pred_proba, single_label=True, thres=None, use_target_int2label_dict=False):
        """
        Args: 
            pred_proba: pred from model
            target_int2word_dict: target, int2label dict
            single_label: True -> single label prediction else -> predict multiple labels 
            thres: thres to choose the correct label
        return: 
            predicted labels for test_data
        """
        pred_labels = []
        pred_probability = []
        for i in range(len(pred_proba)):
            if single_label: # single label prediction 
                pred_pos = pred_proba[i].argmax().item()
                pred_val = pred_proba[i].max().item()
                pred_probability.append(pred_val)
                if (thres is None) or ((thres is not None) and (pred_val >= thres)):
                    pred_labels.append(self.get_label_from_dict(pred_pos, use_target_int2label_dict))
            else:
                multi_labels_ = []
                multi_pred_ = []
                for j in range(len(pred_proba[i])):
                    if (thres is None) or ((thres is not None) and (pred_proba[i][j] >= thres)):
                        multi_labels_.append(self.get_label_from_dict(j, use_target_int2label_dict))
                        multi_pred_.append(pred_proba[i][j])
                pred_labels.append(multi_labels_)
                pred_probability.append(multi_pred_)
        if pred_proba.shape[0] == 1:
            return pred_labels[0], pred_probability[0]
        else: 
            return pred_labels, pred_probability
