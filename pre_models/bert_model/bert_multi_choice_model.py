import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.nn import functional as F

from pytorch_pretrained_bert import BertModel, BertForMultipleChoice

class BertForChoice(BertForMultipleChoice):
    """BERT model for multiple choice tasks.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_choices`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, num_choices, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, num_choices, sequence_length]
            with the token types indices selected in [0, 1]. Type 0 corresponds to a `sentence A`
            and type 1 corresponds to a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, num_choices, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_choices].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[[31, 51, 99], [15, 5, 0]], [[12, 16, 42], [14, 28, 57]]])
    input_mask = torch.LongTensor([[[1, 1, 1], [1, 1, 0]],[[1,1,0], [1, 0, 0]]])
    token_type_ids = torch.LongTensor([[[0, 0, 1], [0, 1, 0]],[[0, 1, 1], [0, 0, 1]]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_choices = 2

    model = BertForMultipleChoice(config, num_choices)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, params):
        super(BertForMultipleChoice, self).__init__(config, params['n_choices'])
        self.n_choices = params['n_choices']
        
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1) # 1 or n_labels
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        """
        Args:
          input_ids: (batch, seq_len), word index of text, start with [CLS] and end with [SEP] token ids
          token_type_ids: (batch, seq_len), values from [0,1], indicates whether it's from sentence A(0) or B(1)
          attention_mask: (batch, seq_len), mask for input text, values from [0,1], 1 means word is padded
          labels: (batch), y 
        """
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        logging.info('flat_input_ids shape {}, flat_token_type_ids shape {}, flat_attention_mask shape {}'.format(flat_input_ids.shape, flat_token_type_ids.shape, flat_attention_mask.shape))
        
        _, pooled_output = self.bert(flat_input_ids, flat_token_type_ids, flat_attention_mask, output_all_encoded_layers=False)
        logging.info('pooled_output shape {}'.format(pooled_output.shape))
        pooled_output = self.dropout(pooled_output)
        
        # final prediction layer
        logits = self.classifier(pooled_output)
        self.logits = logits.view(-1, self.num_choices)

        if labels is not None:
            loss_fn = CrossEntropyLoss()
            self.loss = loss_fn(self.logits, labels)
            return self.loss
        else:
            return self.logits

    def get_loss(self):
        return self.loss
    
    def get_prediction_result(self):
        return self.prediction_result
    
    def get_logits(self):
        return self.logits
