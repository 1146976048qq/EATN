import torch
from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

#from .bert_datafeatures import convert_examples_to_features

class BertDataset(TensorDataset):
    def __init__(self, *tensors):
        tensors = tuple(self.convert_datatype(data_) for data_ in tensors)
        super(BertDataset, self).__init__(*tensors)
    
    def check_datatype(self, data_tensor):
        return type(data_tensor)==type(torch.tensor([1,2]))
        
    def convert_datatype(self, data_tensor):
        """
        Convert data_tensor to tensor.LongTensor()
        """
        if not self.check_datatype(data_tensor):
            return torch.LongTensor(data_tensor)
        else:
            return data_tensor
   
    def shape(self):
        return self.data_tensor[0].shape    
    
    
def get_batch(data_examples, bert_input_proc,
              label_available=True,batch_size=16,num_workers=-1):
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
    #data = convert_examples_to_features(data_examples, label_map, max_seq_length, tokenizer, output_mode)
    #bert_x_proc = BertInputFeatures(tokenizer, max_seq_len, label_proc)
    data = bert_input_proc.fit(data_examples)
    
    # loop over data
    # to do: think an efficient way to process features
    input_ids = [f.input_ids for f in data]
    input_mask = [f.input_mask for f in data]
    segment_ids = [f.segment_ids for f in data]
    
    if label_available:
      label_id = [f.label_id for f in data]
      # for train and dev dataset
      data_set = BertDataset(input_ids, input_mask, segment_ids, label_id)
      
      # use sampler
      if num_workers == -1:
        data_sampler = RandomSampler(data_set)
      else:
        data_sampler = DistributedSampler(data_set)
    
    else:
      # for test dataset
      data_set = BertDataset(input_ids, input_mask, segment_ids)
      #data_sampler = SequentialSampler(data_set)
      data_sampler = None
    
    return DataLoader(data_set, sampler=data_sampler, batch_size=batch_size)    
  


# import torch
# from torch.utils.data import Dataset, TensorDataset, DataLoader, RandomSampler, SequentialSampler
# from torch.utils.data.distributed import DistributedSampler

# from .bert_datafeatures import convert_examples_to_features

# class BertDataset(TensorDataset):
#     def __init__(self, *tensors):
#         tensors = tuple(self.convert_datatype(data_) for data_ in tensors)
#         super(BertDataset, self).__init__(*tensors)
    
#     def check_datatype(self, data_tensor):
#         return type(data_tensor)==type(torch.tensor([1,2]))
        
#     def convert_datatype(self, data_tensor):
#         """
#         Convert data_tensor to tensor.LongTensor()
#         """
#         if not self.check_datatype(data_tensor):
#             return torch.LongTensor(data_tensor)
#         else:
#             return data_tensor
   
#     def shape(self):
#         return self.data_tensor[0].shape    
    
    
# def get_batch(data_examples, label_map, max_seq_length, tokenizer, output_mode="classification",
#               label_available=True, batch_size=32, num_workers=-1):
#     """
#     Args:
#         data_examples: examples from DataProcessor get_*_examples
#         #label_list: list of all labels
#         label_map: dict, {label:label_index}
#         max_seq_length: int, fixed length that sentences are converted to
#         tokenizer: BertTokenizer
#         output_mode: task mode, whether it is classification or regression
#         label_availabel: True, whether there is label in dataset
#         batch_size: int
#         num_workers: int, for distributed training
#     return:
#         DataLoader
#     """
#     data = convert_examples_to_features(data_examples, label_map, max_seq_length, tokenizer, output_mode)
    
#     # loop over data
#     # to do: think an efficient way to process features
#     input_ids = [f.input_ids for f in data]
#     input_mask = [f.input_mask for f in data]
#     segment_ids = [f.segment_ids for f in data]
    
#     if label_available:
#       label_id = [f.label_id for f in data]
#       # for train and dev dataset
#       data_set = BertDataset(input_ids, input_mask, segment_ids, label_id)
      
#       # use sampler
#       if num_workers == -1:
#         data_sampler = RandomSampler(data_set)
#       else:
#         data_sampler = DistributedSampler(data_set)
    
#     else:
#       # for test dataset
#       data_set = BertDataset(input_ids, input_mask, segment_ids)
#       data_sampler = SequentialSampler(data_set)
    
#     return DataLoader(data_set, sampler=data_sampler, batch_size=batch_size)    
  