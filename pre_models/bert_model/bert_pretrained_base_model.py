# import torch
# from torch import nn

# from pytorch_pretrained_bert import BertConfig


# PRETRAINED_MODEL_ARCHIVE_MAP = {
#     'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
#     'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
#     'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
#     'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
#     'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
#     'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
#     'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
# }
# BERT_CONFIG_NAME = 'bert_config.json'
# TF_WEIGHTS_NAME = 'model.ckpt'

# class BertPreTrainedModel(nn.Module):
#     """ An abstract class to handle weights initialization and
#         a simple interface for dowloading and loading pretrained models.
#     """
#     def __init__(self, config, *inputs, **kwargs):
#         super(BertPreTrainedModel, self).__init__()
#         if not isinstance(config, BertConfig):
#             raise ValueError(
#                 "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
#                 "To create a model from a Google pretrained model use "
#                 "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
#                     self.__class__.__name__, self.__class__.__name__
#                 ))
#         self.config = config

#     def init_bert_weights(self, module):
#         """ Initialize the weights.
#         """
#         if isinstance(module, (nn.Linear, nn.Embedding)):
#             # Slightly different from the TF version which uses truncated_normal for initialization
#             # cf https://github.com/pytorch/pytorch/pull/5617
#             module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
#         elif isinstance(module, BertLayerNorm):
#             module.bias.data.zero_()
#             module.weight.data.fill_(1.0)
#         if isinstance(module, nn.Linear) and module.bias is not None:
#             module.bias.data.zero_()

#     @classmethod
#     def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
#         """
#         Instantiate a BertPreTrainedModel from a pre-trained model file or a pytorch state dict.
#         Download and cache the pre-trained model file if needed.

#         Params:
#             pretrained_model_name_or_path: either:
#                 - a str with the name of a pre-trained model to load selected in the list of:
#                     . `bert-base-uncased`
#                     . `bert-large-uncased`
#                     . `bert-base-cased`
#                     . `bert-large-cased`
#                     . `bert-base-multilingual-uncased`
#                     . `bert-base-multilingual-cased`
#                     . `bert-base-chinese`
#                 - a path or url to a pretrained model archive containing:
#                     . `bert_config.json` a configuration file for the model
#                     . `pytorch_model.bin` a PyTorch dump of a BertForPreTraining instance
#                 - a path or url to a pretrained model archive containing:
#                     . `bert_config.json` a configuration file for the model
#                     . `model.chkpt` a TensorFlow checkpoint
#             from_tf: should we load the weights from a locally saved TensorFlow checkpoint
#             cache_dir: an optional path to a folder in which the pre-trained models will be cached.
#             state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
#             *inputs, **kwargs: additional input for the specific Bert class
#                 (ex: num_labels for BertForSequenceClassification)
#         """
#         state_dict = kwargs.get('state_dict', None)
#         kwargs.pop('state_dict', None)
#         cache_dir = kwargs.get('cache_dir', None)
#         kwargs.pop('cache_dir', None)
#         from_tf = kwargs.get('from_tf', False)
#         kwargs.pop('from_tf', None)

#         if pretrained_model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP:
#             archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name_or_path]
#         else:
#             archive_file = pretrained_model_name_or_path
#         # redirect to the cache, if necessary
#         try:
#             resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
#         except EnvironmentError:
#             logging.error(
#                 "Model name '{}' was not found in model name list ({}). "
#                 "We assumed '{}' was a path or url but couldn't find any file "
#                 "associated to this path or url.".format(
#                     pretrained_model_name_or_path,
#                     ', '.join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
#                     archive_file))
#             return None
#         if resolved_archive_file == archive_file:
#             logging.info("loading archive file {}".format(archive_file))
#         else:
#             logging.info("loading archive file {} from cache at {}".format(
#                 archive_file, resolved_archive_file))
#         tempdir = None
#         if os.path.isdir(resolved_archive_file) or from_tf:
#             serialization_dir = resolved_archive_file
#         else:
#             # Extract archive to temp dir
#             tempdir = tempfile.mkdtemp()
#             logging.info("extracting archive file {} to temp dir {}".format(
#                 resolved_archive_file, tempdir))
#             with tarfile.open(resolved_archive_file, 'r:gz') as archive:
#                 archive.extractall(tempdir)
#             serialization_dir = tempdir
#         # Load config
#         config_file = os.path.join(serialization_dir, CONFIG_NAME)
#         if not os.path.exists(config_file):
#             # Backward compatibility with old naming format
#             config_file = os.path.join(serialization_dir, BERT_CONFIG_NAME)
#         config = BertConfig.from_json_file(config_file)
#         logger.info("Model config {}".format(config))
#         # Instantiate model.
#         model = cls(config, *inputs, **kwargs)
#         if state_dict is None and not from_tf:
#             weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
#             state_dict = torch.load(weights_path, map_location='cpu')
#         if tempdir:
#             # Clean up temp dir
#             shutil.rmtree(tempdir)
#         if from_tf:
#             # Directly load from a TensorFlow checkpoint
#             weights_path = os.path.join(serialization_dir, TF_WEIGHTS_NAME)
#             return load_tf_weights_in_bert(model, weights_path)
#         # Load from a PyTorch state_dict
#         old_keys = []
#         new_keys = []
#         for key in state_dict.keys():
#             new_key = None
#             if 'gamma' in key:
#                 new_key = key.replace('gamma', 'weight')
#             if 'beta' in key:
#                 new_key = key.replace('beta', 'bias')
#             if new_key:
#                 old_keys.append(key)
#                 new_keys.append(new_key)
#         for old_key, new_key in zip(old_keys, new_keys):
#             state_dict[new_key] = state_dict.pop(old_key)

#         missing_keys = []
#         unexpected_keys = []
#         error_msgs = []
#         # copy state_dict so _load_from_state_dict can modify it
#         metadata = getattr(state_dict, '_metadata', None)
#         state_dict = state_dict.copy()
#         if metadata is not None:
#             state_dict._metadata = metadata

#         def load(module, prefix=''):
#             local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
#             module._load_from_state_dict(
#                 state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
#             for name, child in module._modules.items():
#                 if child is not None:
#                     load(child, prefix + name + '.')
#         start_prefix = ''
#         if not hasattr(model, 'bert') and any(s.startswith('bert.') for s in state_dict.keys()):
#             start_prefix = 'bert.'
#         load(model, prefix=start_prefix)
#         if len(missing_keys) > 0:
#             logger.info("Weights of {} not initialized from pretrained model: {}".format(
#                 model.__class__.__name__, missing_keys))
#         if len(unexpected_keys) > 0:
#             logger.info("Weights from pretrained model not used in {}: {}".format(
#                 model.__class__.__name__, unexpected_keys))
#         if len(error_msgs) > 0:
#             raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
#                                model.__class__.__name__, "\n\t".join(error_msgs)))
#         return model
     
#     def get_loss(self):
#       return self.loss
    
#     def get_prediction_result(self):
#       return self.prediction_result
    
#     def get_logits(self):
#       return self.logits