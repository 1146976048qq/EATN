from .text_processor_base import TextProcessorBase
from .text_tokenization import SentenceProcessor

class DocProcessor(TextProcessorBase):
    """
    This is intended to process docs as str, list of str and list of list of str
    the bottom level str must be separated by blank space
    It is a wrapper for sentence_processor
    """
    def __init__(self, lower=False, start_tok='<sos>', end_tok='<eos>', unk_tok='<unk>', pad_tok='<pad>',
                 use_start=False, use_end=False, use_unk=True, use_pad=True):
        super(DocProcessor, self).__init__()
        self.sent_proc = SentenceProcessor(lower=lower, start_tok=start_tok, end_tok=end_tok, unk_tok=unk_tok, 
                                           pad_tok=pad_tok,use_start=use_start,use_end=use_end,use_unk=use_unk,use_pad=use_pad)
        self.max_seq_len=0
        self.max_doc_len=0
        
    def flatten_doc(self, doc):
        """
        reduce the doc from level 3 to level 2
        Args:
            doc: list of list of str, level 3
        return:
            sents: list of str, level 2
        """
        import itertools
        return list(itertools.chain(*doc))

    def get_docs_sents_length_no_pad(self, doc):
        """
        just calculate the number of sents and seq_len in doc without any padding
        Args:
            doc: list of list of str, level 3
            max_doc_len: int
        return:
            docs_length: (N_samples, )
            sents_length: (N_samples, doc_len) just list of list, need to pad to max_doc_len
        """
        from collections import Iterable
        docs_length = [] # shape (batch,)
        sents_length = [] # shape (batch, max_doc_len)
        for sents in doc:
            # sents is list of str, level 2
            sents_length_ = self.sent_proc.get_sents_length(sents)
            # for now len(sents_length) is doc_len, number of sentences in doc[i] or sent
            docs_length.append(len(sents_length_))
            sents_length.append(sents_length_)
        return docs_length, sents_length
    
    def get_docs_sents_length(self, doc, max_doc_len):
        """
        just calculate the number of sents and seq_len in doc and pad sents_length to max_doc_len
        Args:
            doc: list of list of str, level 3
            max_doc_len: int
        return:
            docs_length: (N_samples, )
            sents_length: (N_samples, max_doc_len)
        """
        from collections import Iterable
        docs_length = [] # shape (batch,)
        sents_length = [] # shape (batch, max_doc_len)
        for sents in doc:
            # sents is list of str, level 2
            sents_length_ = self.sent_proc.get_sents_length(sents)
            # for now len(sents_length) is doc_len, number of sentences in doc[i] or sent
            docs_length.append(len(sents_length_))
            if len(sents_length_) >= max_doc_len:
                sents_length_ = sents_length_[:max_doc_len]
            else:
                sents_length_ = sents_length_ + [0]*max(0,(max_doc_len-len(sents_length_)))
            sents_length.append(sents_length_)
        return docs_length, sents_length

    def get_max_docs_sents_length(self, doc=None, docs_length=None, sents_length=None):
        import numpy as np
        if doc is not None:
            docs_length, sents_length = self.get_docs_sents_length_no_pad(doc) # sents_length is just list of list of int
        return np.max(docs_length), np.max([np.max(sen_len) for sen_len in sents_length])
    
    def get_params(self):
        """
        just copy params from sent_proc here
        """
        self.special_toks = self.sent_proc.special_toks
        self.end_tok = self.sent_proc.end_tok
        self.unk_tok = self.sent_proc.unk_tok
        self.pad_tok = self.sent_proc.pad_tok
        self.start_tok = self.sent_proc.start_tok
        self.num_special_toks = self.sent_proc.num_special_toks
        self.word_index_start_idx = self.sent_proc.word_index_start_idx 
        self.word2index = self.sent_proc.word2index
        self.index2word = self.sent_proc.index2word
        self.vocab_size = self.sent_proc.vocab_size
        self.max_seq_len = self.sent_proc.max_seq_len if self.max_seq_len==0 else self.max_seq_len
        self.max_doc_len = 1 if self.max_doc_len==0 else self.max_doc_len
        self.word_cnts = self.sent_proc.word_cnts
        
    def fit(self, doc):
        """
        Args:
            doc: str, list of str, list of list of str
        """
        level = self.check_doc_level(doc)
        self.print_doc_level(level)
        
        if level==0:
            print('please add corpus to be processed')
            return
        elif level>=1 and level<=2:
            # just call fit function in sentence_processor
            self.sent_proc.fit(doc)
        elif level==3:
            # flatten the docs to level 2 in order to get word2index
            self.sent_proc.fit(self.flatten_doc(doc))
            self.max_doc_len, self.max_seq_len = self.get_max_docs_sents_length(doc)
        self.get_params()
    
    def fit_transform(self, doc, max_seq_len=None, max_doc_len=None, add_start=False, add_end=False, 
                          return_sents_length=False,return_padded_length=False, return_array=True):
        self.fit(doc)
        return self.transform(doc, max_seq_len=max_seq_len, max_doc_len=max_doc_len, add_start=add_start, add_end=add_end,
                              return_sents_length=return_sents_length,return_padded_length=return_padded_length,
                              return_array=return_array)
    
    def transform(self, doc, max_seq_len=None, max_doc_len=None, add_start=False, add_end=False, 
                      return_sents_length=False,return_padded_length=False, return_array=True):
        import numpy as np
        assert (max_doc_len is not None or self.max_doc_len is not None), ('must specify a maxlen of docs')
        assert (max_seq_len is not None or self.max_seq_len is not None), ('must specify a maxlen of sequence')
        if max_seq_len is None: 
            max_seq_len = self.max_seq_len
        else:
            self.max_seq_len = max_seq_len
            
        if max_doc_len is None: 
            max_doc_len = self.max_doc_len
        else:
            self.max_doc_len = max_doc_len
            
        level = self.check_doc_level(doc)
        self.print_doc_level(level)
        
        if level==0:
            return []
        elif level>=1 and level<=2:
            # just call transform function in sentence_processor
            return self.sent_proc.transform(doc, max_seq_len=max_seq_len, add_start=add_start, add_end=add_end, 
                                            return_sents_length=return_sents_length,
                                            return_padded_length=return_padded_length, return_array=return_array)
        elif level==3:
            # doc is list of list of str
            # need to call transform function in loop and merge counts etc
            docs_seq_pad = []
            for sents in doc:
                sents_seq_padded = self.sent_proc.transform(sents, max_seq_len=max_seq_len, 
                                                            add_start=add_start, add_end=add_end,
                                                            return_sents_length=False,return_padded_length=return_padded_length,
                                                            return_array=False)
                docs_seq_pad.append(sents_seq_padded)
            
            docs_length, sents_length = self.get_docs_sents_length(doc, max_doc_len)
            
            self.batch_size = len(docs_seq_pad)
            if return_padded_length:
                docs_length, sents_length = self.get_padded_length(self.batch_size, max_seq_len, max_doc_len)
#             else:
#                 self.sents_length = self.pad_sents_length(self.sents_length, max_doc_len)

            docs_seq_pad = self.pad_doc(docs_seq_pad,max_seq_len,max_doc_len,add_start=add_start,add_end=add_end)
    
            if return_array:
                docs_seq_pad = self.shrink_dim(np.array(docs_seq_pad))
                docs_length = np.array(docs_length)
                sents_length = np.array(sents_length)
            
            if return_sents_length:
                return docs_seq_pad, docs_length, sents_length
            else:
                return docs_seq_pad
    
    def shrink_dim(self, docs_seq_pad):
        # docs_seq_pad is 3D 
        import numpy as np
        batch, doc_len, d = docs_seq_pad.shape
        if d > 1:
            return docs_seq_pad
        return np.reshape(docs_seq_pad, (batch, doc_len))
    
    def pad_sents_length(self, sents_length, max_doc_len):
        # fix sents_length to shape of (batch, max_doc_len)
        for i in range(len(sents_length)):
            if len(sents_length[i])>= max_doc_len:
                sents_length[i] = sents_length[i][:max_doc_len]
            else:
                sents_length[i] = sents_length[i] + [0]*(max_doc_len-len(sents_length[i]))
        return sents_length
    
    def get_padded_length(self, batch_size, max_seq_len, max_doc_len):
        docs_length = [max_doc_len]*batch_size
        sents_length = [[max_seq_len]*max_doc_len]*batch_size
        return docs_length, sents_length

    def pad_doc(self, docs_seq_pad, max_seq_len, max_doc_len, add_start=False, add_end=False):
        """
        Pad or truncate the doc to the same length
        Args:
            docs_seq_pad: sents has been padded to the same length
            max_seq_len: max_seq_length of each sentence
            max_doc_len: by default the max sents_seq_pad
            add_start: bool
            add_end: bool
        """
        for i, sents_seq in enumerate(docs_seq_pad):
            if len(sents_seq)>=max_doc_len:
                # just truncate the sents_seq
                docs_seq_pad[i] = sents_seq[:max_doc_len]
                continue
            
            while len(docs_seq_pad[i])<max_doc_len:
                extra_seq = []
                if add_start:
                    extra_seq.append(self.sent_proc.word2index[self.sent_proc.start_tok])
                extra_seq.extend([self.sent_proc.word2index[self.sent_proc.pad_tok]]*(max_seq_len-len(extra_seq)))     
                if add_end:
                    extra_seq.insert(max_seq_len-1,self.sent_proc.word2index[self.sent_proc.end_tok])
                #docs_seq_pad[i] = np.vstack((docs_seq_pad[i], np.array(extra_seq)))
                docs_seq_pad[i].append(extra_seq)

        return docs_seq_pad

    def inverse_transform(self, docs_seq_pad, reverse=False, keepdim=False):
        return self.decode_doc(docs_seq_pad, reverse=reverse)
            
    def decode_doc(self, docs_seq_pad, reverse=False, keepdim=False):
        level = self.check_doc_level(docs_seq_pad)
        self.print_doc_level(level)
        
        if level==0:
            return []
        elif level>=1 and level<=2:
            # just call fit function in sentence_processor
            return self.sent_proc.inverse_transform(docs_seq_pad, reverse=reverse, keepdim=keepdim)
        elif level==3:
            # need to call transform function in loop and merge counts etc
            doc = []
            for doc_seq in docs_seq_pad:
                decoded_seq_ = self.sent_proc.inverse_transform(doc_seq, reverse=reverse, keepdim=keepdim)
                if len(decoded_seq_)==0 or (len(decoded_seq_)==1 and len(decoded_seq_[0])==0):
                    break
                else:
                    doc.append(decoded_seq_)
            return doc