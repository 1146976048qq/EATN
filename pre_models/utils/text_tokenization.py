from .text_processor_base import TextProcessorBase

class SentenceProcessor(TextProcessorBase):
    """
    Conventions: 
        inputs: str or list of str
        str must be seperated by blank space
    preprocess the text corpus, create word2index, index2word map, pad sentence, 
    transform text to wordindex sequences, decode sequences to original raw text
    """
    def __init__(self, lower=False, start_tok='<sos>', end_tok='<eos>', unk_tok='<unk>', pad_tok='<pad>',
                 use_start=False, use_end=False, use_unk=True, use_pad=True):
        super(SentenceProcessor, self).__init__()
        self.lower = lower # convert english words to lower case
        
        self.use_start = use_start
        self.start_tok = start_tok if use_start else None
        self.use_end = use_end
        self.end_tok = end_tok if use_end else None
        self.use_unk = use_unk 
        self.unk_tok = unk_tok if use_unk else None
        self.use_pad = use_pad 
        self.pad_tok = pad_tok if use_pad else None
        
        self.special_toks = []
        self.add_special_toks()
        self.word2index = {}
        self.word_index_start_idx = self.add_special_word_index()
        self.num_special_toks = self.word_index_start_idx
        self.special_tok_end_idx = max(0,self.word_index_start_idx-1)
        self.max_seq_len = 0
    
    def add_special_toks(self):
        if self.use_pad:
            self.special_toks.append(self.pad_tok)
        if self.use_start:
            self.special_toks.append(self.start_tok)
        if self.use_end:
            self.special_toks.append(self.end_tok)
        if self.use_unk:
            self.special_toks.append(self.unk_tok)
        
    def add_special_word_index(self):
        if len(self.special_toks)==0:
            return 0
        for i, tok in enumerate(self.special_toks):
            self.word2index[tok] = i
        return len(self.word2index)
    
    def get_word_counts(self, sents):
        """
        calculate the counts of each word token
        level 2: multiple sentences
        Args:
            sents: list of str
        """
        from collections import defaultdict
        
        self.sents_length = []
        self.word_cnts = defaultdict(int)
        for sent in sents:
            words = sent.split()
            self.sents_length.append(len(words))
            for word in words:
                if self.lower:
                    word = word.lower()
                self.word_cnts[word] += 1        
        return dict(self.word_cnts)
    
    def get_sents_length(self, sents):
        """
        calculate sentences length
        level 2: multiple sentences
        Args:
            sents: list of str
        """
        sents_length = []
        for sent in sents:
            sents_length.append(len(sent.split()))
        return sents_length

    def get_total_word_counts(self):
        self.total_word_counts = 0
        for value in self.word_cnts.values():
            self.total_word_counts += value
        return self.total_word_counts
    
    def get_word_ratio(self):
        self.word_ratio = {word:value*1.0/self.total_word_counts for word, value in self.word_cnts.items()}
        return self.word_ratio
    
    def get_word2index(self):
        code = self.word_index_start_idx
        for word in self.word_cnts.keys():
            self.word2index[word] = code
            code += 1
        
    def get_index2word(self):
        self.index2word = {code:word for word, code in self.word2index.items()}
    
    def get_vocab_size(self):
        self.vocab_size = len(self.word2index.keys())
        return self.vocab_size
    
    def get_vocab(self, include_special_toks=True):
        if include_special_toks:
            return list(self.word2index.keys())
        else:
            return list(self.word_cnts.keys())
        
    def get_max_sents_length(self, sents=None, return_sents_length=False):
        import numpy as np
        if sents:
            sents_length = self.get_sents_length(sents)
            if return_sents_length:
                return np.max(sents_length), docs_length
            else:
                return np.max(sents_length)
        else:
            if return_sents_length:
                return np.max(self.sents_length), self.sents_length
            else:
                return np.max(self.sents_length)

    def fit(self, sents):
        level = self.check_doc_level(sents)
        self.print_doc_level(level)
        if level==0:
            print('Please add some text data to be processed')
            return 
        if level == 1:
            sents = self.expand_doc_level(sents)
        
        self.get_word_counts(sents)
        self.max_seq_len = self.get_max_sents_length()
        self.get_total_word_counts()
        self.get_word_ratio()
        self.get_word2index()
        self.get_index2word()
        self.get_vocab_size()
        return level
        
    def fit_transform(self, sents, max_seq_len=0, add_start=False, add_end=False, return_sents_length=False,
                      return_padded_length=False, return_array=True):
        level = self.fit(sents)
        return self.transform(sents,max_seq_len=max_seq_len, add_start=add_start, add_end=add_end, 
                              return_sents_length=return_sents_length, return_padded_length=return_padded_length,
                              return_array=return_array)
        
    def get_padded_sents_length(self, batch_size, max_seq_len=0):
        """
        Args: 
            batch_size: int
            max_doc_len: int, the length that all sentences are fixed to
        """
        return [max_seq_len]*batch_size

    def transform(self, sents, max_seq_len=0, add_start=False, add_end=False, return_sents_length=False,
                  return_padded_length=False, return_array=True):
        """
        convert sentences to word index
        pad or truncate sentences to a fixed seq_len
        Args:
            sents: level 2, list of str
            max_seq_len: the fixed length that all sentences must be converted to
            add_start: add start_token in the beginning of sentence
            add_end: add end_token at the end of sentence
            return_sents_length: bool
            return_padded_length: bool
            return_array: bool
        """
        import numpy as np
        #print('sent_proc transform ')
        sents_seq, sents_length = self.texts_to_sequences(sents)
        #print('sent_proc text_to_sequence ')
        sents_seq_padded = self.pad_sequences(sents_seq, max_seq_len=max_seq_len, add_start=add_start, add_end=add_end)
        #print('sent_proc pad_sequence ')
        
        if return_padded_length:
            sents_length = self.get_padded_sents_length(len(sents), max_seq_len)
        if return_array:
            sents_seq_padded = np.array(sents_seq_padded)
            sents_length = np.array(sents_length)
        #print('sent_proc if pad ')
        
        if return_sents_length:
            return sents_seq_padded, sents_length
        else:
            return sents_seq_padded

    def texts_to_sequences(self, sents):
        """
        Args:
            sents: level 2, list of str
        """
        sents_seq = []
        sents_length = [] # actual length of each sentence
        for sent in sents:
            seq = []
            words = sent.split()
            sents_length.append(len(words))
            for word in words:
                if self.lower:
                    word = word.lower()
                if word in self.word2index.keys():
                    seq.append(self.word2index[word])
                else:
                    seq.append(self.word2index[self.unk_tok])
            sents_seq.append(seq)
        return sents_seq, sents_length
    
    def pad_sequences(self, sents_seq, max_seq_len=0, add_start=False, add_end=False):
        """
        Args:
            sents_seq: list of list of int
            max_seq_len: the fixed length that all sentences must be converted to
            add_start: add start_token in the beginning of sentence
            add_end: add end_token at the end of sentence
        """
        assert (max_seq_len!=0 or self.max_seq_len!=0), ('must specify a max_seq_len of sequence')
        if max_seq_len==0: 
            max_seq_len = self.max_seq_len
        else:
            self.max_seq_len = max_seq_len
        
        for i, sent in enumerate(sents_seq):
            if add_start:
                if not self.start_tok:
                    raise ValueError('No start tok is specified, please add enable use_start option')
                sent.insert(0, self.word2index[self.start_tok])
            if add_end:
                end_pos = min([max_seq_len-1, len(sentence)])
                if not self.end_tok:
                    raise ValueError('No end tok is specified, please add enable use_end option')
                sent.insert(end_pos, self.word2index[self.end_tok])
            if len(sent) > max_seq_len:
                sent = sent[:max_seq_len]
            sent += [self.word2index[self.pad_tok]]*(max_seq_len-len(sent))
            sents_seq[i] = sent
        return sents_seq
    
    def inverse_transform(self, sents_seq, reverse=False, keepdim=False):
        return self.decode_sequences(sents_seq, reverse=reverse, keepdim=keepdim)
    
    def decode_sequences(self, sents_seq, reverse=False, keepdim=False):
        """
        decode docs_seq from word2index to inde2word
        if reverse==True, docs_seq ends with start_tok
        otherwise, it starts from start_tok
        """
        from collections import Iterable
        if len(sents_seq)>0 and (not isinstance(sents_seq[0], Iterable)):
            sents_seq = [list(sents_seq)]
        sents = []
        for i, sentence in enumerate(sents_seq):
            if reverse:
                sentence.reverse()
            #sentence.pop(0) #remove start_tok
            words = []
            for word_index in sentence:
                word = self.index2word[word_index]
                if  word == self.end_tok:
                    break
                if word in self.special_toks:
                    continue
                words.append(word)
            if not keepdim:
                sents.append(' '.join(words))    
            else:
                sents.append(words)
        return sents