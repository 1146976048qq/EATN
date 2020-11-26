from .text_processor_base import TextProcessorBase

class TextNormalization(TextProcessorBase):
    def __init__(self, stop_words='./data/chinese_punct.json'):
        super(TextNormalization, self).__init__()
        import json
        self.stop_words = json.load(open(stop_words)) 
    
    def remove_special_chars(self, text):
        """
        level 1
        Args:
            text: str
        """
        import re
        text = re.sub("(?s)<ref>.+?</ref>", "", text) # remove reference links
        text = re.sub("(?s)<[^>]+>", "", text) # remove html tags
        text = re.sub("&[a-z]+;", "", text) # remove html entities
        text = re.sub("(?s){{.+?}}", "", text) # remove markup tags
        text = re.sub("(?s){.+?}", "", text) # remove markup tags
        #text = re.sub("(?s)\[\[([^]]+\|)", "", text) # remove link target strings
        #text = re.sub("(?s)\[\[([^]]+\:.+?]])", "", text) # remove media links
        text = re.sub('<(S*?)[^>]*>.*?|<.*? />','',text)
        text = re.sub("[']{2,5}", "", text) # remove italic+bold symbols
        text = re.sub('^s*|s*$','', text)
        #text = re.sub('[~！@#￥%……&*（）——+-=？·：；\” \' \【\】 {}，：]','',text)
        text = re.sub('[~！@#￥……&*（）——+={}【】、|；：“‘’”《》，。？/]','', text)
        text = re.sub('[!@#$^&*()_+=\{\}\[\];\'\'\"\",.\<\>?]','', text)
        text = re.sub('[%]','', text)
        text = re.sub('[a-zA-z]+://[^s]*', '', text) #match url
        #text = regex.sub(u"[^\r\n\p{Han}。！？]", "", text)
        #text = re.sub('[\r\n]','',text)
        text = re.sub("[ ]{2,}", " ", text) # Squeeze spaces.
        text = re.sub('\u3000','', text)
        text = re.sub('\r','', text)
    
        text = ' '.join(text.split()) #TBD
        return text
    
    def remove_stopwords(self, text):
        """
        level 1
        Args: 
            text: str
        """
        new_text = []
        for word in text:
            if word not in self.stop_words:
                new_text.append(word)
        return ' '.join(new_text)
    
    def clean_text(self, text):
        """
        level 1
        Args:
            text: str
        """
        import jieba
        from copy import deepcopy
        
        text_ = deepcopy(text)
        text_ = self.remove_special_chars(text_)
        text_ = jieba.cut(text_)
        return self.remove_stopwords(text_)
    
    def segment_text(self, text, pattern_str="['！','？','。','；']"):
        """
        segment long doc/article into multiple sentences
        level 1 -> level 2
        Args:
            doc: str
        return:
            list of str, level 2
        """
        import re
        from copy import deepcopy
        
        text_ = deepcopy(text)
        pattern = re.compile(pattern_str)
        return [i for i in filter(None, re.split(pattern, text_))]
    
    def clean_sentences(self, sents):
        """
        level 2, multiple sentences or a single dialogue that contains multiple sentences
        Args:
            sents: list of str
        """
        from copy import deepcopy
        sents_ = deepcopy(sents)
        for i, sent in enumerate(sents_):
            sents_[i] = self.clean_text(sent)
        return sents_
    
    def segment_long_doc(self, doc, pattern_str="['！','？','。','；']"):
        """
        segment long doc/article into multiple sentences
        level 2 -> level 3
        Args:
            doc: list of str
        return: list of list of str (level 3)
        """
        import re
        from copy import deepcopy
        
        doc_ = deepcopy(doc)
        pattern = re.compile(pattern_str)
        for i, sent in enumerate(doc_):
            doc_[i] = self.segment_text(sent, pattern_str=pattern_str)
            # doc_[i] becomes list of str, level 2
        return doc_
        
    def clean_and_segment_sentences(self, sents, pattern_str="['！','？','。','；']"):
        """
        segment sents into small sents (level 2 -> level 3),
        clean each small sent
        level 2 -> level 3
        Args:
            sents: list of str (level 2)
        return:
            list of list of str (level 3)
        """
        import re
        from copy import deepcopy
        
        sents_ = deepcopy(sents)
        pattern = re.compile(pattern_str)
        for i, sent in enumerate(sents_):
            sents_i = self.segment_text(sent, pattern_str=pattern_str)
            # doc[i] becomes list of str, level 2
            sents_[i] = self.clean_sentences(sents_i)
        return sents_
                
    def clean_docs(self, docs):
        """
        Args:
            docs: list of list of str (level 3)
        return:
            list of list of str (level 3)
        """
        from copy import deepcopy
        
        docs_ = deepcopy(docs)
        for i, sents in enumerate(docs_):
            docs_[i] = self.clean_sentences(sents)
        return docs_
    
    def fit(self, doc, long_doc=False, pattern_str="['！','？','。','；','\n','\t']", keepdim=True):
        """
        segment doc(optional), remove special chars and stopwords
        scenario:
            str: single sentence, or use apply function in single dataframe row
            list of str: multiple sentences, whole text in multiple rows of dataframe, or single dialogue in one row
            list of list of str: dialogues, whole dialogues in multiple rows of dataframe
        Args: 
            doc: str, list of str, list of list of str
        return:
            output: the same format and shape as input, but the bottom level str is seperated by blank space
        """
        level = self.check_doc_level(doc)
        self.print_doc_level(level)
        
        if level==0:
            return []
        elif level==1:
            if long_doc:
                return self.clean_sentences(self.segment_text(doc, pattern_str=pattern_str))
            else:
                return self.clean_text(doc)
        elif level==2:
            if long_doc:
                return self.clean_and_segment_sentences(doc, pattern_str=pattern_str)
            else:
                return self.clean_sentences(doc)
        elif level==3:
            if long_doc:
                return [] # not supported so far
            else:
                return self.clean_docs(doc)
            
    def transform(self, doc, long_doc=False, pattern_str="['！','？','。','；']", keepdim=True):
        return self.fit(doc, long_doc=long_doc, pattern_str=pattern_str, keepdim=keepdim)
    
    def fit_transform(self, doc, long_doc=False, pattern_str="['！','？','。','；']", keepdim=True):
        return self.fit(doc, long_doc=long_doc, pattern_str=pattern_str, keepdim=keepdim)