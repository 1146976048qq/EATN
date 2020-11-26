from sklearn.base import BaseEstimator

class TextProcessorBase(BaseEstimator):
    def __init__(self):
        pass
    
    def check_doc_level(self, doc):
        """
        check if the doc is str, list of str or list of list of str
        Args:
            doc: the doc is str, list of str or list of list of str
        """
        from collections import Iterable
        if doc is None or len(doc)==0:
            return 0
        elif isinstance(doc, str) or (not isinstance(doc, Iterable)):
            # single sentence, str
            return 1
        elif isinstance(doc[0], str) or (not isinstance(doc[0], Iterable)): # TO do advanced way to check format
            # multiple sentences, a single dialogue contains multiple sentences, list of str
            return 2
        elif isinstance(doc[0][0], str) or (not isinstance(doc[0][0], Iterable)): # TO do advanced way to check format
            # multiple dialogues / long document/ articles, list of list of str
            return 3
        else:
            raise ValueError('{} is not supported.'.format(type(doc)))
            
    def print_doc_level(self, level):
        print('level ',level)
        if level==0:
            print('Input Doc is empty, nothing to do here')
        elif level==1:
            print('Process string ===> ')
        elif level==2:
            print('Process multiple sentences ===> ')
        elif level==3:
            print('Process dialogues/documents ====> ')
            
    def expand_doc_level(self, doc):
        """
        Expand doc from str to list of str 
        Args:
            doc: str, level 1
        return:
            doc: level 2
        """
        # to do
        return [doc]
    
    def shrink_doc_level(self, doc):
        """
        Shrink list of str to str
        Args:
            doc: list of str, level 2
        return:
            doc: level 1
        """
        # to do
        return doc[0]