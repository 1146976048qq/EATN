class MultiLabel_Vectorizer(object):
    def __init__(self, onehot=True):
        self.unk_label = '<unk>'
        self.unk_id = 0
        self.onehot = onehot
        
    def fit(self, multi_labels):
        """
        Args:
            multi_labels: list of (str, list)
        """
        multi_labels = self.expand_label_dims(multi_labels)
        self.get_unique_labels(multi_labels)
        self.build_label2int_dict(self.unique_labels)
        self.build_int2label_dict(self.label2int_dict)
        
    def transform(self, multi_labels, onehot=None, return_array=True):
        """
        Args:
            multi_labels: list of (str, list)
        """
        if onehot is not None:
            self.onehot = onehot
        if self.onehot:
            return self.encode_label2int_onehot(multi_labels,return_array=return_array)
        else:
            return self.encode_label2int_pos(multi_labels,return_array=return_array)
        
    def inverse_transform(self, multi_ints, onehot=None):
        """
        Args:
            multi_ints: list of (int, list)
        """
        if onehot is not None:
            self.onehot = onehot
        if self.onehot:
            return self.decode_int2label_onehot(multi_ints)
        else:
            return self.decode_int2label_pos(multi_ints)
        
    def get_unique_labels(self, multi_labels):
        self.unique_labels = list(set([t for tt in multi_labels for t in tt]))
        self.n_labels = len(self.unique_labels)
        
    def build_label2int_dict(self, labels):
        self.label2int_dict = dict(zip(labels, range(len(labels))))
        
    def build_int2label_dict(self, label2int_dict):
        self.int2label_dict = {val:key for key, val in label2int_dict.items()}
        
    def expand_label_dims(self, multi_labels):
        from collections import Iterable
        if multi_labels is None or len(multi_labels)==0:
            return None
        if isinstance(multi_labels, str) or not isinstance(multi_labels, Iterable):
            multi_labels = [multi_labels]
        if isinstance(multi_labels[0], str) or not isinstance(multi_labels[0], Iterable):
            multi_labels = [[label] for label in multi_labels]
        return multi_labels
    
    def shrink_label_dims(self, multi_labels, return_array=True):
        from collections import Iterable
        import numpy as np
        if multi_labels is None or len(multi_labels)==0:
            return None
        #print(type(multi_labels), multi_labels)
        multi_labels = np.array(multi_labels)
        #print('after array ',type(multi_labels), multi_labels)
        if len(multi_labels)==1:
            return multi_labels[0] if return_array else multi_labels.tolist()
        elif multi_labels.shape[1]==1:
            multi_labels = np.reshape(multi_labels, (-1))
        return multi_labels if return_array else multi_labels.tolist()
        
    def encode_label2int_pos(self, multi_labels, return_array=True):
        multi_labels = self.expand_label_dims(multi_labels)
        multi_ints = []
        for labels in multi_labels:
            multi_ints.append([self.label2int_dict.get(label,0) for label in labels])
        return self.shrink_label_dims(multi_ints, return_array=return_array)

    def decode_int2label_pos(self, multi_ints):
        multi_ints = self.expand_label_dims(multi_ints)
        multi_labels = []
        for ints in multi_ints:
            multi_labels.append([self.int2label_dict.get(i, self.unk_label) for i in ints])
        return self.shrink_label_dims(multi_labels)
    
    def encode_label2int_onehot(self, multi_labels, return_array=True):
        multi_labels = self.expand_label_dims(multi_labels)
        multi_ints = []
        for labels in multi_labels:
            #multi_ints.append([self.label2int_dict.get(label,0) for label in labels])
            int_pos = [0]*len(self.unique_labels)
            for label in labels:
                int_pos[self.label2int_dict[label]] = 1
            multi_ints.append(int_pos)
        return self.shrink_label_dims(multi_ints,return_array=return_array)
    
    def decode_int2label_onehot(self, multi_ints):
        multi_ints = self.expand_label_dims(multi_ints)
        multi_labels = []
        for ints in multi_ints:
            label_pos = []
            for i, j in enumerate(ints):
                if j>0:
                    label_pos.append(self.int2label_dict[i])
            multi_labels.append(label_pos)
        return self.shrink_label_dims(multi_labels)
    
###===============================
# a = ['a','b','c']
# b = [1,2,3]
# c = [['a','b'],['c','c'],['d','e']]
# d = [[2],[3],[0]]

# vec = MultiLabel_Vectorizer()

# vec.fit(d)
# print(vec.unique_labels, vec.label2int_dict, vec.int2label_dict)
# vec.transform(d)
# vec.inverse_transform(vec.transform(d))

# vec.fit(train_data['gold_label'])
# vec.transform(train_data['gold_label'], onehot=False)

###==============================