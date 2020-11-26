### EATN 

Dataset and source code for **"EATN: An Efﬁcient Adaptive Transfer Network for Aspect-level Sentiment Analysis".**

# 

### Requirements

— Python3.6

— Numpy 1.13.3

— [Google Word2Vec](https://code.google.com/archive/p/word2vec/) 

— Transformer

— sklearn

To install requirements, please run `pip install -r requirements.txt.`



# Environment

— OS: CentOS Linux release 7.4.1708

— CPU: Two 2.20 GHz Intel Xeon E5-2650

— GPU: Four Tesla K80 GPU

— CUDA: 10.2



# Running

**Prepare the Pre-trained model :**

— 1. Get the BERT pre-trained model and generate the embeddings ;

​			    — You can get the Word Embeddings through official BERT or Bert-As-Service ;

​				— [Google Word2Vec](https://code.google.com/archive/p/word2vec/) ;

— 2. Put the pre-trained model (Google-Word2Vec/Bert) to the coresponseding path ;



**Run the baseline model :** 

 — 1. python train_base.py --model_name xxx --dataset xxx

**Run the EATN model :** 

— 1. python train_model.py 

