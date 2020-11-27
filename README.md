### EATN 

Dataset and source code for **"EATN: An Efﬁcient Adaptive Transfer Network for Aspect-level Sentiment Analysis".**

# 

### Requirements

— Python3.6

— Numpy 1.13.3

— [Google Word2Vec](https://code.google.com/archive/p/word2vec/) 

— Transformer

— sklearn

— other pakages

To install requirements, please run `pip install -r requirements.txt.`



# Environment

— OS: CentOS Linux release 7.4.1708

— CPU: Two 2.20 GHz Intel Xeon E5-2650

— GPU: Four Tesla K80 GPU

— CUDA: 10.2



# Running

**Prepare the Pre-trained model :**

— 1. Get the BERT pre-trained model and generate the embeddings (./word2vec/get_pre_bert.sh) ;

​               — You can get the Word Embeddings through official [BERT](https://github.com/google-research/bert) or [Bert-As-Service](https://bert-as-service.readthedocs.io/en/latest/) ;

​               — [Google Word2Vec](https://code.google.com/archive/p/word2vec/) ;

​               — [GloVe](https://nlp.stanford.edu/projects/glove/) ;

— 2. Put the pre-trained model (Google-Word2Vec/Bert) to the coresponseding path ;



**Run the baseline models :** 

 —  python train_base.py --model_name xxx --dataset xxx

**Run the eatn models :** 

 —  python train_eatn.py


**Contact :**
If you have any problem about this library, please create an Issue or send us an Email at:
— [kkzhang0808@gmail.com](kkzhang0808@gmail.com)
— [kkzhang0808@mail.ustc.edu.cn](sa517494@mail.ustc.edu.cn)

