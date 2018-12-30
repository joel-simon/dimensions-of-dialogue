# https://medium.com/@mishra.thedeepak/doc2vec-simple-implementation-example-df2afbbfbad5
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
import nltk
import numpy as np

nltk.download('punkt')
tokenizer = RegexpTokenizer(r'\w+')

# https://www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html
movie_lines_path = "./cornell movie-dialogs corpus/movie_lines.txt"
corpus = []

with open(movie_lines_path, 'r', encoding="utf8", errors='ignore') as movie_lines_file:
    for line in movie_lines_file.readlines():
        line = line.strip()
        split = line.split('+++$+++')
        dialogue = split[-1]
        corpus.append(dialogue)

tagged_data = [ TaggedDocument(words=tokenizer.tokenize(_d.lower()), tags=[ str(i) ]) \
                for i, _d in enumerate(corpus) ]

max_epochs = 100
vec_size = 25
alpha = 0.025

model = Doc2Vec(size=vec_size,
                alpha=alpha,
                min_alpha=0.00025,
                min_count=1,
                dm =1)

model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

model.save("d2v_%i.model"%vec_size)
print("Model Saved")

dataset = np.zeros((len(tagged_data), vec_size), dtype='float32')

for i, tagged_words in enumerate(tagged_data):
    vec = model.infer_vector(tagged_words.words)
    dataset[i] = vec

np.save('doc_to_vec_%i.npy'%vec_size, dataset)