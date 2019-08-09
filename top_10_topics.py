import gensim
from tqdm import tqdm

class TopTopics(object):
	def __init__(self, num_topics, top_n):
		"""
		Initialize our LDA model, gensim Corpus object, and top_n variable.
		"""
		self.model = gensim.models.LdaModel.load('./data/gensim_models/lda_ubuntu.model')
		self.ubuntu_corpus = gensim.corpora.MmCorpus('./data/gensim_models/ubuntu_bow.mm')
		self.num_topics = num_topics
		self.top_n = top_n

	def extract_topic_representations(self):
		"""
		Extract Top Topics from LDA model.
		"""
		top_topics = self.model.top_topics(self.ubuntu_corpus, topn=self.top_n)[:self.num_topics]
		for i, t in tqdm(enumerate(top_topics)):
		    topic_repr, _ = t
		    top_words = [t[1] for t in topic_repr]
		    print
		    print "Topic: {}".format(i+1)
		    print "Topic Representations: {}".format(" | ".join(top_words))
		    print

if __name__ == '__main__':
	ubuntu_top_topics = TopTopics(num_topics=10, top_n=10)
	ubuntu_top_topics.extract_topic_representations()