from gensim.parsing.preprocessing import strip_punctuation
from gensim.utils import simple_preprocess
import csv
import gensim
import glob
import nltk

class UbuntuTopicDetector(object):
    def __init__(self):
        """
        Initialize our LDA model, gensim Dictionary object, and array for STOPWORDs.
        """
        self.model = gensim.models.LdaModel.load('./data/gensim_models/lda_ubuntu.model')
        self.id2word_ubuntu = gensim.corpora.Dictionary.load('./data/gensim_models/ubuntu.dictionary')
        self.STOPWORDS = nltk.corpus.stopwords.words('english')
        
    def extract_text_from_chat_dialogue(self, chat_file):
        """
        Extract Text from chat .tsv file.
        """
        chat_dialogue = self.read_tsv_file(chat_file)
        return ' '.join(chat_dialogue)
    
    def predict_topics(self, chat_file, top_n=3):
        """
        Predict n most relevant topics for a given
        chat file where n = top_n.
        """
        bow_vector = self.vectorize_document(chat_file)
        lda_vector = self.transform_into_semantic_space(bow_vector)
        print "Chat File: {}".format(chat_file)
        self.print_relevant_topics(lda_vector, num_topics=top_n)

    def print_relevant_topics(self, vector, num_topics):
        """
        Print the n most signifcant topics of a model vector
        where n = num_topics.
        """
        most_prominent_topics = list(sorted(vector, key=lambda t: t[1], reverse=True))[:num_topics]
        for idx, t in enumerate(most_prominent_topics):
            topic_repr = self.model.show_topic(t[0])
            top_words = [t[0] for t in topic_repr]
            print "Topic: {}".format(idx+1)
            print "Topic Representations: {}".format(" | ".join(top_words))
            print
        
    def read_tsv_file(self, chat_file):
        """
        Extract text from each row in the chat log file
        and return a list of dialogue containing the
        entire conversation.
        """
        dialogue = []
        with open(chat_file) as tsv_file:
            reader = csv.reader(tsv_file, delimiter='\t')
            for row in reader:
                dialogue.append(str(row[3]).strip())
        return dialogue
    
    def transform_into_semantic_space(self, vector):
        """
        Transform bag of words vector into LDA semantic
        space.
        """
        lda_vector = self.model[vector]
        return lda_vector

    def tokenize(self, text):
        """
        Remove punctuation and lowercase text, then
        generate tokens of our chat file.
        """
        return [token for token in simple_preprocess(strip_punctuation(text.strip())) if token not in self.STOPWORDS]
    
    def vectorize_document(self, chat_file):
        """
        Vectorize document into a bag of words vector.
        """
        text = self.extract_text_from_chat_dialogue(chat_file)
        bow_vector = self.id2word_ubuntu.doc2bow(self.tokenize(text))
        return bow_vector

if __name__ == '__main__':
    # Load Topic Detector
    topic_detector = UbuntuTopicDetector()
    # Load Test Files
    test_chat_files = glob.glob('data/dialogs/5/*.tsv')
    # Extract the 3 most relvant topics for a file in 'dialogs/5' folder.
    topic_detector.predict_topics(test_chat_files[5])
    # Extract 3 most relevant topics for another chat file.
    topic_detector.predict_topics(test_chat_files[9])