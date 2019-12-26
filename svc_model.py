from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from text_features import Lowercase, RemoveTone, CountEmoticons
from feature_transform import FeatureTransformer

# estimator_C = 0.375
# lower_tfidf__ngram_range = (1, 3)
# with_tone_char__ngram_range = (1, 5)
# remove_tone__tfidf__ngram_range = (1, 2)

class SVCModel(object):
    def __init__(self):
        self.clf = self._init_pipeline()

    @staticmethod
    def _init_pipeline():
        pipeline = Pipeline([
            ("transformer", FeatureTransformer()),#sử dụng pyvi tiến hành word segmentation
            ("vect", CountVectorizer()),#bag-of-words
            ("tfidf", TfidfTransformer()),#tf-idf
            ("clf-svm", SGDClassifier(loss='log', penalty='l2', alpha=1e-3, random_state=None))#model svm
        ])

        return pipeline