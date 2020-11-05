
from util.path import *
from util.data import *

class NCETConfig(object):
    def __init__(self):
        # basic
        self.model_name = "NCET"
        self.path_word_vector = path_fasttext_embeddings
        self.use_elmo = False
        self.path_elmo_weights = path_allennlp_elmo_weights
        self.path_elmo_options = path_allennlp_elmo_options
        self.embedding_dim = 100
        self.hidden_dim = 256
        self.embed_dropout_rate = 0.0
        self.dropout_rate = 0.5

        # data
        self.max_word_length = 175
        self.max_sent_length = 10+2
        # gpu
        self.gpu_id = "2"

        # train
        self.batch_size = 1   # batch size fixed to 1
        self.max_epoch_num = 15
        self.learning_rate = 2e-4
        self.min_learning_rate  = 1e-5
        self.patience = 1
        self.decay_factor = 0.2
        self.warm_up_epoch = 3

        # test
        self.test_only = False
        self.model_path = ""
        self.test_batch_size = 1

        # other
        self.use_state_graph = False

        # do some initialization 
        self.init()

        if self.use_elmo:
            self.embedding_dim = 1024

    def init(self):
        ''' '''
        self.vocab_size, self.embedding_dim, self.word2index, self.index2word, self.embeddings = load_word_vector(self.path_word_vector, max_vocab_size=50000)
        self.label_size = 7
        self.label2index = {"<PAD>":0, "<SOS>":1, "<EOS>":2, "NONE":3, "MOVE":4, "CREATE":5, "DESTROY":6}
        self.index2label = {0:"<PAD>", 1:"<SOS>", 2:"<EOS>", 3:"NONE", 4:"MOVE", 5:"CREATE", 6:"DESTROY"}


class IENConfig(object):
    def __init__(self):
        # basic
        self.model_name = "IEN"
        self.path_word_vector = path_fasttext_embeddings
        self.use_elmo = False
        self.path_elmo_weights = path_allennlp_elmo_weights
        self.path_elmo_options = path_allennlp_elmo_options
        self.embedding_dim = 100
        self.hidden_dim = 256
        self.embed_dropout_rate = 0.0
        self.dropout_rate = 0.5
        self.bidirectional = True

        # data
        self.max_word_length = 175
        self.max_sent_length = 10+2
        # gpu
        self.gpu_id = "2"

        # train
        self.batch_size = 1   # batch size fixed to 1
        self.max_epoch_num = 15
        self.learning_rate = 2e-4
        self.min_learning_rate  = 1e-5
        self.patience = 1
        self.decay_factor = 0.2
        self.warm_up_epoch = 3

        # test
        self.test_only = False
        self.model_path = ""
        self.test_batch_size = 1

        # visualization
        self.get_attention_e2e  = False
        self.get_attention_l2e = False

        # do some initialization 
        self.init()

        if self.use_elmo:
            self.embedding_dim = 1024

    def init(self):
        ''' '''
        self.vocab_size, self.embedding_dim, self.word2index, self.index2word, self.embeddings = load_word_vector(self.path_word_vector, max_vocab_size=50000)
        self.label_size = 7
        self.label2index = {"<PAD>":0, "<SOS>":1, "<EOS>":2, "NONE":3, "MOVE":4, "CREATE":5, "DESTROY":6}
        self.index2label = {0:"<PAD>", 1:"<SOS>", 2:"<EOS>", 3:"NONE", 4:"MOVE", 5:"CREATE", 6:"DESTROY"}


