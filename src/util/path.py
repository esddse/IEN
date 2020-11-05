
import os
import sys


# ========== basic setting ============
path_this_file = os.path.abspath(os.path.dirname(__file__))
path_proj_home = os.path.join(path_this_file, "..", "..")

# ============  model ==========
path_model_dir = os.path.join(path_proj_home, "model")

# ============ data ============
path_data_dir = os.path.join(path_proj_home, "data")
path_propara_dir = os.path.join(path_data_dir, "propara")
path_propara_train_dir = os.path.join(path_propara_dir, "data", "train")
path_propara_dev_dir   = os.path.join(path_propara_dir, "data", "dev")
path_propara_test_dir  = os.path.join(path_propara_dir, "data", "test")
path_propara_evaluator_dir = os.path.join(path_propara_dir, "evaluator")

# ========== embedding ===========
path_fasttext_embeddings = os.path.join(path_data_dir, "embedding", "fasttext", "wiki-news-300d-1M.vec")
path_fasttext_vocab = os.path.join(path_data_dir, "embedding", "fasttext", "wiki-news-300d-1M.vocab")

path_allennlp_elmo_weights = os.path.join(path_data_dir, "embedding", "elmo", "elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5")
path_allennlp_elmo_options = os.path.join(path_data_dir, "embedding", "elmo", "elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json")



