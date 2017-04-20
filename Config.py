#coding:utf-8

class Config(object):
    data_dir = 'Data/'
    vec_file = 'Data/vec10.txt'
    init_scale = 0.04
    learning_rate = 0.001
    max_grad_norm = 10
    num_layers = 2
    num_steps = 25 #this value is one more than max number of words in sentence
    hidden_size = 20
    word_embedding_size = 10
    max_epoch = 30
    max_max_epoch = 50
    keep_prob = 0.5
    lr_decay = 0.95
    batch_size = 16
    vocab_size = 543
    keyword_min_count = 2
    save_freq = 10 #The step (counted by the number of iterations) at which the model is saved to hard disk.
    model_path = 'Model_News' #the path of model that need to save or load