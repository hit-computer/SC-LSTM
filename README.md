# SC-LSTM
Text generation is a interesting task, and we want to generates a long text under the meaning of multiple words. In detail, given a set W = {w1, w2, ..., wk}, this generator aims at generates a text under the semantic information of those words. SC-LSTM ([Wen et al., 2015](http://www.emnlp2015.org/proceedings/EMNLP/pdf/EMNLP199.pdf)) is the best paper of EMNLP 2015, which is is a statistical language generator based on a semantically controlled Long Short-term Memory structure for response generation. The author incorporates a dialogue act 1-hot vector into the original LSTM model and enables the generator to output the word-related text. We directly use this model for our task. And we input a set of words represented by 1-hot vector instead of dialogue act vector in our task.

The code in this repository is written in Python 2.7/TensorFlow 0.12. And if you use other versions of Python or TensorFlow, you should modify some code. Since SC-LSTM is based on original LSTM, we modify some code based on BasicLSTMCell class of TensorFlow to develop SC-LSTM model (detail in [SC_LSTM_Model.py](https://github.com/hit-computer/SC-LSTM/blob/master/SC_LSTM_Model.py)). 

We need text-word_set pairs to train SC-LSTM model, but to the best of our knowledge, there is no public large-scale dataset. Therefore, we can only use the public small-scale data to test this model. We have found a news article dataset annotated using AMT(More details about the corpus can be found in [the paper](http://www.cs.cmu.edu/~lmarujo/publications/lmarujo_LREC_2012.pdf))

## Usage

### Data

In `Data/` respository, there are three files `TrainingData_Keywords.txt` , `TrainingData_Text.txt` and `vec5.txt`(word embedding trained by word2vec), which is created from news article dataset mentioned above. `TrainingData_Text.txt` file contains just title, and each line is a title which is regarded as one text(data). Correspondingly, `TrainingData_keywords.txt` file contains word set, and each line is a set of word for text. Then, we use this text-words pair data to train SC-LSTM model.

### Training

Before train the model, you should set some parameters of this model in `Config.py` file. Then, you need to run `Preprocess.py` file for creating `sclstm_data` file(convert trainingdata into binary formats of TensorFlow, and more detail about this can be found in [the blog](https://indico.io/blog/tensorflow-data-inputs-part1-placeholders-protobufs-queues/)), `word_vec.pkl` file(this is word embedding), `word_vec.pkl` file(vocabulary of text) and `kwd_voc.pkl` file(vocabulary of keywords). 

Start training the model using `train.py`:

```
$ python train.py
```

### Generation

After you train the model, you can generate the text under the control of word set. You should modify `generation.py` file and set `test_word` to a set of words. Then, if you want, you can also set some parameters for generation in `Config.py` file. Generate text by run:

```
$ python generation.py
```

### Result

We randomly choose a set of words from trainingdata, `[u'FDA', u'menu']`. The training data is so small that we can't get a desired result, and some result samples show below:

>Depp Calorie proposes the pregnancy END END PAD  
>carries FDA Have of pleading fracas END PAD

If you have large-scale dataset, I think you could get much better result.
