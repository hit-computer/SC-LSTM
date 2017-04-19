# SC-LSTM
We focus on text generation, witch is a challenge task that generates a long text under the theme of multiple words. In detail, given a set W = {w1, w2, ..., wk}, this generator aims at generates a text under the semantic information of those words. SC-LSTM ([Wen et al., 2015](http://www.emnlp2015.org/proceedings/EMNLP/pdf/EMNLP199.pdf)) is the best paper of EMNLP 2015, which is is a statistical language generator based on a semantically controlled Long Short-term Memory structure for response generation. The author incorporates a dialogue act 1-hot vector into the original LSTM model and enables the generator to output the word-related text. We directly use this model for our task. And we input a set of words represented by 1-hot vector instead of dialogue act vector in our task.

The code in this repository is written in Python 2.7/TensorFlow 0.12. And if you use other versions of Python or TensorFlow, you should modify some code. Since SC-LSTM is based on original LSTM, we modify some code based on BasicLSTMCell class of TensorFlow to develop SC-LSTM model (detail in [SC_LSTM_Model.py](https://github.com/hit-computer/SC-LSTM/blob/master/SC_LSTM_Model.py)). 

We need text-word_set pairs to train SC-LSTM model, but to the best of our knowledge, there is no public large-scale dataset. Therefore, we can only use the public small-scale data to test this model. We have found a news article dataset annotated using AMT(More details about the corpus can be found in [the paper](http://www.cs.cmu.edu/~lmarujo/publications/lmarujo_LREC_2012.pdf))

## Usage

### Data

In `DATA/` respository, there are two files `TrainingData_keywords.txt` and `TrainingData_Text.txt`, which is created from news article dataset mentioned above. `TrainingData_Text.txt` file contains just title, and each line is a title which is regarded as one text(data). Correspondingly, `TrainingData_keywords.txt` file contains word set, and each line is a set of word for text. Then, we use this text-words pair data to train SC-LSTM model.