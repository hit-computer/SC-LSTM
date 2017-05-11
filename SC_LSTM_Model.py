#coding:utf-8
import tensorflow as tf
import numpy as np
try:
    from tensorflow.python.ops.rnn_cell import BasicLSTMCell
    from tensorflow.python.ops.rnn_cell import DropoutWrapper
    from tensorflow.python.ops.rnn_cell import _linear
    from tensorflow.python.ops.rnn_cell import MultiRNNCell
except:
    from tensorflow.contrib.rnn.python.ops.core_rnn_cell import BasicLSTMCell
    from tensorflow.contrib.rnn.python.ops.core_rnn_cell import DropoutWrapper
    from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import _linear
    from tensorflow.contrib.rnn.python.ops.core_rnn_cell import MultiRNNCell
    
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops import nn_ops

class SC_LSTM(BasicLSTMCell):
    def __init__(self, kwd_voc_size, *args, **kwargs):
        BasicLSTMCell.__init__(self, *args, **kwargs)
        self.key_words_voc_size = kwd_voc_size
    def __call__(self, inputs, state, d_act, scope=None):
        """Long short-term memory cell (LSTM)."""
        with vs.variable_scope(scope or type(self).__name__):  # "BasicLSTMCell"
        # Parameters of gates are concatenated into one multiply for efficiency.
            if self._state_is_tuple:
                c, h = state
            else:
                try:
                    c, h = array_ops.split(1, 2, state)
                except:
                    c, h = array_ops.split(state, 2, 1)
            concat = _linear([inputs, h], 4 * self._num_units, True)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            try:
                i, j, f, o = array_ops.split(1, 4, concat)
            except:
                i, j, f, o = array_ops.split(concat, 4, 1)
            
            w_d = vs.get_variable('w_d', [self.key_words_voc_size, self._num_units])
            
            new_c = (c * sigmoid(f + self._forget_bias) + sigmoid(i) *
                    self._activation(j)) + tf.tanh(tf.matmul(d_act, w_d))
            new_h = self._activation(new_c) * sigmoid(o)

            if self._state_is_tuple:
                new_state = LSTMStateTuple(new_c, new_h)
            else:
                try:
                    new_state = array_ops.concat(1, [new_c, new_h])
                except:
                    new_state = array_ops.concat([new_c, new_h], 1)
            return new_h, new_state

class SC_MultiRNNCell(MultiRNNCell):
    def __call__(self, inputs, state, d_act, scope=None):
        """Run this multi-layer cell on inputs, starting from state."""
        with vs.variable_scope(scope or type(self).__name__):  # "MultiRNNCell"
            cur_state_pos = 0
            cur_inp = inputs
            new_states = []
            outputls = []
            for i, cell in enumerate(self._cells):
                with vs.variable_scope("Cell%d" % i):
                    if self._state_is_tuple:
                        if not nest.is_sequence(state):
                            raise ValueError(
                                "Expected state to be a tuple of length %d, but received: %s"
                                % (len(self.state_size), state))
                        cur_state = state[i]
                    else:
                        cur_state = array_ops.slice(
                            state, [0, cur_state_pos], [-1, cell.state_size])
                        cur_state_pos += cell.state_size
                    cur_inp, new_state = cell(cur_inp, cur_state, d_act)
                    new_states.append(new_state)
                    outputls.append(cur_inp)
        try:
            new_states = (tuple(new_states) if self._state_is_tuple
                      else array_ops.concat(1, new_states))
            outputs = array_ops.concat(1, outputls)
        except:
            new_states = (tuple(new_states) if self._state_is_tuple
                      else array_ops.concat(new_states, 1))
            outputs = array_ops.concat(outputls, 1)
        return cur_inp, new_states, outputs

class SC_DropoutWrapper(DropoutWrapper):
    def __call__(self, inputs, state, d_act, scope=None):
        """Run the cell with the declared dropouts."""
        if (not isinstance(self._input_keep_prob, float) or
                self._input_keep_prob < 1):
            inputs = nn_ops.dropout(inputs, self._input_keep_prob, seed=self._seed)
        output, new_state = self._cell(inputs, state, d_act, scope)
        if (not isinstance(self._output_keep_prob, float) or
                self._output_keep_prob < 1):
            output = nn_ops.dropout(output, self._output_keep_prob, seed=self._seed)
        return output, new_state
