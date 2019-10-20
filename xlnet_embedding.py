# -*- coding: UTF-8 -*-

import os
import codecs
import numpy as np
from keras.models import Input, Model
from keras_bert.layers import Extract
from keras_xlnet import load_trained_model_from_checkpoint
from keras.layers import Add, Embedding, Average, Maximum, Concatenate, Lambda
from keras_xlnet import Tokenizer, ATTENTION_TYPE_BI, ATTENTION_TYPE_UNI

class XlnetEmbedding(object):
    def __init__(self, hyper_parameters):
        self.layer_indexes = hyper_parameters.get('layer_indexes', [-1])
        self.batch_size = hyper_parameters.get('batch_size', 16)
        self.len_max = hyper_parameters.get('len_max', 50)
        self.trainable = hyper_parameters.get('trainable', False)
        self.target_len = self.len_max
        self.corpus_path = hyper_parameters['model_path']

        self.checkpoint_path = os.path.join(self.corpus_path, 'xlnet_model.ckpt')
        self.config_path = os.path.join(self.corpus_path, 'xlnet_config.json')
        self.spiece_model = os.path.join(self.corpus_path, 'spiece.model')

        self.attention_type = hyper_parameters.get('attention_type', 'bi').lower()  # or 'uni'
        self.attention_type = ATTENTION_TYPE_BI if self.attention_type == 'bi' else ATTENTION_TYPE_UNI
        self.memory_len =  hyper_parameters.get('memory_len', 0)
        self.merge_type = hyper_parameters.get('merge_type', "add").lower()

        self.build()

    def build(self):
        print('load XLNet model start!')
        print([self.target_len, self.memory_len, self.attention_type, self.batch_size])
        # 模型加载
        model = load_trained_model_from_checkpoint(checkpoint_path=self.checkpoint_path,
                                                   attention_type=self.attention_type,
                                                   in_train_phase=self.trainable,
                                                   config_path=self.config_path,
                                                   memory_len=self.memory_len,
                                                   target_len=self.target_len,
                                                   batch_size=self.batch_size,
                                                   mask_index=0)
        # 字典加载
        self.tokenizer = Tokenizer(self.spiece_model)
        self.model_layers = model.layers
        """
        # debug时候查看layers
        for i in range(len(model.layers)):
            print([i, model.layers[i]])
        base版trainable： 129层  9 + 120     trainable=False:  126  6+120
        0-8：输入 + embedding
        9-128： 每10个layer一层

        mid版trainable=True： 249层   9 + 240     trainable=False:  246  6+240
        0-8：输入 + embedding
        9-248： 每10个layer一层
        """
        len_layers = self.model_layers.__len__()
        len_couche = len_layers//10

        layer_0 = len_layers - len_couche*10
        layer_dict = [layer_0 - 1]
        if self.trainable == False:
        	layer_dict[0] += 1
        	sub_diff = 1
        else:
        	sub_diff = 2
        for i in range(len_couche):
            layer_0 += 10
            layer_dict.append(layer_0 - sub_diff)

        if len(self.layer_indexes) == 0:
            encoder_layer = model.output
        elif len(self.layer_indexes) == 1:
            if abs(self.layer_indexes[0]) in [i for i in range(len_couche + 1)]:
                encoder_layer = model.get_layer(index=layer_dict[self.layer_indexes[0]]).get_output_at(-1)
            else:
                encoder_layer = model.get_layer(index=layer_dict[-1]).get_output_at(-1)
        else:
            all_layers = [model.get_layer(index=layer_dict[lay]).get_output_at(-1)
                          if abs(lay) in [i for i in range(len_couche + 1)]
                          else model.get_layer(index=layer_dict[-1]).get_output_at(-1) 
                          for lay in self.layer_indexes]
            all_layers_select = []
            for all_layers_one in all_layers:
                all_layers_select.append(all_layers_one)
            
            # custom
            if self.merge_type == "add":
                encoder_layer = Add()(all_layers_select)
            elif self.merge_type == "avg":
                encoder_layer = Average()(all_layers_select)
            elif self.merge_type == "max":
                encoder_layer = Maximum()(all_layers_select)
            elif self.merge_type == "concat":
                encoder_layer = Concatenate()(all_layers_select)
            else:
                raise RuntimeError("invalid merge type")
            print(encoder_layer)

        self.output = Lambda(lambda x: x, output_shape=lambda s:s)(encoder_layer)
        self.input = model.inputs
        self.model = Model(model.inputs, self.output)

        self.embedding_size = self.model.output_shape[-1]
        self.vocab_size = len(self.tokenizer.sp)
        print("load Keras XLNet Embedding finish")
        #model.summary()

    def sentence2idx(self, text):
        tokens = self.tokenizer.encode(text)
        tokens = tokens + [0] * (self.target_len - len(tokens)) \
                               if len(tokens) < self.target_len \
                               else tokens[0:self.target_len]
        token_input = np.expand_dims(np.array(tokens), axis=0)
        segment_input = np.zeros_like(token_input)
        memory_length_input = np.zeros((1, 1))
        return [token_input, segment_input, memory_length_input]

    def idx2sentence(self, idx):
        text = self.tokenizer.decode(idx)
        return 



def get_embbeding(hyper):
    word_embedding = XlnetEmbedding(hyper_parameters=hyper)
    if os.path.exists(hyper["path_fineture"]) and hyper["trainable"]:
        word_embedding.model.load_weights(hyper["path_fineture"])
    return word_embedding
