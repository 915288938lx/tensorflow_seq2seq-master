import config
import numpy as np
import tensorflow as tf
from config import transformer_config

class Transformer(object):
    def __init__(self,
                 embedding_size=transformer_config.embedding_size,
                 num_layers=transformer_config.num_layers,
                 keep_prob=transformer_config.keep_prob,
                 learning_rate=transformer_config.learning_rate,
                 learning_decay_steps=transformer_config.learning_decay_steps,
                 learning_decay_rate=transformer_config.learning_decay_rate,
                 clip_gradient=transformer_config.clip_gradient,
                 is_embedding_scale=transformer_config.is_embedding_scale,
                 multihead_num=transformer_config.multihead_num,
                 label_smoothing=transformer_config.label_smoothing,
                 max_gradient_norm=transformer_config.clip_gradient,
                 encoder_vocabs=config.encoder_vocabs + 2,
                 decoder_vocabs=config.decoder_vocabs + 2,
                 max_encoder_len=config.max_encoder_len,
                 max_decoder_len=config.max_decoder_len,
                 share_embedding=config.share_embedding,
                 pad_index=None
                 ):
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.keep_prob = keep_prob
        self.learning_rate = learning_rate
        self.learning_decay_steps = learning_decay_steps
        self.learning_decay_rate = learning_decay_rate
        self.clip_gradient = clip_gradient
        self.encoder_vocabs = encoder_vocabs
        self.decoder_vocabs = decoder_vocabs
        self.max_encoder_len = max_encoder_len
        self.max_decoder_len = max_decoder_len
        self.share_embedding = share_embedding
        self.is_embedding_scale = is_embedding_scale
        self.multihead_num = multihead_num
        self.label_smoothing = label_smoothing
        self.max_gradient_norm = max_gradient_norm
        self.pad_index = pad_index
        self.build_model()

    def build_model(self):
        # 初始化变量
        self.encoder_inputs = tf.placeholder(tf.int32, [None, None], name='encoder_inputs')
        self.encoder_inputs_length = tf.placeholder(tf.int32, [None], name='encoder_inputs_length')
        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.decoder_inputs = tf.placeholder(tf.int32, [None, None], name='decoder_inputs')
        self.decoder_targets = tf.placeholder(tf.int32, [None, None], name='decoder_targets')
        self.decoder_inputs_length = tf.shape(self.decoder_inputs)[1]
        self.decoder_targets_length = tf.placeholder(tf.int32, [None], name='decoder_targets_length')
        self.targets_mask = tf.sequence_mask(self.decoder_targets_length, self.max_decoder_len,
                                            dtype=tf.float32, name='masks')
        self.itf_weight = tf.placeholder(tf.float32, [None, None], name='itf_weight')

        # embedding层
        with tf.name_scope("embedding"):
            # encoder_embedding = tf.get_variable(
            #     'encoder_embedding', [self.encoder_vocabs, self.embedding_size],
            #     initializer=tf.random_normal_initializer(0., self.embedding_size ** -0.5)
            # )
            zero = tf.zeros([1, self.embedding_size], dtype=tf.float32)  # for padding
            # embedding_table = tf.Variable(tf.random_uniform([self.voca_size-1, self.embedding_size], -1, 1))
            encoder_embedding = tf.get_variable(
                # https://github.com/tensorflow/models/blob/master/official/transformer/model/embedding_layer.py
                'embedding_table',
                [self.encoder_vocabs - 1, self.embedding_size],
                initializer=tf.random_normal_initializer(0., self.embedding_size ** -0.5))
            front, end = tf.split(encoder_embedding, [self.pad_index, self.encoder_vocabs - 1 - self.pad_index])
            encoder_embedding = tf.concat((front, zero, end), axis=0)  # [self.voca_size, self.embedding_size]
            encoder_position_encoding = self.positional_encoding(self.max_encoder_len)
            if not self.share_embedding:
                decoder_embedding = tf.get_variable(
                    'decoder_embedding', [self.decoder_vocabs, self.embedding_size],
                    initializer=tf.random_normal_initializer(0., self.embedding_size ** -0.5)
                )
                decoder_position_encoding= self.positional_encoding(self.max_decoder_len)

        # encoder
        with tf.name_scope("encoder"):
            encoder_inputs_embedding, encoder_inputs_mask = self.add_embedding(
                encoder_embedding, encoder_position_encoding, self.encoder_inputs,tf.shape(self.encoder_inputs)[1]
            )
            self.encoder_outputs = self.encoder(encoder_inputs_embedding, encoder_inputs_mask)

        # decoder
        with tf.name_scope('decoder'):
            if self.share_embedding:
                decoder_inputs_embedding, decoder_inputs_mask = self.add_embedding(
                    encoder_embedding, encoder_position_encoding, self.decoder_inputs,self.decoder_inputs_length
                )
            else:
                decoder_inputs_embedding, decoder_inputs_mask = self.add_embedding(
                    decoder_embedding, decoder_position_encoding, self.decoder_inputs,self.decoder_inputs_length
                )
            self.decoder_outputs, self.predict_ids= self.decoder(decoder_inputs_embedding, self.encoder_outputs,
                                                                 decoder_inputs_mask,encoder_inputs_mask)

        # loss
        with tf.name_scope('loss'):
            # label smoothing
            self.targets_one_hot = tf.one_hot(
                self.decoder_targets,
                depth=self.decoder_vocabs,
                on_value=(1.0 - self.label_smoothing) + (self.label_smoothing / self.decoder_vocabs),
                off_value=(self.label_smoothing / self.decoder_vocabs),
                dtype=tf.float32
            )

            loss = tf.nn.softmax_cross_entropy_with_logits(
                labels=self.targets_one_hot,
                logits=self.decoder_outputs
            )
            if config.use_itf_loss:
                loss *= self.itf_weight
            else:
                loss *= self.targets_mask
            self.loss = tf.reduce_sum(loss) / tf.reduce_sum(self.targets_mask)

        # 优化函数，对学习率采用指数递减的形式
        self.global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,
                                                   self.learning_decay_steps, self.learning_decay_rate,
                                                   staircase=True)

        optimizer = tf.train.AdamOptimizer(learning_rate)
        trainable_params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, trainable_params)
        clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
        self.train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params))

        # summary
        tf.summary.scalar('loss', self.loss)
        self.merged = tf.summary.merge_all()

    def encoder(self, encoder_inputs_embedding, encoder_inputs_mask):
        # multi-head attention mask
        encoder_self_attention_mask = tf.tile(
            tf.matmul(encoder_inputs_mask, tf.transpose(encoder_inputs_mask, [0, 2, 1])),
            [self.multihead_num, 1, 1]
        )
        encoder_outputs = encoder_inputs_embedding
        for i in range(self.num_layers):
            # multi-head selt-attention sub_layer
            multi_head_outputs = self.multi_head_attention_layer(
                query=encoder_outputs,
                key_value=encoder_outputs,
                score_mask=encoder_self_attention_mask,
                output_mask=encoder_inputs_mask,
                activation=None,
                name='encoder_multi_' + str(i)
            )

            # point-wise feed forward sub_layer
            encoder_outputs = self.feed_forward_layer(
                multi_head_outputs,
                output_mask=encoder_inputs_mask,
                activation=tf.nn.relu,
                name='encoder_dense_' + str(i)
            )
        return encoder_outputs

    def decoder(self, decoder_inputs_embedding, encoder_outputs, decoder_inputs_mask,encoder_inputs_mask):
        # mask
        decoder_encoder_attention_mask = tf.tile(
            tf.transpose(encoder_inputs_mask,[0, 2, 1]),
            [self.multihead_num, 1, 1]
        )

        decoder_self_attention_mask = tf.tile(tf.expand_dims(tf.sequence_mask(
            tf.range(start=1, limit=self.decoder_inputs_length + 1),
            maxlen=self.decoder_inputs_length,
            dtype=tf.float32),axis=0
        ),[self.multihead_num*tf.shape(decoder_inputs_embedding)[0],1,1])

        decoder_outputs = decoder_inputs_embedding
        for i in range(self.num_layers):
            # masked multi-head selt-attention sub_layer
            masked_multi_head_outputs = self.multi_head_attention_layer(
                query=decoder_outputs,
                key_value=decoder_outputs,
                score_mask=decoder_self_attention_mask,
                output_mask=decoder_inputs_mask,
                activation=None,
                name='decoder_first_multi_' + str(i)
            )

            # multi-head selt-attention sub_layer
            multi_head_outputs = self.multi_head_attention_layer(
                query=masked_multi_head_outputs,
                key_value=encoder_outputs,
                score_mask=decoder_encoder_attention_mask,
                output_mask=decoder_inputs_mask,
                activation=None,
                name='decoder_second_multi_' + str(i)
            )

            # point-wise feed forward sub_layer
            decoder_outputs = self.feed_forward_layer(
                multi_head_outputs,
                output_mask=decoder_inputs_mask,
                activation=tf.nn.relu,
                name='decoder_dense_' + str(i)
            )

        # output_layer
        decoder_outputs = tf.layers.dense(decoder_outputs,units=self.decoder_vocabs,activation=None,name='outputs')
        predict_ids = tf.argmax(decoder_outputs,axis=-1,output_type=tf.int32)
        return decoder_outputs, predict_ids

    def multi_head_attention_layer(self, query, key_value, score_mask=None, output_mask=None,
                                   activation=None,name=None):
        """
        multi-head self-attention sub_layer
        :param query:
        :param key_value:
        :param score_mask:
        :param output_mask:
        :param activation:
        :param name:
        :return:
        """
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            # 计算Q、K、V
            V = tf.layers.dense(key_value,units=self.embedding_size,activation=activation,use_bias=False,name='V')
            K = tf.layers.dense(key_value,units=self.embedding_size,activation=activation,use_bias=False,name='K')
            Q = tf.layers.dense(query,units=self.embedding_size,activation=activation,use_bias=False,name='Q')

            # 将Q、K、V分离为multi-heads的形式
            V = tf.concat(tf.split(V, self.multihead_num, axis=-1),axis=0)
            K = tf.concat(tf.split(K, self.multihead_num, axis=-1),axis=0)
            Q = tf.concat(tf.split(Q, self.multihead_num, axis=-1),axis=0)

            # 计算Q、K的点积，并进行scale
            score = tf.matmul(Q, tf.transpose(K, [0, 2, 1])) / tf.sqrt(self.embedding_size / self.multihead_num)

            # mask
            if score_mask is not None:
                score *= score_mask
                score += ((score_mask - 1) * 1e+9)

            # softmax
            softmax = tf.nn.softmax(score,dim=2)

            # dropout
            softmax = tf.nn.dropout(softmax, keep_prob=self.keep_prob)

            # attention
            attention = tf.matmul(softmax,V)

            # 将multi-head的输出进行拼接
            concat = tf.concat(tf.split(attention, self.multihead_num, axis=0),axis=-1)

            # Linear
            Multihead = tf.layers.dense(concat,units=self.embedding_size,activation=activation,
                                        use_bias=False,name='linear')

            # output mask
            if output_mask is not None:
                Multihead *= output_mask

            # 残差连接前的dropout
            Multihead = tf.nn.dropout(Multihead, keep_prob=self.keep_prob)

            # 残差连接
            Multihead += query

            # Layer Norm
            Multihead = tf.contrib.layers.layer_norm(Multihead, begin_norm_axis=2)
            return Multihead

    def feed_forward_layer(self, inputs, output_mask=None, activation=None, name=None):
        """
        point-wise feed_forward sub_layer
        :param inputs:
        :param output_mask:
        :param activation:
        :param name:
        :return:
        """
        with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
            # dense layer
            inner_layer = tf.layers.dense(inputs,units=4 * self.embedding_size,activation=activation)
            dense = tf.layers.dense(inner_layer,units=self.embedding_size,activation=None)

            # output mask
            if output_mask is not None:
                dense *= output_mask

            # dropout
            dense = tf.nn.dropout(dense, keep_prob=self.keep_prob)

            # 残差连接
            dense += inputs

            # Layer Norm
            dense = tf.contrib.layers.layer_norm(dense, begin_norm_axis=2)
        return dense


    def add_embedding(self, embedding,position_encoding,inputs_data,data_length):
        # 将词汇embedding与位置embedding进行相加
        inputs_embedded = tf.nn.embedding_lookup(embedding,inputs_data)
        if self.is_embedding_scale is True:
            inputs_embedded *= self.embedding_size ** 0.5
        inputs_embedded += position_encoding[:data_length, :]

        # embedding_mask
        embedding_mask = tf.expand_dims(
            tf.cast(tf.not_equal(inputs_data, self.pad_index), dtype=tf.float32),
            axis=-1
        )
        inputs_embedded *= embedding_mask

        # embedding dropout
        inputs_embedded = tf.nn.dropout(inputs_embedded, keep_prob=self.keep_prob)
        return inputs_embedded,embedding_mask

    def positional_encoding(self,sequence_length):
        """
        positional encoding
        :return:
        """
        position_embedding = np.zeros([sequence_length, self.embedding_size])
        for pos in range(sequence_length):
            for i in range(self.embedding_size // 2):
                position_embedding[pos, 2 * i] = np.sin(pos / np.power(10000, 2 * i / self.embedding_size))
                position_embedding[pos, 2 * i + 1] = np.cos(pos / np.power(10000, 2 * i / self.embedding_size))
        position_embedding = tf.convert_to_tensor(position_embedding, dtype=tf.float32)
        return position_embedding

    def train(self, sess, encoder_inputs, encoder_inputs_length, decoder_inputs,
              decoder_targets, decoder_targets_length, itf_weight,
              keep_prob=transformer_config.keep_prob):
        feed_dict = {self.encoder_inputs: encoder_inputs,
                     self.encoder_inputs_length: encoder_inputs_length,
                     self.decoder_inputs: decoder_inputs,
                     self.decoder_targets: decoder_targets,
                     self.decoder_targets_length: decoder_targets_length,
                     self.keep_prob: keep_prob,
                     self.batch_size: len(encoder_inputs),
                     self.itf_weight: itf_weight}
        _, train_loss = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        return train_loss

    def eval(self, sess, encoder_inputs_val, encoder_inputs_length_val, decoder_inputs_val,
             decoder_targets_val, decoder_targets_length_val, itf_weight_val):
        feed_dict = {self.encoder_inputs: encoder_inputs_val,
                     self.encoder_inputs_length: encoder_inputs_length_val,
                     self.decoder_inputs: decoder_inputs_val,
                     self.decoder_targets: decoder_targets_val,
                     self.decoder_targets_length: decoder_targets_length_val,
                     self.keep_prob: 1.0,
                     self.batch_size: len(encoder_inputs_val),
                     self.itf_weight: itf_weight_val}
        val_loss = sess.run(self.loss, feed_dict=feed_dict)
        summary = sess.run(self.merged, feed_dict=feed_dict)
        return val_loss, summary