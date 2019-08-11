import tensorflow as tf
# import tensorflow.contrib.eager as tfe
# tfe.enable_eager_execution()
class Seq2seq(object):
	
	def build_inputs(self, config):#
		self.seq_inputs = tf.placeholder(shape=(config.batch_size, None), dtype=tf.int32, name='seq_inputs') #(128, 8)             model.seq_inputs: source_batch,
		self.seq_inputs_length = tf.placeholder(shape=(config.batch_size,), dtype=tf.int32, name='seq_inputs_length') # (128,1)     model.seq_inputs_length: source_lens,
		self.seq_targets = tf.placeholder(shape=(config.batch_size, None), dtype=tf.int32, name='seq_targets') #(128,9 )           model.seq_targets: target_batch,
		self.seq_targets_length = tf.placeholder(shape=(config.batch_size,), dtype=tf.int32, name='seq_targets_length') #(128, 1)  model.seq_targets_length: target_lens
#
		
	def build_loss(self, logits):

		loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
			labels=self.seq_targets,
			logits=logits,
		)
		loss = tf.reduce_mean(loss)
		return loss

		
	def build_optim(self, loss, lr):
		return tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
	
	def attn(self, hidden, encoder_outputs): #  (50,)  (128,13,50)
		# hidden: B * D          B:batch,   D:hidden_dim   S: each sequence length, 最大输入8, 最大输出9
		# encoder_outputs: B * S * D
		attn_weights = tf.matmul(encoder_outputs, tf.expand_dims(hidden, 2)) # (128,8,50)张量乘以(128,50,1), 得到(128,8,1), 将hidden 张量(128,50) 在axis= 2 维度增加一个维度 变成(128,50,1)
		# attn_weights: B * S * 1
		context = tf.squeeze(tf.matmul(tf.transpose(encoder_outputs, [0,2,1]), attn_weights)) # (128,50) 将所有维度维1的维都删掉
		# context: B * D batch_size * hidden_dim
		return context # (128,50)
				
	def __init__(self, config, w2i_target, useTeacherForcing=True, useAttention=True, useBeamSearch=1):
	
		self.build_inputs(config)
		
		with tf.variable_scope("encoder"):
		
			encoder_embedding = tf.Variable(tf.random_uniform([config.source_vocab_size, config.embedding_dim]), dtype=tf.float32, name='encoder_embedding') # shape=(13,100) , 要嵌入的空间
			encoder_inputs_embedded = tf.nn.embedding_lookup(encoder_embedding, self.seq_inputs)
			print(encoder_inputs_embedded)# 词嵌入,查表,将(128,8)形状的词嵌入(13,100)的空间, (128,13,100)
			
			((encoder_fw_outputs, encoder_bw_outputs), (encoder_fw_final_state, encoder_bw_final_state)) = tf.nn.bidirectional_dynamic_rnn( # 添加双向LSTM层
				cell_fw=tf.nn.rnn_cell.GRUCell(config.hidden_dim),  #hidden_dim = 50
				cell_bw=tf.nn.rnn_cell.GRUCell(config.hidden_dim), 
				inputs=encoder_inputs_embedded, 
				sequence_length=self.seq_inputs_length, #source_length: [3, 6, 7, 3, 2, 8, 8, 2, 5, 2, 4, 4, 4, 1, 3, 3, 3, 3, 8, 6, 5, 4, 4, 3, 2, 2, 2, 2, 4, 5, 1, 3, 7, 8, 5, 6,......] len=128
				dtype=tf.float32, 
				time_major=False
			)
			encoder_state = tf.add(encoder_fw_final_state, encoder_bw_final_state) # (128,50)
			encoder_outputs = tf.add(encoder_fw_outputs, encoder_bw_outputs) #(128,13,50)
		
		with tf.variable_scope("decoder"):
			
			decoder_embedding = tf.Variable(tf.random_uniform([config.target_vocab_size, config.embedding_dim]), dtype=tf.float32, name='decoder_embedding') #(13,100)
					
			with tf.variable_scope("gru_cell"):
				decoder_cell = tf.nn.rnn_cell.GRUCell(config.hidden_dim)
				decoder_initial_state = encoder_state # (128,50), 用encoder得到的state传入decoder中
			
			# if useTeacherForcing and not useAttention:
				# decoder_inputs = tf.concat([tf.reshape(tokens_go,[-1,1]), self.seq_targets[:,:-1]], 1)
				# decoder_inputs_embedded = tf.nn.embedding_lookup(decoder_embedding, decoder_inputs)
				# decoder_outputs, decoder_state = tf.nn.dynamic_rnn(cell=decoder_cell, inputs=decoder_inputs_embedded, initial_state=decoder_initial_state, sequence_length=self.seq_targets_length, dtype=tf.float32, time_major=False)
			
			tokens_go = tf.ones([config.batch_size], dtype=tf.int32, name='tokens_GO') * w2i_target["_GO"] # (128,)
			tokens_eos = tf.ones([config.batch_size], dtype=tf.int32, name='tokens_EOS') * w2i_target["_EOS"] #(128,)*"2"
			tokens_eos_embedded = tf.nn.embedding_lookup(decoder_embedding, tokens_eos) #(128,100)
			tokens_go_embedded = tf.nn.embedding_lookup(decoder_embedding, tokens_go) #(128,100)
			
			W = tf.Variable(tf.random_uniform([config.hidden_dim, config.target_vocab_size]), dtype=tf.float32, name='decoder_out_W') #(50,13)
			b = tf.Variable(tf.zeros([config.target_vocab_size]), dtype=tf.float32, name="decoder_out_b") #(13,), 后面可以通过广播
			
			def loop_fn(time, previous_output, previous_state, previous_loop_state):
				if previous_state is None:    # time step == 0
					initial_elements_finished = (0 >= self.seq_targets_length)  # all False at the initial step
					initial_state = decoder_initial_state # last time steps cell state 获取上一个时间步的cell_state,即为encoder的cell_state
					initial_input = tokens_go_embedded # last time steps cell input 获取上一个时间步的输入, 这里为"开始符" GO
					if useAttention:
							# (128,150) =  (128,100) concatenate(axis=1)  (128,50)
						initial_input = tf.concat([initial_input, self.attn(initial_state, encoder_outputs)], 1) #(128,150)  如果应用attention机制, 则将初始输入和attention context 进行向量axis=1的拼接, 当做初始输入
					initial_output = None #none
					initial_loop_state = None  # we don't need to pass any additional information
					return (initial_elements_finished, initial_input, initial_state, initial_output, initial_loop_state)
				else: # 从第二个时间步开始,必先获取下一个batch输入
					# 当在第一次调用了loop_fn时, previous_state =None, loop_fn返回finished状态, 初始输入(即"开始符" GO), 初始状态(encoder中的输出state),初始输出(None),初始循环状态
					# 下一次调用时, 走本else过程
					# 先获取下一个batch输入 input
					# 对input进行处理, 由于每个序列的长度不一致, 处理时间步时候, 先判断是否到达序列末尾, 是则输入"结束符" EOS, 否则获取下一个batch的某一个时间步输入
					# 若是运用注意力机制, 则对input再处理
					# loop_fn通过传入time, previous_output(来自cell自环), previous_state(来自cell自环), previous_loop_state,
					# 进而控制下一时间步cell要传入的next_input(只对input进行加工), 状态(直接来自上一个cell自环), 输出(直接来自上一个cell自环)
					def get_next_input(): #获取下一个输入
						if useTeacherForcing:
							prediction = self.seq_targets[:,time-1] # 这里仅仅包含batch中一个时间步的值, 如果用teacherForcing, 那就以原始输入进行训练
						else:
							output_logits = tf.add(tf.matmul(previous_output, W), b)
							prediction = tf.argmax(output_logits, axis=1) #(128,) 如果不用teacherForcing, 那就以预测出来的(即概率最大的那个)作为输入进行训练
						next_input = tf.nn.embedding_lookup(decoder_embedding, prediction) #(128,100) # 下一个输入为将prediction 进行嵌入
						return next_input # (128,100) next_input指的是下一个batch批量的一个值
					#
					elements_finished = (time >= self.seq_targets_length)  #判断是否到达, 这里决定了tensorarray的序列长度为9
					finished = tf.reduce_all(elements_finished) #Computes the "logical and"
					# 获取下一个batch输入
					input = tf.cond(finished, lambda: tokens_eos_embedded, get_next_input) #  (128,100)如果finished, input = tokens_eos_embeded, 否则调用get_next_input()函数获取下一个输入
					if useAttention:     # (128,100)   (128,50)   attn( (50,)     (128,13,50) )
						input = tf.concat([input, self.attn(previous_state, encoder_outputs)], 1)# (128,150)


					state = previous_state #( 128,50)
					output = previous_output #(128,50)
					loop_state = None
					#        boolin (128,)     , (128,150),  (128,50),  (128,50),      None), 最开始的state是encoder 传入的,然后直接将其返回给decoder_cell(state,input), 随后的state, 皆由decoder_cell自环产生
					return (elements_finished,   input,     state,    output, 		loop_state)
			#tensorarray对象,  (128,50)       ,_   elements_finished = (time >= self.seq_targets_length)  #判断是否到达, 这里决定了tensorarray的序列长度为9
			decoder_outputs_ta, decoder_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn) #,先运行一次loop_fn,获取time=0的input输入,这里隐藏了一个cell自环　(output, cell_state) = cell(next_input, state)
			# shape=(9, 128, 50)
			decoder_outputs = decoder_outputs_ta.stack() #axis参数留空, 默认对axis=0进行堆叠
			#  shape=(128, 9, 50)         shape=(9, 128, 50)
			decoder_outputs = tf.transpose(decoder_outputs, perm=[1,0,2]) # S*B*D -> B*S*D B:batch_size, S:max_steps, D:dim
		    #  128            ,        9         ,    50
			decoder_batch_size, decoder_max_steps, decoder_dim = tf.unstack(tf.shape(decoder_outputs)) # 默认对axis=0 拆散
			#                  为了和后续W,b操作方便, 所以要reshape成2维, decoder_outputs(128,9,50), hidden_dim(50),所以decoder_outputs_flat (128*9, 50)
			decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, config.hidden_dim)) # (128*9, 50)
			decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W), b) #线性变换(128*9, 50) 点积 (50,13),得到(128*9,13)
			# (128,9,13)
			decoder_logits = tf.reshape(decoder_logits_flat, (decoder_batch_size, decoder_max_steps, config.target_vocab_size))
			print(decoder_logits)
		# 所谓输出, 即是最终这13个概率值中最大的概率的索引号对饮的词
		self.out = tf.argmax(decoder_logits, 2) # infer出来的词 , 特征维度(13)上的最大值, 返回形状为(128, 13)

		# 求cross_entropy_with_logits 的时候,其实是用1*9的向量 点积 9*13 的向量, 得到1*13 的向量, 向量里包含13个各个词的概率,加入batch之后, 即是(128,13)
		loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
			labels=self.seq_targets,#(128,9)# 最后一个维度是9, 但是也可以有13dim
			logits=decoder_logits, #(128,9,13)
		)								# self.seqtargets_length = [4, 7, 8, 4, 3, 9, 9, 3, 6, 3, 5, 5, 5, 2, 4, 4, 4, 4, 9, 7, 6, 5, 5, 4, 3, 3, 3, 3, 5, 6, 2, 4, 8,....] len=128
		sequence_mask = tf.sequence_mask(self.seq_targets_length, dtype=tf.float32)#(128,)
		loss = loss * sequence_mask
		self.loss = tf.reduce_mean(loss) # 一个batch128个数据的平均loss值
		
		self.train_op = tf.train.AdamOptimizer(learning_rate=config.learning_rate).minimize(self.loss)
			
			
			
			
