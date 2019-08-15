import tensorflow as tf
import numpy as np
import random
import time
# from model_seq2seq_contrib import Seq2seq
from model_seq2seq import Seq2seq
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
tf_config = tf.ConfigProto(allow_soft_placement=True)
tf_config.gpu_options.allow_growth = True 

class Config(object):
	embedding_dim = 100
	hidden_dim = 50
	batch_size = 128
	learning_rate = 0.005
	source_vocab_size = None
	target_vocab_size = None


def load_data(path):
	num2en = {"1":"one", "2":"two", "3":"three", "4":"four", "5":"five", "6":"six", "7":"seven", "8":"eight", "9":"nine", "0":"zero"}
	docs_source = []
	docs_target = []
	for i in range(10000):
		doc_len = random.randint(1,8) # 随机产生1到8之间的一个整数
		doc_source = []
		doc_target = []
		for j in range(doc_len):
			num = str(random.randint(0,9))
			doc_source.append(num)
			doc_target.append(num2en[num])
		docs_source.append(doc_source)
		docs_target.append(doc_target)
	
	return docs_source, docs_target  # 10000个不等长的列表(输入句子) ['3', '8'],   ['9', '8', '5', '5'],          ['2', '2', '9', '8', '1', '6'], ['6', '5', '5', '6', '2', '2', '4']...
									 # 10000个不等长的列表(输出句子) ['three', 'eight'], ['nine', 'eight', 'five', 'five'], ['six', 'five', 'five', 'six', 'two', 'two', 'four']....

def make_vocab(docs):
	w2i = {"_PAD":0, "_GO":1, "_EOS":2}
	i2w = {0:"_PAD", 1:"_GO", 2:"_EOS"}
	for doc in docs:
		for w in doc:
			if w not in w2i:
				i2w[len(w2i)] = w
				w2i[w] = len(w2i)
	return w2i, i2w

	
	
def doc_to_seq(docs):
	w2i = {"_PAD":0, "_GO":1, "_EOS":2}
	i2w = {0:"_PAD", 1:"_GO", 2:"_EOS"}
	seqs = []
	for doc in docs:
		seq = []
		for w in doc:
			if w not in w2i:
				i2w[len(w2i)] = w
				w2i[w] = len(w2i)
			seq.append(w2i[w])
		seqs.append(seq)
	return seqs, w2i, i2w


def get_batch(docs_source, w2i_source, docs_target, w2i_target, batch_size):
	ps = [] #ps列表为长度为128的0到10000之间的随机整数
	while len(ps) < batch_size:
		ps.append(random.randint(0, len(docs_source)-1)) #产生0到10000的随机整数128个 ,并组装到ps列表中
	
	source_batch = []
	target_batch = []
	
	source_lens = [len(docs_source[p]) for p in ps] # 每个batch里所有输入句子的长度的列表
	target_lens = [len(docs_target[p])+1 for p in ps] #每个batch里所有输出句子的长度加1的列表
	
	max_source_len = max(source_lens) #最大输入长度8
	max_target_len = max(target_lens) #最大target长度9
		
	for p in ps:
		source_seq = [w2i_source[w] for w in docs_source[p]] + [w2i_source["_PAD"]]*(max_source_len-len(docs_source[p]))
		target_seq = [w2i_target[w] for w in docs_target[p]] + [w2i_target["_EOS"]] + [w2i_target["_PAD"]]*(max_target_len-1-len(docs_target[p]))
		source_batch.append(source_seq)
		target_batch.append(target_seq)
	
	return source_batch, source_lens, target_batch, target_lens #返回一个batch(128组)经过填充,且长度均为8的输入数组及长度列表, target输出数组长度均为9的target数组及长度列表
	# seq_inputs = source_batch:    [6, 4, 5, 0, 0, 0, 0, 0],   [12, 10, 12, 12, 5, 6, 0, 0],  [9, 9, 12, 6, 7, 12, 12, 0] len=8
	# seq_inputs_length = source_length: [3, 6, 7, 3, 2, 8, 8, 2, 5, 2, 4, 4, 4, 1, 3, 3, 3, 3, 8, 6, 5, 4, 4, 3, 2, 2, 2, 2, 4, 5, 1, 3, 7, 8, 5, 6,......] len=128

	# seq_targets = target_batch:  9个  [6, 4, 5, 2, 0, 0, 0, 0, 0],  [12, 10, 12, 12, 5, 6, 2, 0, 0],  [9, 9, 12, 6, 7, 12, 12, 2, 0] len = 9
	# seq_targets_length = target_length:   [4, 7, 8, 4, 3, 9, 9, 3, 6, 3, 5, 5, 5, 2, 4, 4, 4, 4, 9, 7, 6, 5, 5, 4, 3, 3, 3, 3, 5, 6, 2, 4, 8,....] len=128

if __name__ == "__main__":

	print("(1)load data......")
	docs_source, docs_target = load_data("")
	w2i_source, i2w_source = make_vocab(docs_source) #制作输入句子(输入就是数字列表)的词汇表

	# {'_PAD': 0, '_GO': 1, '_EOS': 2, '1': 3, '0': 4, '4': 5, '3': 6, '9': 7, '6': 8, '8': 9, '7': 10, '5': 11, '2': 12} len = 13     w2i
	# {0: '_PAD', 1: '_GO', 2: '_EOS', 3: '1', 4: '0', 5: '4', 6: '3', 7: '9', 8: '6', 9: '8', 10: '7', 11: '5', 12: '2'} len = 13     i2w
	w2i_target, i2w_target = make_vocab(docs_target) #制作输出句子(输出为对应数字的英文表示)

 	#{0: '_PAD', 1: '_GO', 2: '_EOS', 3: 'nine', 4: 'three', 5: 'six', 6: 'five', 7: 'two', 8: 'one', 9: 'four', 10: 'eight', 11: 'seven', 12: 'zero'}  len = 13
	#{'_PAD': 0, '_GO': 1, '_EOS': 2, 'nine': 3, 'three': 4, 'six': 5, 'five': 6, 'two': 7, 'one': 8, 'four': 9, 'eight': 10, 'seven': 11, 'zero': 12}  len = 13
	
	print("(2) build model......")
	config = Config()
	config.source_vocab_size = len(w2i_source) #13
	config.target_vocab_size = len(w2i_target) #13
	model = Seq2seq(config=config, w2i_target=w2i_target, useTeacherForcing=True, useAttention=True, useBeamSearch=1)
	
	
	print("(3) run model......")
	batches = 3000
	print_every = 100
	
	with tf.Session(config=tf_config) as sess:
		tf.summary.FileWriter('graph', sess.graph)
		saver = tf.train.Saver()
		sess.run(tf.global_variables_initializer())
		
		losses = []
		total_loss = 0
		for batch in range(batches):
			source_batch, source_lens, target_batch, target_lens = get_batch(docs_source, w2i_source, docs_target, w2i_target, config.batch_size)
			
			feed_dict = {
				model.seq_inputs: source_batch,#(128,8) 填充了0的
				model.seq_inputs_length: source_lens,#(128,) 实际长度 , 变长
				model.seq_targets: target_batch, #(128,9)
				model.seq_targets_length: target_lens #(128,)
			}
			
			loss, _ = sess.run([model.loss, model.train_op], feed_dict)
			total_loss += loss
			
			if batch % print_every == 0 and batch > 0:
				print_loss = total_loss if batch == 0 else total_loss / print_every
				losses.append(print_loss)
				total_loss = 0
				print("-----------------------------")
				print("batch:",batch,"/",batches)
				print("time:",time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time())))
				print("loss:",print_loss)
				
				print("samples:\n")
				predict_batch = sess.run(model.out, feed_dict)
				for i in range(3):
					print("in:", [i2w_source[num] for num in source_batch[i] if i2w_source[num] != "_PAD"])
					print("out:",[i2w_target[num] for num in predict_batch[i] if i2w_target[num] != "_PAD"])
					print("tar:",[i2w_target[num] for num in target_batch[i] if i2w_target[num] != "_PAD"])
					print("")
		
		print(losses)
		print(saver.save(sess, "checkpoint/model.ckpt"))		
		
	


