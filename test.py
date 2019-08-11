import tensorflow as tf
import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()
inputs = tf.convert_to_tensor([[6, 4, 5, 0, 0, 0, 0, 0],[12, 10, 12, 12, 5, 6, 0, 0]],dtype=tf.float32)
encoder_embedding = tf.contrib.eager.Variable(tf.random_uniform([13., 100.]), dtype=tf.float32, name='encoder_embedding') # shape=(13,100) , 要嵌入的空间
encoder_inputs_embedded = tf.nn.embedding_lookup(encoder_embedding, inputs)
print(encoder_inputs_embedded)