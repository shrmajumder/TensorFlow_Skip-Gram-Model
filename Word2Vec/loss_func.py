import tensorflow as tf

def cross_entropy_loss(inputs, true_w):
    """
    ==========================================================================

    inputs: The embeddings for context words. Dimension is [batch_size, embedding_size].
    true_w: The embeddings for predicting words. Dimension of true_w is [batch_size, embedding_size].

    Write the code that calculate A = log(exp({u_o}^T v_c))

    """
    A = tf.log(tf.exp(tf.matmul(true_w,inputs,transpose_a=True)))
   #A=tf.log(tf.exp(np.dot(true_w,inputs)))
    #return tf.subtract(B,A)


    #And write the code that calculate B = log(\sum{exp({u_w}^T v_c)})


    B = tf.log(tf.reduce_sum(tf.exp(tf.matmul(true_w,inputs,transpose_a=True)),axis=1))
    return tf.subtract(B,A)


   
    """
    """

def nce_loss(inputs, weights, biases, labels, sample, unigram_prob):
    """
    ==========================================================================
    
    inputs: Embeddings for context words. Dimension is [batch_size, embedding_size].
    weigths: Weights for nce loss. Dimension is [Vocabulary, embeeding_size].
    biases: Biases for nce loss. Dimension is [Vocabulary, 1].
    labels: Word_ids for predicting words. Dimesion is [batch_size, 1].
    samples: Word_ids for negative samples. Dimension is [num_sampled].
    unigram_prob: Unigram probability. Dimesion is [Vocabulary].

    Implement Noise Contrastive Estimation Loss Here

    ==========================================================================
    """
    
    k = tf.placeholder(dtype=float)
    s_weights = tf.nn.embedding_lookup(weights, labels)
    batch_size = inputs.get_shape()[0]
    embedding_size = inputs.get_shape()[1]
    s_weights = tf.reshape(s_weights, [batch_size, embedding_size])
    s_biases = tf.nn.embedding_lookup(biases, labels)
    s_biases = tf.reshape(s_biases, [batch_size, 1])
    score = tf.add(tf.reduce_sum(tf.multiply(inputs, s_weights), axis=1),tf.transpose(s_biases))
    score=tf.transpose(score)
    print ("Hello 3",score)
   #s_unigram_prob = (tf.nn.embedding_lookup([unigram_prob], labels))
    s_unigram_prob = tf.nn.embedding_lookup([unigram_prob], labels)
    print ("Hello 4",s_unigram_prob)
    k=sample.shape[0]
    print("Hi5")
    sample_prob = tf.log(tf.scalar_mul(k, s_unigram_prob)+0.00000001)
    result1= tf.log(tf.sigmoid(tf.subtract(score, sample_prob))+0.00000001)

    print ("Hi2",result1)
    neg_sample_size = sample.shape[0]
    neg_weights = tf.nn.embedding_lookup(weights, sample)
    neg_weights = tf.reshape(neg_weights,[neg_sample_size, embedding_size])

    neg_biases = tf.nn.embedding_lookup(biases, sample)
   #neg_biases = tf.reshape(neg_biases, [neg_sample_size, 1])
    neg_biases = tf.transpose(neg_biases)

    neg_unigram_prob = tf.nn.embedding_lookup([unigram_prob], sample)
   #neg_unigram_prob=tf.reshape(neg_sample_size,1)


    score = tf.add(tf.matmul(inputs,neg_weights, transpose_b=True), neg_biases)


  #neg_unigram_prob=tf.scalar_mul(neg_weights, unigram_prob)
    neg_sample_prob = tf.log(len(sample) * neg_unigram_prob+0.00000001)
    one_vec=tf.ones([neg_sample_size,1],tf.float32)
    result2=tf.reduce_sum(tf.log(tf.subtract(tf.transpose(one_vec),tf.sigmoid(tf.subtract(score, neg_sample_prob)))+0.00000001),axis=1)

    print ("1:",result1)
    print ("2:",result2)

    final_result=tf.negative(tf.add(result1,result2))
    return final_result


  #sec_part=tf.reduce_sum(tf.log(tf.subtract(1,result)))
    
    