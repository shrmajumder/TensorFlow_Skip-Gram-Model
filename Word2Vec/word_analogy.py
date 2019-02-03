import os
import pickle
import numpy as np


model_path = './models/'
loss_model = 'nce'
#loss_model = 'nce'

model_filepath = os.path.join(model_path, 'word2vec_%snew_sh6.model'%(loss_model))

dictionary, steps, embeddings = pickle.load(open(model_filepath, 'rb'))

"""
==========================================================================

Write code to evaluate a relation between pairs of words.
You can access your trained model via dictionary and embeddings.
dictionary[word] will give you word_id
and embeddings[word_id] will return the embedding for that word.

word_id = dictionary[word]
v1 = embeddings[word_id]

or simply

v1 = embeddings[dictionary[word_id]]

==========================================================================
"""
from scipy import spatial
with open("word_analogy_dev.txt") as fp :
   words = fp.readlines()

words = [x.replace('\n', '') for x in words]
output_file =open("word_analogy_dev_predictions_new_sh8.txt","w")

for i in words :
   #print (i)
   list_ex=[]
   #rem_pipe=[elem.strip().split('||') for elem in words]
   #words = [x.replace('\n', '') for x in words]
   rem_ex,rem_ch=i.split("||")
   explanation=rem_ex.split(",")
   choice=rem_ch.split(",")
   for j in explanation :
       word_embed= j.split(":")
       #print (word_embed)
       word_embed = [x.replace('"' , '')  for x in word_embed]
       #word_embed = [x.replace('\n' , '')  for x in word_embed]
       #print (embeddings[dictionary[word_embed[0]]])
       #print (embeddings[dictionary[word_embed[1]]])
       uc1=(embeddings[dictionary[word_embed[0]]])
       uc2=(embeddings[dictionary[word_embed[1]]])
       difference=np.subtract(uc1,uc2)
       #print (difference)
       list_ex.append(difference)
   mean=np.mean(list_ex)
   list_ch=[]
   for j in choice :
       word_embed= j.split(":")
       #print (word_embed)
       word_embed = [x.replace('"' , '')  for x in word_embed]
       #word_embed = [x.replace('\n' , '')  for x in word_embed]
       #print (word_embed)
       u01=(embeddings[dictionary[word_embed[0]]])
       u02=(embeddings[dictionary[word_embed[1]]])
       difference=np.subtract(u01, u02)
       #print (difference)
       similarity = (1-spatial.distance.cosine(difference, mean))
       print (similarity)
       #print (j)
       output_file.write(j)
       #print (similarity)
       output_file.write(" ")
       list_ch.append(similarity)
   
   most_illustrative_pair= (choice[list_ch.index(max(list_ch))])
   least_illustrative_pair= (choice[list_ch.index(min(list_ch))])
   #most_illustrative_pair=choice[index(max(list_ch))]
   #print(most_illustrative_pair)
   
   output_file.write(least_illustrative_pair)
   output_file.write(" ")
   output_file.write(most_illustrative_pair)
   #output_file.write(" ")
   output_file.write("\n")

output_file.close()

   
