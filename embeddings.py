# we need to first identify how to convert the input into the "Input Embedding"

import torch
import torch.nn as nn

sentence = "Life is short, eat dessert first"

dc = {s:i for i,s in enumerate(sorted(sentence.replace(',',' ').split()))}

sentence_int = torch.tensor(
    [dc[s] for s in sentence.replace(',', ' ').split()]
)

print(f"Sentence number embedding: {sentence_int}")


vocab_size = 50_000
torch.manual_seed(123)

# we need to make an object of the torch.nn.Embedding class 

# we use 3 dim embedding (for illustration purpose only) otherwise in Llama2 we have embedding size 4096
embed = torch.nn.Embedding(vocab_size, 3)

# we need to pass the sentence from the object: "embed" to create the vector embedding of the sentence
embedded_sentence = embed(sentence_int).detach()

print(f"embedded sentence is: {embedded_sentence}")
print(f"size of the embedded sentence is: {embedded_sentence.shape}")


# now we  need to calculate the scaled dot product attention
torch.manual_seed(123)

d = embedded_sentence.shape[1]

d_q, d_k, d_v = 2,2,4

W_query = torch.nn.Parameter(torch.rand(d,d_q))
W_key = torch.nn.Parameter(torch.rand(d,d_k))
W_value = torch.nn.Parameter(torch.rand(d,d_v))

print(f"Weight matrix of Query shape: {W_query.shape}")
print(f"Weight matrix of Key shape: {W_key.shape}")
print(f"Weight matrix of Value shape: {W_value.shape}")


x_2 = embedded_sentence[1]

query_2 = x_2 @ W_query
key_2 = x_2 @ W_key
value_2 = x_2 @ W_value

print(f"Shape of query_2: {query_2.shape}")
print(f"Shape of key_2: {key_2.shape}")
print(f"Shape of value_2: {value_2.shape}")

# now lets generalize this for all keys and value element

keys = embedded_sentence @ W_key
values = embedded_sentence @ W_value

print(f"Key shape: {keys.shape}")
print(f"Values shape: {values.shape}")

# we need to multiply Query and Key(Transpose)

omega_2 = query_2 @ keys.T
print(omega_2)


# now once we got un-normalized omega, we need to make them noramlized by applying sofmax function as well as divede by sqrt(d_k)

import torch.nn.functional as F

attention_weights_2 = F.softmax(omega_2 / d_k ** 0.5, dim=0)

print(attention_weights_2)


# final step is, we need to compute the context vector z_2, which is the attention weighted version of our original version of query inpur x_2

context_vector_2 = attention_weights_2 @ values
print(f"shape of contect vector: {context_vector_2.shape}")
print(f"context vector: {context_vector_2}")




