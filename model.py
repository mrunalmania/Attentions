import torch
import torch.nn as nn

torch.manual_seed(123)


class SelfAttention(nn.Module):
    
    def __init__(self, d_in, d_out_kq, d_out_v) -> None:
        super().__init__()
        self.d_out_kq = d_out_kq
        self.W_query = nn.Parameter(torch.rand(d_in, d_out_kq))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out_kq))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out_v))

    def forward(self, x):
        # first we need to do the weighted mupltiplication of key, query, value

        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value

        # now  we need to calculate the un-normalized self-attention
        attn_scores = queries @ keys.T

        attn_weights = torch.softmax(
            attn_scores / self.d_out_kq ** 0.5, dim=-1
        )

        context_vec = attn_weights @ values

        return context_vec
    


class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out_kq, d_out_v, num_heads) -> None:
        super().__init__()
        self.heads = nn.ModuleList(
            [SelfAttention(d_in, d_out_kq, d_out_v)
             for _ in range(num_heads)]

        )
    
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
        # dim = -1 means last available dimension


sentence = "Life is short, eat dessert first"

dc = {s:i for i,s in enumerate(sorted(sentence.replace(',',' ').split()))}

sentence_int = torch.tensor(
    [dc[s] for s in sentence.replace(',', ' ').split()]
)

print(f"Sentence number embedding: {sentence_int}")


vocab_size = 50_000


# we need to make an object of the torch.nn.Embedding class 

# we use 3 dim embedding (for illustration purpose only) otherwise in Llama2 we have embedding size 4096
embed = torch.nn.Embedding(vocab_size, 3)

# we need to pass the sentence from the object: "embed" to create the vector embedding of the sentence
embedded_sentence = embed(sentence_int).detach()


# intialize the d_in, d_out_kq, d_out_v
d_in, d_out_kq, d_out_v = 3,2,1
sa = SelfAttention(d_in=d_in, d_out_kq=d_out_kq, d_out_v=d_out_v)

print("*********** Self Attention *****************")
print(sa(embedded_sentence))

print("*********** Now Multi-Head attention ********")


block_size = embedded_sentence.shape[1]

mha = MultiHeadAttention(
    d_in=d_in, d_out_kq=d_out_kq, d_out_v=d_out_v, num_heads=4
)

context_vecs = mha(embedded_sentence)

print(f"contect vectors shape: {context_vecs.shape}")
print(f"context vector: {context_vecs}")

## Cross self-attention

class CrossSelfAttention(nn.Module):
    def __init__(self, d_in, d_out_kq, d_out_v) -> None:
        super().__init__()
        self.d_out_kq = d_out_kq
        self.W_query = nn.Parameter(torch.rand(d_in, d_out_kq))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out_kq))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out_v))

    def forward(self,x_1, x_2):
        queries_1 = x_1 @ self.W_query
        keys_2 = x_2 @ self.W_key
        values_2 = x_2 @ self.W_value

        attn_scores = queries_1 @ keys_2.T
        # now we need to do normalization
        attn_weights = torch.softmax(
            attn_scores/ self.d_out_kq ** 0.5,
            dim = -1
        )

        # now we need to calculate the contxt vector
        context_vec = attn_weights @ values_2
        return context_vec
    
d_in, d_out_kq, d_out_v = 3,2,4

crossattn = CrossSelfAttention(d_in=d_in, d_out_kq=d_out_kq, d_out_v=d_out_v)
first_input = embedded_sentence
second_input = torch.rand(8,d_in)

print(f"First input shape:{first_input.shape}")
print(f"Second input shape: {second_input.shape}")

context_vectors = crossattn(first_input, second_input)

print(f"context vector: {context_vectors}")
print(f"context vecotrs shape: {context_vectors.shape}")


