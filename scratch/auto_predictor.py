import numpy as np
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchtext as tt

def l2norm(x):
    norm = x.norm(p=2, dim=1, keepdim=True)
    return x.div(norm.expand_as(x))

def argmax(x):
    return x.max(1)[1]

TEXT = tt.data.ReversibleField(sequential=True, tokenize= lambda x: x.split(), lower=True)
LABEL = tt.data.Field(sequential=False, use_vocab=False)

# Lookup words in GloVe, then find the most similar other word
words = ["king","alpha","queen","beta","man","gamma","woman","delta", "run", "walk"]
idx2word = {idx: word for idx, word in enumerate(words)}
TEXT.build_vocab([words], vectors="glove.6B.100d")
k, a, q, b, m, g, w, d = TEXT.vocab.vectors[2:10]

# https://stackoverflow.com/questions/37558899/efficiently-finding-closest-word-in-tensorflow-embedding
embedding = TEXT.vocab.vectors[2:]

batch_array = torch.stack([w, d])
normed_embedding = l2norm(embedding)
normed_array = l2norm(batch_array)

cosine_similarity = torch.matmul(normed_array, torch.transpose(normed_embedding, 1, 0))
closest_words = np.argsort(cosine_similarity)[:,-2]
print([idx2word[word] for word in closest_words])



#import revtok

vocab_tensor = torch.Tensor(TEXT.vocab.vectors[2:6])

TEXT.reverse(vocab_tensor)

# =====================================



torch.manual_seed(1)

def invert_map(m):
    return {v: k for k, v in m.items()}

def argmax(x):
    values, indices = torch.max(x, 1)
    return indices

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)


training_data = [
    ("The dog ate the apple".split(), ["NN","NN","NN","NN","NN"]),
    ("Everybody read that book".split(), ["NN","NN","NN","NN"])
]

word_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

print(word_to_ix)
tag_to_ix = {"DET": 0, "NN": 1, "V": 2}
ix_to_tag = invert_map(tag_to_ix)
print(ix_to_tag)

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 6
HIDDEN_DIM = 6


class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
inputs = prepare_sequence(training_data[0][0], word_to_ix)
tag_scores = model(inputs)
print(tag_scores)

for epoch in range(300):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Variables of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

# See what the scores are after training
inputs = prepare_sequence(training_data[0][0], word_to_ix)
tag_scores = model(inputs)
# The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
#  for word i. The predicted tag is the maximum scoring tag.
# Here, we can see the predicted sequence below is 0 1 2 0 1
# since 0 is index of the maximum value of row 1,
# 1 is the index of maximum value of row 2, etc.
# Which is DET NOUN VERB DET NOUN, the correct sequence!
print(tag_scores)


print("Classification")
print(training_data[0][0])
print([ix_to_tag[i] for i in argmax(tag_scores).data])


Vocab

tt.vocab.Vocab({})
'glove.42B.300d'
