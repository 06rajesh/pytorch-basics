import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

# Tags are: DET - determiner; NN - noun; V - verb
# For example, the word "The" is a determiner
training_data = [
    ("The dog ate the apple".split(), "NEG"),
    ("Everybody read that book".split(), "POS"),
    ("I fell nothing".split(), "NEUT"),
    ("I split on your grave".split(), "NEG"),
    ("Everybody like the apple".split(), "POS"),
]

word_to_ix = {}

# For each words-list (sentence) and tags-list in each tuple of training_data
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

print(word_to_ix)
tag_to_ix = {"NEG": 0, "POS": 1, "NEUT": 2}  # Assign each tag with a unique index

EMBEDDING_DIM = 6
HIDDEN_DIM = 6

# LSTM model for classification problem, output
# one class from a sentence
class LSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, output_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, (h_n, c_n)  = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.linear(h_n[-1]) ## select the last hidden state from the lstm output
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
with torch.no_grad():
    inputs = prepare_sequence(training_data[2][0], word_to_ix)
    tag_scores = model(inputs)
    print(tag_scores)
#
for epoch in range(50):
    for sentence, tag in training_data:
        model.zero_grad()

        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = torch.tensor([tag_to_ix[tag]])

        tag_scores = model(sentence_in)

        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

with torch.no_grad():
    inputs = prepare_sequence(training_data[2][0], word_to_ix)
    tag_scores = model(inputs)
    print(tag_scores)