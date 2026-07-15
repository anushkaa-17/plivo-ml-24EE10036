import os
import pickle

class BPETokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.vocab_size = 256

    def train(self, text, vocab_size=512):
        tokens = list(text.encode("utf-8"))
        for i in range(256, vocab_size):
            counts = {}
            for j in range(len(tokens) - 1):
                pair = (tokens[j], tokens[j+1])
                counts[pair] = counts.get(pair, 0) + 1
            if not counts:
                break
            best = max(counts, key=counts.get)
            self.merges[best] = i
            self.vocab[i] = self.vocab[best[0]] + self.vocab[best[1]]
            
            new_tokens = []
            j = 0
            while j < len(tokens):
                if j < len(tokens) - 1 and (tokens[j], tokens[j+1]) == best:
                    new_tokens.append(i)
                    j += 2
                else:
                    new_tokens.append(tokens[j])
                    j += 1
            tokens = new_tokens
        self.vocab_size = vocab_size

    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stats = {}
            for i in range(len(tokens)-1):
                stats[(tokens[i], tokens[i+1])] = 1
            
            pair = min(stats.keys(), key=lambda p: self.merges.get(p, float('inf')))
            if pair not in self.merges:
                break
                
            idx = self.merges[pair]
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair:
                    new_tokens.append(idx)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

    def decode(self, ids):
        b = b"".join(self.vocab[i] for i in ids)
        return b.decode("utf-8", errors="replace")

def load():
    tok = BPETokenizer()
    merge_file = "merges.pkl"
    if os.path.exists(merge_file):
        with open(merge_file, "rb") as f:
            tok.merges, tok.vocab = pickle.load(f)
            tok.vocab_size = len(tok.vocab)
    else:
        print("Training BPE Tokenizer (vocab 512)...")
        text = open("../data/train_corpus.txt", encoding="utf-8").read()
        tok.train(text[:300000], vocab_size=512)
        with open(merge_file, "wb") as f:
            pickle.dump((tok.merges, tok.vocab), f)
    return tok
