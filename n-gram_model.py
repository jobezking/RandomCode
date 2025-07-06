import re
import random
from collections import defaultdict, Counter

def tokenize(text):
    text = text.lower()
    return re.findall(r'\b\w+\b', text)

def generate_ngrams(tokens, n):
    return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def build_ngram_model(text, n):
    tokens = tokenize(text)
    ngrams = generate_ngrams(tokens, n)
    model = defaultdict(Counter)
    for ngram in ngrams:
        context, target = ngram[:-1], ngram[-1]
        model[context][target] += 1
    return model

def generate_text(model, context, num_words=20):
    generated = list(context)
    for _ in range(num_words):
        current_context = tuple(generated[-(len(context)):])
        if current_context not in model:
            break
        next_word = random.choices(
            list(model[current_context].keys()),
            weights=list(model[current_context].values()),
            k=1
        )[0]
        generated.append(next_word)
    return ' '.join(generated)

text = """The future is already here, it's just not evenly distributed. 
The future belongs to those who believe in the beauty of their dreams. 
The present is theirs; the future, for which I have really worked, is mine. 
The best way to predict the future is to create it. 
The only thing we know about the future is that it will be different. 
The truth is incontrovertible. Malice may attack it, 
ignorance may deride it, but in the end, there it is."""

bigram_model = build_ngram_model(text, 2)
unigram_model = Counter(tokenize(text))
top_unigrams = unigram_model.most_common(3)
bigram_preds = dict(bigram_model[("the",)])
total = sum(bigram_preds.values())
bigram_probs = sorted([(word, count/total) for word, count in bigram_preds.items()], key=lambda x: x[1], reverse=True)[:3]

print(f"Unigram top predictions (context-free): {top_unigrams}")
print(f"Bigram predictions after 'the': {bigram_probs}")
print("Generated text using bigram model:", generate_text(bigram_model, ('the',), 10))
