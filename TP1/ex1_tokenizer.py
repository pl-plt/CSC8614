from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
phrase = "GPT models use BPE tokenization to process unusual words like antidisestablishmentarianism."

tokens = tokenizer.tokenize(phrase)

print(tokens)

# TODO: obtenir les IDs
token_ids = tokenizer.encode(phrase)
print("Token IDs:", token_ids)

print("Détails par token:")
for tid in token_ids:
    # TODO: décoder un seul token id
    txt = tokenizer.decode([tid])
    print(tid, repr(txt))

phrase2 = "GPT models use BPE tokenization to process unusual words like antidisestablishmentarianism."

tokens2 = tokenizer.tokenize(phrase2)
print(tokens2)

# Extraire uniquement les tokens correspondant au mot long
long_word = "antidisestablishmentarianism"
long_word_tokens = tokenizer.tokenize(long_word)
print("Tokens du mot long:", long_word_tokens)

# Compter le nombre de sous-tokens
num_subtokens = len(long_word_tokens)
print(f"Nombre de sous-tokens pour '{long_word}': {num_subtokens}")