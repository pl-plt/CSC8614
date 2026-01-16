import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time

SEED = 42
torch.manual_seed(SEED)

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

prompt = "The future of artificial intelligence is"
inputs = tokenizer(prompt, return_tensors="pt")

# outputs = model.generate(
#     **inputs,
#     max_length=50,
# )

# text = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print(text)

def generate_once(seed):
    torch.manual_seed(seed)
    out = model.generate(
        **inputs,
        max_length=50,
        do_sample=True,
        temperature=2.0,
        top_k=50,
        top_p=0.95,
        repetition_penalty=2.0,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)

# for s in [1, 2, 3, 4, 5]:
#     print("SEED", s)
#     print(generate_once(s))
#     print("-" * 40)

t0 = time.time()
out_beam = model.generate(
    **inputs,
    max_length=50,
    num_beams=20,
    early_stopping=True
)
txt_beam = tokenizer.decode(out_beam[0], skip_special_tokens=True)
print(txt_beam)
t1 = time.time()
print("Beam search generation time (s):", t1 - t0)