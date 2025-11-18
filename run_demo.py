from transformers import MambaConfig, MambaForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-370m-hf")
model = MambaForCausalLM.from_pretrained("state-spaces/mamba-370m-hf")
# Return the attention mask and pass it to generate to avoid the warning
q = "Write me a poem about a lonely computer:"
inputs = tokenizer(q, return_tensors="pt", return_attention_mask=True)
input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]

out = model.generate(
	input_ids=input_ids,
	attention_mask=attention_mask,
	max_new_tokens=100,
	do_sample=True,
	temperature=0.7,
	top_p=0.95,
	repetition_penalty=1.2,
	no_repeat_ngram_size=3,
)
# Decode only the newly generated tokens (exclude the original prompt)
generated_tokens = out[:, input_ids.shape[-1]:]
print(q, *tokenizer.batch_decode(generated_tokens, skip_special_tokens=True))