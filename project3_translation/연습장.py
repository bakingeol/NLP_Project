#%%
from transformers import AutoModel

# %% #################################################### 1번 예시 opus-mt-en-ko, markintokenizer- 512
from transformers import MarianMTModel, MarianTokenizer

model_name = "Helsinki-NLP/opus-mt-en-ko"
tokenizer = MarianTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)
#%%

src_text = [
    "The government will shorten the current seven-day isolation mandate period for COVID-19 patients to five, as the government is considering downgrading the virus to a lower infection level, said Prime Minister Han Duck-soo during a meeting of the Central Disaster and Safety Countermeasures Headquarters."
]
translated = model.generate(**tokenizer(src_text, return_tensors="pt", padding=True))

for t in translated:
    print( tokenizer.decode(t, skip_special_tokens=True) )
#%% #################################################### 2번 예시 facebook - m2m100 모델, max_len :1024
from transformers import M2M100Config, M2M100ForConditionalGeneration, M2M100Tokenizer

model = M2M100ForConditionalGeneration.from_pretrained("facebook/m2m100_418M")
tokenizer = M2M100Tokenizer.from_pretrained("facebook/m2m100_418M", src_lang="en", tgt_lang="ko")

# %%
#src_text = "Life is like a box of chocolates."
src_text = ["The presidential office on Thursday ruled out resumption of seafood imports from Japan's Fukushima region over radiation contamination concerns, saying the health and safety of people are the foremost priority. Fukushima seafood will never come into the country, the office said in a notice to media. With regard to the import of Japanese seafood products, the government's stance remains unchanged that the health and safety of the people are the top priority."]
encoded_en = tokenizer(src_text[0], return_tensors="pt")
generated_tokens = model.generate(**encoded_en, forced_bos_token_id=tokenizer.get_lang_id("ko"))
answer = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
''' fine-tuning 시 loss 값 뽑는 코드(forward pass)

model_inputs = tokenizer(src_text, text_target=tgt_text, return_tensors="pt")
loss = model(**model_inputs).loss  # forward pass
'''
#%%
answer
#%% #################################################### . microsoft/unicoderlarge"
from transformers import pipeline

translator = pipeline(
    "translation_en_to_ko",
    model="microsoft/unicoderlarge",
    tokenizer="microsoft/unicoderlarge"
)
# %%
# %%
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
#%%
src_text = "Life is like a box of chocolates."
tokenizer = T5Tokenizer.from_pretrained("t5-base", src_lang="en", tgt_lang="ko")
model = T5ForConditionalGeneration.from_pretrained("t5-base")
#%%
encoded_en = tokenizer(src_text, return_tensors="pt")
generated_tokens = model.generate(**encoded_en)
generated_tokens
#%%
tokenizer.batch_decode(generated_tokens.squeeze(0).tolist())
# %%
################### kot5 모델 https://huggingface.co/psyche/KoT5/discussions

#%%#
#################################### t5
from transformers import AutoTokenizer, T5ForConditionalGeneration

tokenizer = AutoTokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")
# %%
