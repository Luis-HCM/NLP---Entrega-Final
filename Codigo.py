
# Pegar os artigos diretamento do wikipedia
import wikipediaapi
import re

ARTIGO = 'Terra'

user_agent = 'MeuApp/0.1 (http://meusite.com)'

wiki_wiki = wikipediaapi.Wikipedia(
    language='pt',
    user_agent=user_agent
)

page = wiki_wiki.page(ARTIGO).text
page = re.sub(r'[\n\s+]', ' ',page)


# Função que separa o texto em tamanhos fixos de 200 caracteres.
def separate_the_text(page):
    PAGES = [pages.strip() for pages in re.split(r'(?<=[?!\.,;—])', page)]
    lista = ['']
    while True:

        if len(PAGES) > 0:
            
            LENGTH = len(lista[-1]) + len(PAGES[0])
            if LENGTH < 200:
                TEXTO = ' '.join([lista[-1], PAGES[0]])
                lista[-1] = TEXTO
                
            else:
                lista.append(PAGES[0])

            PAGES.pop(0)
        else:
            break

    lista[0] = lista[0].strip()
    return lista




# Função para fazer a o embedding dos textos.



from transformers import BertModel, BertTokenizer
import torch
from tqdm import tqdm



def embedding_search(text):
    
    # para cada modelo deve se mudar o endereço onde o modelo pre-treinado esta salvo.
    tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
    model = BertModel.from_pretrained('neuralmind/bert-base-portuguese-cased')

    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length = 512)
    with torch.no_grad():

        outputs = model(**inputs)
   
    embedding = outputs.last_hidden_state[:, 0, :]

    return embedding




def embedding_text(texts):

    tokenizer = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
    model = BertModel.from_pretrained('neuralmind/bert-base-portuguese-cased')

    EMBEDDING = []

    for text in tqdm(texts):

        inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True,  max_length = 512)

        with torch.no_grad():

            outputs = model(**inputs)

        sentence_embedding = outputs.last_hidden_state[:, 0, :]

        EMBEDDING.append(sentence_embedding)
    
    return EMBEDDING



# Função para fazer a comparação da similaridade de cosseno entre os embedding

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd


def results(search,texts,embedding):

    dis = pd.DataFrame(columns=['similaridade','frase'])
    for i in range(len(texts)):
        X = np.array(embedding[i])
        Y = np.array(search)
        similarity = cosine_similarity(X,Y)
        dis.loc[len(dis)] = similarity[0][0], texts[i]
        
    dis.sort_values('similaridade',ascending=False,inplace=True)
    dis.reset_index(inplace=True,drop=True)
    
    dis.index = dis.index + 1
    return dis



# Fazer as limpezas das stopwords e o processo de teste da janela deslizante.

import re
import nltk
from nltk.corpus import stopwords

def limpa_frase(frase_completa):
    
    stop_words = set(stopwords.words("portuguese"))

    texto_limpo = re.sub(r'[^a-zA-Z0-9á-úÁ-ÚçÇãõÕ\s]', ' ', frase_completa)

    palavras = texto_limpo.split()

    palavras_filtradas = [palavra for palavra in palavras if palavra.lower() not in stop_words]

    texto_final = " ".join(palavras_filtradas)

    return texto_final


import pandas as pd
from tqdm import tqdm



resultado = pd.DataFrame(columns=['Pesquisa', 'Resultado'])

for frase in tqdm(texts):
    
    texto = limpa_frase(frase)
    texto_dividido = texto.split()


    for i in range(len(texto_dividido) - 2):

        search_text = ' '.join(texto_dividido[i:i+3])
        search = embedding_search(search_text)
        df = results(search, texts, embedding)
        index_frase = df[df['frase'] == frase].index.tolist()

        resultado.loc[len(resultado)] = [search_text, index_frase[0]]


# Fazer a criação do processo de tokenização do modelo Transformers.

from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizerFast, BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from torch.utils.data import Dataset
import torch

# Treinar o Tokenizer


# https://huggingface.co/docs/tokenizers/v0.20.3/en/components?code=python#normalizers
# https://huggingface.co/docs/tokenizers/api/normalizers
tokenizer = BertWordPieceTokenizer() # Todas as conf estão como padrão do BERT

# Usando o WordLevelTrainer pois, é mais semelhante ao BERT
# https://huggingface.co/docs/tokenizers/v0.20.3/en/api/trainers#tokenizers.trainers.WordLevelTrainer
# https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertConfig
tokenizer.train(
    files=['text.txt'],
    vocab_size=30522,  # O tamanho do vocabulário final, incluindo todos os tokens e alfabeto. Tamanho padrão do vocabulário do BERT
    min_frequency=0, # A frequência mínima que um par deve ter para ser mesclado.
    show_progress=True, # Mostra barras de progresso durante o treinamento.
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"], # Uma lista de tokens especiais que o modelo deve conhecer.
)


import os

# Verifique e crie o diretório se ele não existir
os.makedirs(f"./tokenizer{ARTIGO}", exist_ok=True)

#Carregar o Tokenizer com a biblioteca Transformers
# BertTokenizerFast é necessário para compatibilidade com a biblioteca Hugging Face.
tokenizer = BertTokenizerFast.from_pretrained(f"tokenizer{ARTIGO}")


# 3. Criar a configuração do modelo
# https://huggingface.co/transformers/v3.0.2/model_doc/bert.html#bertconfig
config = BertConfig()

# 4. Inicializar o modelo não treinado
# https://huggingface.co/transformers/v3.0.2/model_doc/bert.html#bertformaskedlm
model = BertForMaskedLM(config)


# Treinar o modelo para, o processo é o mesmo para o fine-tuning e para o transformers.


# Preparar seus dados para o treinamento
class TextDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size=512):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        batch_encoding = tokenizer(
            lines,
            add_special_tokens=True,
            truncation=True,
            max_length=block_size,
            padding="max_length",
            return_tensors='pt',
        )
        self.input_ids = batch_encoding['input_ids']
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {'input_ids': self.input_ids[idx]}
    
# Crie o dataset
dataset = TextDataset(tokenizer, "text.txt")

# Configurar o Data Collator para aplicar o mascaramento

# https://huggingface.co/docs/transformers/v4.46.2/en/main_classes/data_collator#transformers.DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer
)

# Configurar os argumentos de treinamento
# https://huggingface.co/docs/transformers/v4.46.2/en/main_classes/trainer#transformers.TrainingArguments


training_args = TrainingArguments(
    output_dir=f"./bert-from-scratch{ARTIGO}",
    overwrite_output_dir=True,
    num_train_epochs=3,  # Valor estipulado de acordo com a documentação do BERT
    per_device_train_batch_size=32,# Valor estipulado de acordo com a documentação do BERT
    disable_tqdm=False,
)


# Iniciar o processo de treinamento
# https://huggingface.co/docs/transformers/v4.46.2/en/main_classes/trainer#api-reference%20][%20transformers.Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)

# Iniciar o treinamento
trainer.train()


# Salvar o modelo e o tokenizer treinados
trainer.save_model(f"./bert-from-scratch{ARTIGO}")
tokenizer.save_pretrained(f"./bert-from-scratch{ARTIGO}")