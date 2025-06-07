from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

# Carregar os perfis gerados
df = pd.read_csv("perfis_compra_llm.csv")  # precisa ter coluna 'perfil_compra_texto'

# Carregar o modelo de embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')

# Gerar os embeddings
embeddings = model.encode(df['perfil_compra_texto'].tolist(), show_progress_bar=True)

# Transformar para DataFrame
embeddings_df = pd.DataFrame(embeddings)
embeddings_df.insert(0, 'customer_email', df['customer_email'])

# Salvar para uso posterior (ex: clustering ou recomendação)
embeddings_df.to_csv("embeddings_perfis.csv", index=False)
print("Embeddings gerados e salvos em embeddings_perfis.csv")
