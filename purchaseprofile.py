import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sentence_transformers import SentenceTransformer
from connect_athena import run_query, get_query_results, DATABASE, OUTPUT_LOCATION, wait_for_query_completion
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from collections import Counter

# Função responsável por executar uma query SQL no Athena
# e retornar os resultados em formato de DataFrame do Pandas
def fetch_dataframe(query: str, database: str, output_location: str) -> pd.DataFrame:
    # Inicia a execução da query no Athena
    execution_id = run_query(query, database, output_location)
    # Aguarda a execução finalizar (tempo fixo de espera de 5s)
    wait_for_query_completion(execution_id)
        
    # Obtém os resultados da consulta
    results = get_query_results(execution_id)
    # A primeira linha contém os nomes das colunas
    columns = [col["VarCharValue"] for col in results[0]["Data"]]
    # As demais linhas são os dados retornados
    data = [
        [col.get("VarCharValue", "") for col in row["Data"]]
        for row in results[1:]
    ]
    
    # Constrói o DataFrame com colunas e dados
    return pd.DataFrame(data, columns=columns)

# Função que transforma uma linha de dados da tabela 'orders'
# em uma frase descritiva sobre o histórico de compra do cliente
def gerar_perfil_compra_texto(row) -> str:
    # Quebra os campos de string em listas, separadas por vírgula
    items = str(row.get('items_names', '')).split(',')
    brands = str(row.get('items_brands', '')).split(',')
    prices = str(row.get('items_prices', '')).split(',')

    # Cria frases como "comprou camiseta da marca Nike por R$120"
    compras = [
        f"comprou {item.strip()} da marca {brand.strip()} por R${price.strip()}"
        for item, brand, price in zip(items, brands, prices)
    ]
    
    # Adiciona o status de pagamento à frase
    status = f"status do pagamento: {row.get('payment_status', 'desconhecido')}"
    
    # Retorna o histórico completo da compra em texto
    return "; ".join(compras) + f" ({status})"

# Função que aplica a geração de texto para todas as linhas do DataFrame
# e retorna apenas as colunas com email e o texto gerado
def gerar_perfis_para_llm(df: pd.DataFrame) -> pd.DataFrame:
    # Aplica a função de geração de texto linha a linha
    df['perfil_compra_texto'] = df.apply(gerar_perfil_compra_texto, axis=1)
    
    # Retorna apenas as colunas relevantes para o LLM
    return df[['customer_email', 'perfil_compra_texto']]

#####################
#veio do arquivo clustering_dbscan.py
def carregar_embeddings(caminho_csv: str) -> pd.DataFrame:
    """
    Carrega o arquivo de embeddings gerado para os perfis.
    """
    return pd.read_csv(caminho_csv)

def aplicar_dbscan(df: pd.DataFrame, eps=0.5, min_samples=3) -> pd.DataFrame:
    """
    Aplica o algoritmo DBSCAN aos embeddings dos clientes.
    Retorna o DataFrame com uma nova coluna 'cluster'.
    """
    X = df.drop(columns=['customer_email'])
    X_scaled = StandardScaler().fit_transform(X)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    cluster_labels = dbscan.fit_predict(X_scaled)
    df['cluster'] = cluster_labels
    return df

def salvar_clusters(df: pd.DataFrame, caminho_saida: str) -> None:
    """
    Salva o DataFrame com os rótulos de cluster no arquivo CSV.
    """
    df.to_csv(caminho_saida, index=False)

def gerar_distribuicao_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """
    Gera a distribuição de clientes por cluster (perfil dominante).
    """
    return df.groupby('cluster')['customer_email'].count().reset_index().rename(columns={
        'customer_email': 'total_clientes'
    })

def plotar_clusters_2d(df: pd.DataFrame, output_image: str = "clusters_visualizacao.png") -> None:
    """
    Aplica PCA para redução de dimensionalidade e gera um gráfico 2D dos clusters.
    """
    X = df.drop(columns=['customer_email', 'cluster'])
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    df['pca_1'] = X_pca[:, 0]
    df['pca_2'] = X_pca[:, 1]

    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x='pca_1', y='pca_2', hue='cluster', palette='tab10')
    plt.title("Visualização 2D dos Clusters de Perfis de Compra (DBSCAN)")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(output_image)
    plt.close()
    print(f"Gráfico salvo como: {output_image}")
# FIM Veio do arquivo clustering_dbscan.py
# ######################
# ######################
# Veio do arquivo product_analysis.py
def carregar_dados_cluster(caminho_csv: str) -> pd.DataFrame:
    """
    Carrega os dados de clusters com os itens comprados.
    """
    return pd.read_csv(caminho_csv)

def contar_produtos_por_cluster(df: pd.DataFrame, campo='items_names') -> dict:
    """
    Conta a frequência de produtos comprados dentro de cada cluster.
    Retorna um dicionário: {cluster: Counter(produto -> contagem)}
    """
    cluster_produtos = {}

    for cluster, grupo in df.groupby('cluster'):
        todos_itens = []
        for linha in grupo[campo].dropna():
            itens = [i.strip().lower() for i in str(linha).split(',')]
            todos_itens.extend(itens)

        contagem = Counter(todos_itens)
        cluster_produtos[cluster] = contagem

    return cluster_produtos

def gerar_recomendacoes(cluster_produtos: dict, top_n=3) -> dict:
    """
    Gera sugestões de campanha com base nos produtos mais comprados por cluster.
    """
    sugestoes = {}
    for cluster, contagem in cluster_produtos.items():
        top_itens = [produto for produto, _ in contagem.most_common(top_n)]
        sugestoes[cluster] = f"Campanha recomendada: promoção de {', '.join(top_itens)}"
    return sugestoes
#FIM Veio do arquivo product_analysis.py
# ######################

# Função principal que orquestra a extração, transformação e exportação
def main():
    # Consulta SQL com JOIN entre orders e profiles
    query = """
        SELECT 
            orders.customer_email,
            orders.payment_status,
            orders.items_names,
            orders.items_brands,
            orders.items_prices
        FROM orders
        JOIN profiles ON orders.customer_email = profiles.email
        WHERE orders.tenant_id = 3575
          AND profiles.tenant_id = 3575
          AND orders.date BETWEEN current_timestamp - interval '30' day AND current_timestamp
    """

    # Executa a consulta e transforma em DataFrame
    df = fetch_dataframe(query, DATABASE, OUTPUT_LOCATION)
    
    # Gera os textos de perfil para cada cliente
    perfis = gerar_perfis_para_llm(df)
    
    # Salva o DataFrame original com produtos + perfil gerado
    df_completo = df.copy()
    df_completo['perfil_compra_texto'] = perfis['perfil_compra_texto']
    df_completo.to_csv("perfis_compra_llm.csv", index=False)
    print("Perfis de compra exportados para perfis_compra_llm.csv")

    #########################################
    #Veio do arquivo embedding_generation.py
    
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
    #FIM veio do arquivo embedding_generation.py
    #########################################

    #########################################
    # veio do arquivo clustering_dbscan.py
    caminho_entrada = "embeddings_perfis.csv"
    caminho_saida = "clusters_perfis.csv"
    imagem_saida = "clusters_visualizacao.png"

    # Etapas do pipeline
    embeddings = carregar_embeddings(caminho_entrada)
    clusters_df = aplicar_dbscan(embeddings, eps=0.5, min_samples=3)
    salvar_clusters(clusters_df, caminho_saida)
    distribuicao = gerar_distribuicao_clusters(clusters_df)

    print("Distribuição dos clusters:")
    print(distribuicao)

    plotar_clusters_2d(clusters_df, output_image=imagem_saida)
    #fim veio do arquivo clustering_dbscan.py
    #########################################

    #########################################
    # Veio do arquivo product_analysis.py
    caminho_csv = "clusters_perfis.csv"  # deve conter as colunas 'cluster' e 'items_names'

    # Recarrega os perfis com os campos 'items_names'
    df_produtos = pd.read_csv("perfis_compra_llm.csv")
    df_clusters = pd.read_csv("clusters_perfis.csv")

    # Junta para ter: customer_email, items_names, cluster
    df_merged = pd.merge(df_clusters, df_produtos[['customer_email', 'items_names']], on='customer_email', how='left')

    # Verifica se coluna existe
    if 'items_names' not in df_merged.columns:
        raise KeyError("A coluna 'items_names' não está disponível para gerar recomendações.")

    cluster_produtos = contar_produtos_por_cluster(df_merged, campo='items_names')
    recomendacoes = gerar_recomendacoes(cluster_produtos, top_n=3)

    for cluster, sugestao in recomendacoes.items():
        emails = df_merged[df_merged['cluster'] == cluster]['customer_email'].tolist()
        print(f"\nCluster {cluster} ({len(emails)} clientes):")
        print(f"{sugestao}")
        print(f"E-mails: {emails}")
    # FIM Veio do arquivo product_analysis.py
    #########################################
    print("Análise de perfis de compra concluída com sucesso!")

# Ponto de entrada do script
if __name__ == "__main__":
    main()
