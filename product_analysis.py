import pandas as pd
from collections import Counter

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

# Exemplo de uso
if __name__ == "__main__":
    caminho_csv = "clusters_perfis.csv"  # deve conter as colunas 'cluster' e 'items_names'

    df = carregar_dados_cluster(caminho_csv)
    cluster_produtos = contar_produtos_por_cluster(df, campo='items_names')
    recomendacoes = gerar_recomendacoes(cluster_produtos, top_n=3)

    for cluster, sugestao in recomendacoes.items():
        print(f"Cluster {cluster}: {sugestao}")
