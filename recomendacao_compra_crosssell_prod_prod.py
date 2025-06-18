
import pandas as pd
import time
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from connect_athena import run_query, get_query_results, DATABASE, OUTPUT_LOCATION, wait_for_query_completion

def fetch_dataframe(query: str, database: str, output_location: str) -> pd.DataFrame:
    execution_id = run_query(query, database, output_location)
    wait_for_query_completion(execution_id)
    results = get_query_results(execution_id)
    columns = [col["VarCharValue"] for col in results[0]["Data"]]
    data = [[col.get("VarCharValue", "") for col in row["Data"]] for row in results[1:]]
    return pd.DataFrame(data, columns=columns)

def gerar_perfil_compra_texto(row) -> str:
    items = str(row.get('items_names', '')).split(',')
    brands = str(row.get('items_brands', '')).split(',')
    prices = str(row.get('items_prices', '')).split(',')
    compras = [f"comprou {item.strip()} da marca {brand.strip()} por R${price.strip()}"
               for item, brand, price in zip(items, brands, prices)]
    status = f"status do pagamento: {row.get('payment_status', 'desconhecido')}"
    return "; ".join(compras) + f" ({status})"

def gerar_perfis_para_llm(df: pd.DataFrame) -> pd.DataFrame:
    df['perfil_compra_texto'] = df.apply(gerar_perfil_compra_texto, axis=1)
    return df[['customer_email', 'perfil_compra_texto']]

def aplicar_dbscan(df: pd.DataFrame, eps=0.5, min_samples=1) -> pd.DataFrame:
    X = df.drop(columns=['customer_email'])
    X_scaled = StandardScaler().fit_transform(X)
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    df['cluster'] = dbscan.fit_predict(X_scaled)
    return df

def contar_produtos_por_cluster(df: pd.DataFrame, campo='items_names') -> dict:
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
    sugestoes = {}
    for cluster, contagem in cluster_produtos.items():
        top_itens = [produto for produto, _ in contagem.most_common(top_n)]
        sugestoes[cluster] = f"Campanha recomendada: promoção de {', '.join(top_itens)}"
    return sugestoes

def main():
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
          AND orders.date BETWEEN current_timestamp - interval '90' day AND current_timestamp
    """
    df = fetch_dataframe(query, DATABASE, OUTPUT_LOCATION)
    perfis = gerar_perfis_para_llm(df)
    df_completo = df.copy()
    df_completo['perfil_compra_texto'] = perfis['perfil_compra_texto']
    df_completo.to_csv("perfis_compra_llm.csv", index=False)

    model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = model.encode(df_completo['perfil_compra_texto'].tolist(), show_progress_bar=True)
    embeddings_df = pd.DataFrame(embeddings)
    embeddings_df.insert(0, 'customer_email', df_completo['customer_email'])
    embeddings_df.to_csv("embeddings_perfis.csv", index=False)

    clusters_df = aplicar_dbscan(embeddings_df)
    print(clusters_df['cluster'].value_counts())
    clusters_df.to_csv("clusters_perfis.csv", index=False)

    df_clusters = clusters_df
    df_produtos = df_completo
    df_merged = pd.merge(df_clusters, df_produtos[['customer_email', 'items_names']], on='customer_email', how='left')

    if 'items_names' not in df_merged.columns:
        raise KeyError("A coluna 'items_names' não está disponível para gerar recomendações.")

    cluster_produtos = contar_produtos_por_cluster(df_merged, campo='items_names')
    recomendacoes = gerar_recomendacoes(cluster_produtos, top_n=3)

    for cluster, sugestao in recomendacoes.items():
        emails = df_merged[df_merged['cluster'] == cluster]['customer_email'].tolist()
        print(f"\nCluster {cluster} — {sugestao}")
        print(f"E-mails: {', '.join(emails)}")

    # 1. Consulta SQL para identificar coocorrência de produtos
    coocorrencia_query = """
        SELECT 
            a.name AS produto_a,
            b.name AS produto_b,
            COUNT(*) AS vezes_juntos
        FROM orders_items a
        JOIN orders_items b 
          ON a.order_id = b.order_id 
         AND a.product_id != b.product_id
         AND a.tenant_id = b.tenant_id
        WHERE a.tenant_id = 3575
        GROUP BY a.name, b.name
        HAVING COUNT(*) > 1
        ORDER BY vezes_juntos DESC
    """
    coocorrencia_df = fetch_dataframe(coocorrencia_query, DATABASE, OUTPUT_LOCATION)

    # 2. Construção do mapa produto -> produtos frequentemente comprados juntos
    coocorrencia_map = {}
    for _, row in coocorrencia_df.iterrows():
        produto_a = row['produto_a']
        produto_b = row['produto_b']
        if produto_a not in coocorrencia_map:
            coocorrencia_map[produto_a] = set()
        coocorrencia_map[produto_a].add(produto_b)

    # 3. Preparar listas auxiliares para cada cluster
    produtos_por_cluster = {cluster: [produto for produto, _ in contagem.most_common(2)]
                            for cluster, contagem in cluster_produtos.items()}
    emails_por_cluster = {cluster: df_merged[df_merged['cluster'] == cluster]['customer_email'].tolist()
                          for cluster in df_merged['cluster'].unique()}

    # 4. Gerar sugestões de cross-sell por coocorrência
    sugestoes_cross_coocorrencia = {}
    for cluster, produtos in produtos_por_cluster.items():
        sugestoes = set()
        for produto in produtos:
            sugestoes.update(coocorrencia_map.get(produto, []))
        sugestoes_cross_coocorrencia[cluster] = list(sugestoes)

    # 5. Exibir resultados
    for cluster, sugestoes in sugestoes_cross_coocorrencia.items():
        print(f"\nCluster {cluster} — Produtos principais: {', '.join(produtos_por_cluster[cluster])}")
        print(f"Recomendações por coocorrência: {', '.join(sugestoes) if sugestoes else 'Nenhuma sugestão encontrada'}")
        print(f"E-mails do cluster: {', '.join(emails_por_cluster[cluster])}")


if __name__ == "__main__":
    main()
