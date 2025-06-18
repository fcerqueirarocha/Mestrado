
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

def aplicar_dbscan(df: pd.DataFrame, eps=0.5, min_samples=3) -> pd.DataFrame:
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
          AND orders.date BETWEEN current_timestamp - interval '30' day AND current_timestamp
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

    # Consulta de produtos e categorias para cross-selling
    cross_selling_query = """
        SELECT 
            ordi.name AS produto, 
            ordi.brand AS marca, 
            catgi.name AS categoria
        FROM orders_items ordi
        INNER JOIN orders_items_categories catgi 
            ON ordi.product_id = catgi.product_id
            AND ordi.tenant_id = catgi.tenant_id
        WHERE ordi.tenant_id = 3575
        GROUP BY ordi.name, ordi.brand, catgi.name
        ORDER BY ordi.name;
    """
    cross_df = fetch_dataframe(cross_selling_query, DATABASE, OUTPUT_LOCATION)

    produtos_por_cluster = {cluster: [produto for produto, _ in contagem.most_common(2)]
                            for cluster, contagem in cluster_produtos.items()}
    emails_por_cluster = {cluster: df_merged[df_merged['cluster'] == cluster]['customer_email'].tolist()
                          for cluster in df_merged['cluster'].unique()}

    cross_map = {}
    for produto in cross_df['produto'].unique():
        categorias = cross_df[cross_df['produto'] == produto]['categoria'].unique()
        produtos_sugeridos = cross_df[
            cross_df['categoria'].isin(categorias) & (cross_df['produto'] != produto)
        ]['produto'].unique()
        cross_map[produto] = list(produtos_sugeridos)

    sugestoes_cross = {}
    for cluster, produtos in produtos_por_cluster.items():
        sugestoes = set()
        for produto in produtos:
            sugestoes.update(cross_map.get(produto, []))
        sugestoes_cross[cluster] = list(sugestoes)

    for cluster, sugestoes in sugestoes_cross.items():
        print(f"\nCluster {cluster} — Produtos principais: {', '.join(produtos_por_cluster[cluster])}")
        print(f"Sugestão de cross-selling: {', '.join(sugestoes) if sugestoes else 'Nenhuma sugestão encontrada'}")
        print(f"E-mails do cluster: {', '.join(emails_por_cluster[cluster])}")

if __name__ == "__main__":
    main()
