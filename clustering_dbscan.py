import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

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

# Exemplo de uso
if __name__ == "__main__":
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
