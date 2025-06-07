import pandas as pd
from connect_athena import run_query, get_query_results, DATABASE, OUTPUT_LOCATION, wait_for_query_completion

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
    
    # Salva os perfis em um arquivo CSV para uso posterior
    perfis.to_csv("perfis_compra_llm.csv", index=False)
    print("Perfis de compra exportados para perfis_compra_llm.csv")

# Ponto de entrada do script
if __name__ == "__main__":
    main()
