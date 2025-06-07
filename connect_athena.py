import os
import boto3
import time

#constants for AWS Athena
DATABASE = os.getenv("ATHENA_DATABASE", "data-platform-gold")  # Default database
OUTPUT_LOCATION = os.getenv("ATHENA_OUTPUT_LOCATION", "s3://eventbroadcaster-athena-queries-dev/")  # Default S3 output location

athena = boto3.client(
    "athena",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION"),
)

def run_query(query, database, output_location):
    response = athena.start_query_execution(
        QueryString=query,
        QueryExecutionContext={"Database": database},
        ResultConfiguration={"OutputLocation": output_location},
    )
    execution_id = response["QueryExecutionId"]
    return execution_id

def get_query_results(execution_id):
    """Fetch the results of the Athena query."""
    results = athena.get_query_results(QueryExecutionId=execution_id)
    rows = results["ResultSet"]["Rows"]
    return rows

def test_connection():
    """Attempt a simple Athena API call to verify connectivity."""
    try:
        athena.list_data_catalogs()
        return True
    except Exception as exc:
        print(f"Connection test failed: {exc}")
        return False


def wait_for_query_completion(execution_id, delay=2, timeout_seconds=60):
    """
    Aguarda a execução da query no Athena até 1 minuto.
    Se a query falhar ou ultrapassar o tempo, levanta uma exceção com a razão.
    """
    start_time = time.time()

    while True:
        response = athena.get_query_execution(QueryExecutionId=execution_id)
        status = response['QueryExecution']['Status']
        state = status['State']

        if state == 'SUCCEEDED':
            return state
        elif state in ['FAILED', 'CANCELLED']:
            reason = status.get('StateChangeReason', 'Motivo não informado')
            raise Exception(
                f"Query {execution_id} falhou com estado: {state}\nMotivo: {reason}"
            )
        
        # Verificar se ultrapassou o tempo limite de 1 minuto
        elapsed = time.time() - start_time
        if elapsed > timeout_seconds:
            raise TimeoutError(
                f"A execução da query {execution_id} excedeu o limite de {timeout_seconds} segundos."
            )
        
        time.sleep(delay)