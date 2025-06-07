import os
import boto3

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
