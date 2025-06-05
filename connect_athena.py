import boto3

# Credentials are loaded from aws_config.py. Replace the placeholders in that
# file with your actual AWS credentials. Do **not** commit real credentials to
# version control.
import aws_config

athena = boto3.client(
    "athena",
    aws_access_key_id=aws_config.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=aws_config.AWS_SECRET_ACCESS_KEY,
    region_name=aws_config.AWS_REGION,
)

def run_query(query, database, output_location):
    response = athena.start_query_execution(
        QueryString=query,
        QueryExecutionContext={"Database": database},
        ResultConfiguration={"OutputLocation": output_location},
    )
    execution_id = response["QueryExecutionId"]
    return execution_id


def test_connection():
    """Attempt a simple Athena API call to verify connectivity."""
    try:
        athena.list_data_catalogs()
        return True
    except Exception as exc:
        print(f"Connection test failed: {exc}")
        return False

# Example usage
if __name__ == "__main__":
    if test_connection():
        query = "SELECT * FROM my_table LIMIT 10;"
        database = "my_database"
        output_location = "s3://my-query-results/"
        exec_id = run_query(query, database, output_location)
        print(f"Started Athena query: {exec_id}")

