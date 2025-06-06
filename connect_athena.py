import os
import boto3

# Credentials are loaded from aws_config.py. Replace the placeholders in that
# file with your actual AWS credentials. Do **not** commit real credentials to
# version control.

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

# Example usage
if __name__ == "__main__":
    if test_connection():
        emails = ['rafaella.duque@hotmail.com', 'fortunee.levi@gmail.com']
        email_list = "', '".join(emails)  # concatena para formar parte do SQL
        
        query = f"""
                select 
                    ord.order_id, 
                    ord.customer_email || ' - ' || ord.order_id as email_com_id,
                    status, 
                    coupons, 
                    total, 
                    date
                from orders ord
                where ord.tenant_id=3575
                and ord.date >= current_timestamp - interval '7' day
                and ord.date <= current_timestamp
                and ord.customer_email in ('{email_list}');
                """
        # Define the database and output location
        database = "data-platform-gold"
        output_location = "s3://eventbroadcaster-athena-queries-dev/"
        exec_id = run_query(query, database, output_location)
        print(f"Started Athena query: {exec_id}")

         # Wait for the query to complete (optional: add retry logic)
        import time
        while True:
            status = athena.get_query_execution(QueryExecutionId=exec_id)["QueryExecution"]["Status"]["State"]
            if status in ["SUCCEEDED", "FAILED", "CANCELLED"]:
                break
            time.sleep(2)

        if status == "SUCCEEDED":
            rows = get_query_results(exec_id)
            # A primeira linha contém os nomes das colunas
            column_names = [col["VarCharValue"] for col in rows[0]["Data"]]
            
            # Itera sobre as linhas restantes (os dados)
            for row in rows[1:]:
                values = [col.get("VarCharValue", "NULL") for col in row["Data"]]
                formatted_row = ", ".join(f"{column}: {value}" for column, value in zip(column_names, values))
                print(formatted_row)
        else:
            response = athena.get_query_execution(QueryExecutionId=exec_id)
            print(response)  # Log completo da execução
            status = response["QueryExecution"]["Status"]["State"]
            reason = response["QueryExecution"]["Status"].get("StateChangeReason", "No reason provided")
            print(f"Query failed with status: {status}, reason: {reason}")
            print(f"Query failed with status: {status}")

