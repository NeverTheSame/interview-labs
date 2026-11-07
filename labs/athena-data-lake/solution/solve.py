"""Reference solution for the data lake lab."""

from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data" / "orders"


def solve(aws) -> None:
    for bucket in ("lake", "lake-athena-results"):
        aws.s3.create_bucket(Bucket=bucket)
    for path in DATA.glob("dt=*/*.csv"):
        aws.s3.upload_file(str(path), "lake", f"raw/orders/{path.parent.name}/{path.name}")

    aws.glue.create_database(DatabaseInput={"Name": "shop"})
    aws.glue.create_table(DatabaseName="shop", TableInput={
        "Name": "orders", "TableType": "EXTERNAL_TABLE",
        "Parameters": {"classification": "csv", "skip.header.line.count": "1"},
        "PartitionKeys": [{"Name": "dt", "Type": "string"}],
        "StorageDescriptor": {
            "Location": "s3://lake/raw/orders/",
            "Columns": [{"Name": "order_id", "Type": "string"}, {"Name": "customer", "Type": "string"},
                        {"Name": "amount", "Type": "double"}, {"Name": "country", "Type": "string"}],
            "InputFormat": "org.apache.hadoop.mapred.TextInputFormat",
            "OutputFormat": "org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat",
            "SerdeInfo": {"SerializationLibrary": "org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe",
                          "Parameters": {"field.delim": ",", "skip.header.line.count": "1"}},
        },
    })

    aws.athena.create_named_query(
        Name="revenue_by_country", Database="shop",
        QueryString="SELECT country, round(sum(amount), 2) AS revenue FROM orders GROUP BY country ORDER BY revenue DESC")
    aws.athena.create_named_query(
        Name="orders_on_day", Database="shop",
        QueryString="SELECT count(*) AS orders FROM orders WHERE dt = '2026-11-02'")
