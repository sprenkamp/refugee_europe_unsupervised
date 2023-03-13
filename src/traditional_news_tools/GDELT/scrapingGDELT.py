from google.cloud import bigquery

client = bigquery.Client()

query = """
    SELECT *
    FROM `gdelt-bq.gdeltv2.events`
    WHERE ActionGeo_CountryCode = 'US'
    AND SQLDATE BETWEEN '20230201' AND '20230228'
"""

query_job = client.query(query)

results = query_job.result()

for row in results:
    print(row)