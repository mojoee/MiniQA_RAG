from pymilvus import MilvusClient

# # Authentication not enabled
# client = MilvusClient("http://localhost:19530")

# Authentication enabled with the root user
client = MilvusClient(
    uri="http://localhost:19530",
    token="root:Milvus",
    db_name="default"
)

# # Authentication enabled with a non-root user
# client = MilvusClient(
#     uri="http://localhost:19530",
#     token="user:password", # replace this with your token
#     db_name="default"
# )

print(client)