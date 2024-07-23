import socket

# Define server address and port
server_address = ('localhost', 12345)

# Create a TCP socket (SOCK_STREAM means a TCP socket)
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    # Connect to the server
    client_socket.connect(server_address)

    # Send data
    message = "Hello, server! I have the following query: Tell me about Priority for client orders: order handling and recording."
    client_socket.sendall(message.encode())

    # Receive response from the server
    data = client_socket.recv(1024)
    print(f"Received from server: {data.decode()}")

finally:
    # Close the socket
    client_socket.close()