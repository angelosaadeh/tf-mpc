"""All the functions to set up a network between the players, they are common to all users"""
import socket


def serve(ip, port):
    """Create a server"""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((ip, port))
    print('A server is created')
    return server


def connect(ip, port):
    """Connect to a server"""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.connect((ip, port))
    print('You are connected to a server')
    return server


def accept_connections(server):
    """Accept a connection"""
    server.listen()
    print('The server is listening to accept connections')
    client, _ = server.accept()
    print('A client just got connected')
    return client
