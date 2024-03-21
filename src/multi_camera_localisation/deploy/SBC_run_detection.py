import zmq
import numpy as np
import pickle

# Define the address of the server
server_address = "tcp://localhost:5555"

# Create ZeroMQ context and socket
context = zmq.Context()
socket = context.socket(zmq.REQ)

# Connect to the server
socket.connect(server_address)

# Sample NumPy array (replace this with your actual data)
pts = np.random.rand(10, 3, 2)

# Serialize the NumPy array
serialized_pts = pickle.dumps(pts)

# Send the serialized NumPy array
socket.send(serialized_pts)

# Close the socket and context
socket.close()
context.term()
