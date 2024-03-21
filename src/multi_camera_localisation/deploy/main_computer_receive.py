import zmq
import pickle

# Define the address to bind the server socket
server_address = "tcp://*:5555"

# Create ZeroMQ context and socket
context = zmq.Context()
socket = context.socket(zmq.REP)

# Bind the socket to the address
socket.bind(server_address)

# Receive the serialized NumPy array
serialized_pts = socket.recv()

# Deserialize the NumPy array
pts = pickle.loads(serialized_pts)

# Process the received data (pts)
print(pts)

# Close the socket and context
socket.close()
context.term()
