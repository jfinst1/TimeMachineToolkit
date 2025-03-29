import socket
import threading
import qutip as qt
from qutip import basis, tensor
from qutip.qip.operations import hadamard_transform, cnot
import logging
import json

class VirtualParticle:
    def __init__(self, id, state=None):
        self.id = id
        self.state = state if state is not None else basis(2, 0)

    def entangle_with(self, other):
        H = hadamard_transform(1)
        CNOT = cnot()
        self.state = H * self.state
        combined = tensor(self.state, other.state)
        entangled = CNOT * combined
        self.state = entangled.ptrace(0)
        other.state = entangled.ptrace(1)

class QuantumChat:
    def __init__(self):
        self.virtual_particles = []

    def create_virtual_particles(self, count):
        self.virtual_particles = [VirtualParticle(i) for i in range(count)]

    def entangle_virtual_particles(self):
        if len(self.virtual_particles) >= 2:
            self.virtual_particles[0].entangle_with(self.virtual_particles[1])

    def encode_message(self, message):
        qubits = [basis(2, int(bit)) for bit in message]
        return tensor(qubits)

    def decode_message(self, state):
        bits = []
        for i in range(state.dims[0][0]):
            qubit = state.ptrace(i)
            prob_0 = abs((basis(2, 0).dag() * qubit).full()[0][0])**2
            bits.append('0' if prob_0 > 0.5 else '1')
        return ''.join(bits)

    def apply_repetition_code(self, state):
        encoded = tensor(state, basis(2, 0), basis(2, 0))
        CNOT = cnot()
        encoded = CNOT * tensor(encoded.ptrace(0), encoded.ptrace(1)) * tensor(encoded.ptrace(2), basis(2, 0))
        return encoded

    def process_message(self, message):
        self.create_virtual_particles(2)
        self.entangle_virtual_particles()
        state = self.encode_message(message)
        return self.apply_repetition_code(state)

clients = []
clients_lock = threading.Lock()

def handle_client(client_socket, qc):
    while True:
        try:
            data = client_socket.recv(1024)
            if data:
                state_data = json.loads(data.decode('utf-8'))
                state = qt.Qobj(state_data['data'])
                message = qc.decode_message(state)
                logging.info(f"Processed: {message}")
                broadcast(state, client_socket)
            else:
                remove(client_socket)
                break
        except Exception as e:
            logging.error(f"Handle error: {e}")
            break

def broadcast(message, connection):
    state_data = {'data': message.full().tolist()}
    msg_bytes = json.dumps(state_data).encode('utf-8')
    with clients_lock:
        for client in clients[:]:
            if client != connection:
                try:
                    client.send(msg_bytes)
                except:
                    remove(client)

def remove(connection):
    with clients_lock:
        if connection in clients:
            clients.remove(connection)

def main():
    logging.basicConfig(level=logging.INFO)
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('0.0.0.0', 9999))
    server.listen(10)
    logging.info("Server on port 9999")

    while True:
        client_socket, addr = server.accept()
        with clients_lock:
            clients.append(client_socket)
        logging.info(f"Connected: {addr}")
        qc = QuantumChat()
        threading.Thread(target=handle_client, args=(client_socket, qc)).start()

if __name__ == "__main__":
    main()