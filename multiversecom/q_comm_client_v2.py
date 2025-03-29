import qutip as qt
from qutip import basis, tensor
from qutip.qip.operations import hadamard_transform, cnot
import socket
import threading
import sys
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
        self.messages = []

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
        for i in range(state.dims[0][0]):  # Number of qubits
            qubit = state.ptrace(i)
            prob_0 = abs((basis(2, 0).dag() * qubit).full()[0][0])**2
            bits.append('0' if prob_0 > 0.5 else '1')
        return ''.join(bits)

    def apply_repetition_code(self, state):
        # Simple 3-qubit repetition code
        encoded = tensor(state, basis(2, 0), basis(2, 0))
        CNOT = cnot()
        encoded = CNOT * tensor(encoded.ptrace(0), encoded.ptrace(1)) * tensor(encoded.ptrace(2), basis(2, 0))
        return encoded

    def process_message(self, message):
        self.create_virtual_particles(2)
        self.entangle_virtual_particles()
        state = self.encode_message(message)
        state = self.apply_repetition_code(state)
        return state

    def display_log(self):
        logging.info(f"Messages: {self.messages}")

def send_network_message(sock, state):
    state_data = {'data': state.full().tolist()}
    sock.sendall(json.dumps(state_data).encode('utf-8'))

def receive_network_message(sock, qc):
    while True:
        try:
            data = sock.recv(1024)
            if data:
                state_data = json.loads(data.decode('utf-8'))
                state = qt.Qobj(state_data['data'])
                message = qc.decode_message(state)
                qc.messages.append(f"Received: {message}")
                qc.display_log()
        except Exception as e:
            logging.error(f"Receive error: {e}")
            break

def cli_interface(qc, sock):
    while True:
        message = input("Enter binary message (e.g., 1010): ")
        if message == "exit":
            break
        state = qc.process_message(message)
        send_network_message(sock, state)
        qc.messages.append(f"Sent: {message}")
        qc.display_log()

def main():
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) != 3:
        print("Usage: python client.py <host> <port>")
        sys.exit(1)

    host, port = sys.argv[1], int(sys.argv[2])
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((host, port))

    qc = QuantumChat()
    recv_thread = threading.Thread(target=receive_network_message, args=(sock, qc))
    recv_thread.daemon = True
    recv_thread.start()

    cli_interface(qc, sock)
    sock.close()

if __name__ == "__main__":
    main()