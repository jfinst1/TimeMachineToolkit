import numpy as np
from numpy import pi
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
import matplotlib.pyplot as plt

# v2.1.5 - New Improved Non-Zero Edition: Electric Boogaloo

# I was dreaming when I wrote this, 
# forgive me if it goes astray..

class Workspace:
    def __init__(self, size):
        self.data = np.zeros(size)

    def get(self):
        return self.data

# But when I woke up this morning,
# I could have sworn it was judgement day..

def get_random_qubits(qubits_count):

    q = QuantumRegister(qubits_count, 'q')
    c = ClassicalRegister(qubits_count, 'c')
    circ = QuantumCircuit(q, c)

    circ.h(q[0])
    circ.h(q[1])
    circ.h(q[2])
    circ.h(q[3])
    circ.h(q[4])
    circ.s(q[1])
    circ.sdg(q[3])
    circ.ry(pi / 2, q[4])
    circ.tdg(q[2])
    circ.cx(q[0], q[1])
    circ.cx(q[3], q[4])
    circ.rx(pi / 2, q[2])
    circ.ccx(q[2], q[4], q[3])
    circ.cx(q[0], q[1])
    circ.ccx(q[0], q[2], q[1])
    circ.cx(q[3], q[4])
    circ.z(q[2])
    circ.cx(q[4], q[3]).c_if(c, 0)
    circ.cx(q[0], q[1])
    circ.measure(q, c)

    simulator = Aer.get_backend('qasm_simulator')
    job = execute(circ, simulator, shots=5)
    result = job.result()
    counts = result.get_counts(circ)

    for key in counts:
        random_number = int(key, 5)

    return random_number

# … The sky was all purple there were people runnin' everywhere..
# Tryin' to run from the destruction, you know I didn't even care..

def get_random_qubits_dict(qubits_count):
    random_number = get_random_qubits(qubits_count)
    return {'x0': random_number, 'x1': random_number, 't': random_number, 'Phi': random_number}

wsqi40 = Workspace(40)
wso9 = Workspace(10)

RV = {'EQ_Phi_1': 0, 'EQ_Phi_2': 0, 'EQ_Phi_3': 0, 'EQ_Phi_4': 0}

# … Say, say, 2000-00, party over..

def perform_calculation():
    N1_t, N1_x0, N1_x1, N1_Phi = [get_random_qubits(5) for _ in range(4)]
    N2_t, N2_x0, N2_x1, N2_Phi = [get_random_qubits(5) for _ in range(4)]
    N3_t, N3_x0, N3_x1, N3_Phi = [get_random_qubits(5) for _ in range(4)]
    N4_t, N4_x0, N4_x1, N4_Phi = [get_random_qubits(5) for _ in range(4)]
    b_1, d_1_0, d_1_1, c_1, Kappat, Kappa = [get_random_qubits(5) for _ in range(6)]

    qi = wsqi40.get()
    o = wso9.get()

    qi[5] = -N1_t + N3_t
    qi[7] = -N1_x1 + N4_x1
    qi[4] = -N1_x1 + N3_x1
    qi[8] = -N1_t + N4_t
    qi[9] = -(qi[5] * qi[7]) + qi[4] * qi[8]
    qi[0] = -N1_x0 + N2_x0
    qi[3] = -N1_x0 + N3_x0
    qi[2] = -N1_t + N2_t
    qi[1] = -N1_x1 + N2_x1
    qi[10] = qi[2] * qi[7] - qi[1] * qi[8]
    qi[6] = -N1_x0 + N4_x0
    qi[11] = -(qi[2] * qi[4]) + qi[1] * qi[8]
    qi[18] = qi[0] * qi[9] + qi[3] * qi[10] * qi[6] * qi[11]
    qi[19] = 1 / qi[18]
    qi[20] = qi[9] * qi[19]
    qi[12] = qi[5] * qi[6] - qi[3] * qi[8]
    qi[21] = qi[12] * qi[19]
    qi[15] = -(qi[4] * qi[6]) + qi[3] * qi[7]
    qi[22] = qi[15] * qi[19]
    qi[23] = qi[10] * qi[19]
    qi[13] = -(qi[2] * qi[6]) + qi[0] * qi[8]
    qi[24] = qi[13] * qi[19]
    qi[16] = qi[1] * qi[6] - qi[0] * qi[7]
    qi[25] = qi[16] * qi[19]
    qi[26] = qi[11] * qi[19]
    qi[14] = qi[2] * qi[3] - qi[0] * qi[5]
    qi[27] = qi[14] * qi[19]
    qi[17] = -(qi[1] * qi[3]) + qi[0] * qi[4]
    qi[28] = qi[17] * qi[19]
    qi[29] = -qi[20] - qi[23] - qi[26]
    qi[30] = -qi[21] - qi[24] - qi[27]
    qi[31] = -qi[22] - qi[25] - qi[28]
    qi[32] = N2_Phi * qi[22] + N3_Phi * qi[25] + N4_Phi * qi[28] + N1_Phi * qi[31]
    qi[33] = N2_Phi * qi[20] + N3_Phi * qi[23] + N4_Phi * qi[26] + N1_Phi * qi[29]
    qi[34] = N2_Phi * qi[21] + N3_Phi * qi[24] + N4_Phi * qi[27] + N1_Phi * qi[30]
    qi[35] = 0.166666666666667 * np.abs(qi[18])
    qi[36] = qi[35] * (qi[16] + -qi[8])
    qi[37] = 0.250000000000000 * qi[35]
    qi[38] = 0.100000000000000 * qi[35]
    qi[39] = 0.005000000000000 * qi[35]

    o[0] = b_1 * qi[32]
    o[1] = d_1_0 * qi[33]
    o[2] = d_1_1 * qi[34]
    o[3] = o[0] + o[1] + o[2]
    o[4] = o[3] + qi[37]
    o[5] = N2_Phi * c_1
    o[6] = N3_Phi * c_1
    o[7] = N4_Phi * c_1
    o[8] = N1_Phi * c_1
    o[9] = o[4] * (o[0] + o[1] + o[2])
    RV['EQ_Phi_1'] = o[4] + (Kappat * qi[31] * qi[32] + Kappa * qi[29] * qi[33] + Kappa * qi[30] * qi[34]) * qi[36] + N1_Phi * c_1 * qi[38] + (-o[5:9][2:] + -o[5:9][:1]).sum() * qi[36]
    RV['EQ_Phi_2'] = o[1] + (Kappat * qi[22] * qi[32] + Kappa * qi[20] * qi[33] + Kappa * qi[21] * qi[34]) * qi[36] + N2_Phi * c_1 * qi[38] + (-o[5:8][2:] + -o[5:8][:1]).sum() * qi[36]
    RV['EQ_Phi_3'] = o[2] + (Kappat * qi[25] * qi[32] + Kappa * qi[23] * qi[33] + Kappa * qi[24] * qi[34]) * qi[39] + N3_Phi * c_1 * qi[38] + (o[5:8][2:] + o[5:8][:1]).sum() * qi[39]
    RV['EQ_Phi_4'] = o[3] + (Kappat * qi[28] * qi[32] + Kappa * qi[26] * qi[33] + Kappa * qi[27] * qi[34]) * qi[39] + N4_Phi * c_1 * qi[38] + (o[5:9][2:] + o[5:9][:1]).sum() * qi[39]

    return N1_t, N1_x0, N2_t, N2_x0, N3_t, N3_x0, N4_t, N4_x0, N1_Phi, N2_Phi, N3_Phi, N4_Phi

# Oops, out of time!

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

plt.ion()

while True:
    N1_t, N1_x0, N2_t, N2_x0, N3_t, N3_x0, N4_t, N4_x0, N1_Phi, N2_Phi, N3_Phi, N4_Phi = perform_calculation()
#   x = [RV['EQ_Phi_1'], RV['EQ_Phi_2'], RV['EQ_Phi_3'], RV['EQ_Phi_4']]
    x = [N1_Phi, N2_Phi, N3_Phi, N4_Phi]
    y = [N1_t, N2_t, N3_t, N4_t]
    z = [N1_x0, N2_x0, N3_x0, N4_x0]

# So tonight I'm gonna party like it's 1999!

    ax.scatter(x, y, z)

    plt.draw()
    plt.pause(0.05)