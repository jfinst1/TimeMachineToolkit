import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import tkinter as tk
from tkinter import ttk, messagebox
import json
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Global debug flag (set to False to suppress debug prints)
DEBUG = True

# =============================================================================
# Symbolic Definitions (for quantum operators, gates, and Hamiltonians)
# =============================================================================

# Define symbols for symbolic computations (used in parts of the code)
θ, φ, λ, t, α, β, γ, δ, ε, ζ, η, μ, ν, ξ, ρ, σ, τ = sp.symbols('θ φ λ t α β γ δ ε ζ η μ ν ξ ρ σ τ')

def σ_rotation(θ, φ, λ):
    """
    Construct a sigma rotation matrix for a given angle and phase.
    """
    return sp.Matrix([
        [sp.cos(θ / 2), -sp.exp(sp.I * λ) * sp.sin(θ / 2)],
        [sp.exp(sp.I * φ) * sp.sin(θ / 2), sp.exp(sp.I * (φ + λ)) * sp.cos(θ / 2)]
    ])

# Basic matrices
I = sp.eye(2)

X = sp.Matrix([
    [0, 1],
    [1, 0]
])

Y = sp.Matrix([
    [0, -sp.I],
    [sp.I, 0]
])

Z = sp.Matrix([
    [1, 0],
    [0, -1]
])

# Define Hamiltonian H and its time evolution operator U(t) = exp(-i*H*t)
H = X + Y + Z
U_t = sp.exp(-sp.I * H * t)
if DEBUG:
    sp.pretty_print(U_t, num_columns=300)

# =============================================================================
# Rotation Matrices for Single-Qubit Operations
# =============================================================================

def rotation_matrix(axis, angle):
    """
    Return the rotation matrix about the given axis by the specified angle.
    Supported axes: 'X', 'Y', 'Z'.
    """
    if axis == 'X':
        return sp.Matrix([
            [sp.cos(angle / 2), -sp.I * sp.sin(angle / 2)],
            [-sp.I * sp.sin(angle / 2), sp.cos(angle / 2)]
        ])
    elif axis == 'Y':
        return sp.Matrix([
            [sp.cos(angle / 2), -sp.sin(angle / 2)],
            [sp.sin(angle / 2), sp.cos(angle / 2)]
        ])
    elif axis == 'Z':
        return sp.Matrix([
            [sp.exp(-sp.I * angle / 2), 0],
            [0, sp.exp(sp.I * angle / 2)]
        ])
    else:
        raise ValueError("Axis must be 'X', 'Y', or 'Z'")

# Define various rotation matrices using the rotation_matrix function
RX = rotation_matrix('X', θ)
RY = rotation_matrix('Y', φ)
RZ = rotation_matrix('Z', λ)

RX2 = rotation_matrix('X', δ)
RY2 = rotation_matrix('Y', ε)
RZ2 = rotation_matrix('Z', ζ)

RX3 = rotation_matrix('X', η)
RY3 = rotation_matrix('Y', μ)
RZ3 = rotation_matrix('Z', ν)

RX4 = rotation_matrix('X', ξ)
RY4 = rotation_matrix('Y', ρ)
RZ4 = rotation_matrix('Z', σ)

# =============================================================================
# Multi-Qubit Gate Construction
# =============================================================================

# Define the CNOT gate (4x4 matrix for 2 qubits)
CNOT = sp.Matrix([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 0, 1],
    [0, 0, 1, 0]
])

def kronecker_chain(matrices):
    """
    Compute the Kronecker product of a list of matrices sequentially.
    """
    result = matrices[0]
    for matrix in matrices[1:]:
        result = sp.kronecker_product(result, matrix)
    return result

unitary_4q = kronecker_chain([sp.eye(4), CNOT, RZ4, RY4, RX4, RZ3, RY3, RX3, RZ2, RY2, RX2, RZ, RY, RX])
if DEBUG:
    sp.pretty_print(unitary_4q, num_columns=300)

# Define a noise matrix
def noise_matrix(size, ν):
    """
    Construct a diagonal noise matrix of given size with phase noise parameter ν.
    """
    return sp.diag(*[sp.exp(-sp.I * ν)] * size)

noise = noise_matrix(16, ν)
unitary_with_noise = noise * unitary_4q
if DEBUG:
    sp.pretty_print(unitary_with_noise, num_columns=300)

# Define a more complex Hamiltonian and its time evolution operator
H_complex = α * X + β * Y + γ * Z + δ * (X * Y) + ε * (Y * Z) + ζ * (X * Z) + η * (X * X) + μ * (Y * Y) + ν * (Z * Z)
time_evolution_complex = sp.exp(-sp.I * H_complex * t)
if DEBUG:
    sp.pretty_print(time_evolution_complex, num_columns=300)

# =============================================================================
# Numeric Transformation Functions (using NumPy for performance)
# =============================================================================

def temporal_displacement_numeric(state, params):
    """
    Apply temporal displacement transformation on the state.
    
    Parameters:
      state: NumPy array representing the state vector (4,).
      params: List of 8 numerical parameters.
              - First 4 parameters: angles for phase rotations.
              - Next 4 parameters: noise phase offsets.
              
    Returns:
      Transformed state as a NumPy array.
    """
    T = np.diag(np.cos(np.array(params[:4])) - 1j * np.sin(np.array(params[:4])))
    T_noise = np.diag(np.exp(-1j * np.array(params[4:8])))
    return T_noise @ T @ state

def dimensional_transition_numeric(state, params):
    """
    Apply dimensional transition transformation on the state.
    
    Parameters:
      state: NumPy array representing the state vector (4,).
      params: List of 10 numerical parameters.
              - First 4 parameters: global phase factors.
              - params[4] and params[5]: rotation angles for two 2x2 blocks.
              - params[6:10]: noise phase offsets.
              
    Returns:
      Transformed state as a NumPy array.
    """
    D1 = np.diag(np.exp(-1j * np.array(params[:4])))
    angle1 = params[4] / 2
    angle2 = params[5] / 2
    rot1 = np.array([[np.cos(angle1), -np.sin(angle1)],
                     [np.sin(angle1),  np.cos(angle1)]])
    rot2 = np.array([[np.cos(angle2), -np.sin(angle2)],
                     [np.sin(angle2),  np.cos(angle2)]])
    D_rot = np.block([
         [rot1, np.zeros((2, 2))],
         [np.zeros((2, 2)), rot2]
    ])
    D = D1 @ D_rot
    D_noise = np.diag(np.exp(-1j * np.array(params[6:10])))
    return D_noise @ D @ state

def cubical_transition_numeric(state, params):
    """
    Apply cubical transition transformation on the state.
    This transformation raises the diagonal and rotation parts to the 3rd power.
    
    Parameters:
      state: NumPy array representing the state vector (4,).
      params: List of 10 numerical parameters.
              - First 4 parameters: global phase factors.
              - params[4] and params[5]: rotation angles for two 2x2 blocks.
              - params[6:10]: noise phase offsets.
              
    Returns:
      Transformed state as a NumPy array.
    """
    # Raise the diagonal phase factors by multiplying the angles by 3.
    D1 = np.diag(np.exp(-1j * 3 * np.array(params[:4])))
    angle1 = 3 * params[4] / 2
    angle2 = 3 * params[5] / 2
    rot1 = np.array([[np.cos(angle1), -np.sin(angle1)],
                     [np.sin(angle1),  np.cos(angle1)]])
    rot2 = np.array([[np.cos(angle2), -np.sin(angle2)],
                     [np.sin(angle2),  np.cos(angle2)]])
    D_rot = np.block([
         [rot1, np.zeros((2, 2))],
         [np.zeros((2, 2)), rot2]
    ])
    C = D1 @ D_rot
    C_noise = np.diag(np.exp(-1j * np.array(params[6:10])))
    return C_noise @ C @ state

def multiverse_navigation_numeric(state, params_4th, params_5th, params_6th):
    """
    Sequentially apply temporal displacement, dimensional transition, and cubical transition.
    
    Parameters:
      state: NumPy array representing the initial state vector.
      params_4th: List of 8 numerical parameters for temporal displacement.
      params_5th: List of 10 numerical parameters for dimensional transition.
      params_6th: List of 10 numerical parameters for cubical transition.
      
    Returns:
      Transformed state as a NumPy array.
    """
    state_4th = temporal_displacement_numeric(state, params_4th)
    state_5th = dimensional_transition_numeric(state_4th, params_5th)
    state_6th = cubical_transition_numeric(state_5th, params_6th)
    return state_6th

# =============================================================================
# Visualization Functions
# =============================================================================

def plot_state_vector(state_vector, title='State Vector'):
    """
    Plot the real and imaginary parts of the state vector.
    
    Parameters:
      state_vector: List or NumPy array of complex numbers.
      title: Title of the plot.
    """
    try:
        # If state_vector is a Sympy Matrix, convert to a NumPy array.
        state_vector = np.array(state_vector, dtype=complex).flatten()
    except Exception:
        state_vector = np.array([sp.N(s) for s in state_vector], dtype=complex).flatten()
    real_parts = np.real(state_vector)
    imag_parts = np.imag(state_vector)
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    fig.suptitle(title)
    
    axs[0].bar(range(len(real_parts)), real_parts, color='b', alpha=0.6)
    axs[0].set_ylabel('Real Part')
    
    axs[1].bar(range(len(imag_parts)), imag_parts, color='r', alpha=0.6)
    axs[1].set_ylabel('Imaginary Part')
    axs[1].set_xlabel('Index')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def animate_state_transition(initial_state, params_4th, params_5th, params_6th, frames=100, status_label=None, progress_bar=None):
    """
    Animate the state transition using multiverse navigation.
    This function creates a new Tkinter window with an embedded Matplotlib canvas.
    
    Parameters:
      initial_state: NumPy array representing the initial state vector.
      params_4th: List of 8 numerical parameters for temporal displacement.
      params_5th: List of 10 numerical parameters for dimensional transition.
      params_6th: List of 10 numerical parameters for cubical transition.
      frames: Number of frames in the animation.
      status_label: Tkinter label widget for status updates.
      progress_bar: Tkinter progress bar widget for progress updates.
    """
    # Create a new Toplevel window for animation
    anim_window = tk.Toplevel()
    anim_window.title("State Vector Transition Animation")
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    fig.suptitle('State Vector Transition')
    
    canvas = FigureCanvasTkAgg(fig, master=anim_window)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
    
    def update(frame):
        scaled_params_4th = [p * frame/frames for p in params_4th]
        scaled_params_5th = [p * frame/frames for p in params_5th]
        scaled_params_6th = [p * frame/frames for p in params_6th]
        state_t = multiverse_navigation_numeric(initial_state, scaled_params_4th, scaled_params_5th, scaled_params_6th)
        real_parts = np.real(state_t)
        imag_parts = np.imag(state_t)
    
        for ax in axs:
            ax.cla()
    
        axs[0].bar(range(len(real_parts)), real_parts, color='b', alpha=0.6)
        axs[0].set_ylabel('Real Part')
    
        axs[1].bar(range(len(imag_parts)), imag_parts, color='r', alpha=0.6)
        axs[1].set_ylabel('Imaginary Part')
        axs[1].set_xlabel('Index')
    
        if status_label:
            status_label.config(text=f"Animating frame {frame + 1}/{frames}")
        if progress_bar:
            progress_bar['value'] = (frame + 1) / frames * 100
            status_label.update_idletasks()
    
        canvas.draw_idle()
    
    ani = FuncAnimation(fig, update, frames=frames, repeat=False)
    # Note: We do not call plt.show() since the canvas is embedded in the Tkinter window.

# =============================================================================
# Main Program and GUI Application
# =============================================================================

class QuantumApp:
    """
    GUI application for Quantum State Navigation using Tkinter.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Quantum State Navigator")

        # Define Tkinter variables for parameters and state vector
        self.params_4th = [tk.DoubleVar(value=1.0) for _ in range(8)]
        self.params_5th = [tk.DoubleVar(value=1.0) for _ in range(10)]
        self.params_6th = [tk.DoubleVar(value=1.0) for _ in range(10)]
        self.state_vector = [tk.DoubleVar(value=1.0) if i == 0 else tk.DoubleVar(value=0.0) for i in range(4)]

        frame = ttk.Frame(root, padding="10")
        frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # State vector initialization
        ttk.Label(frame, text="State Vector Initialization").grid(row=0, column=0, columnspan=4)
        for i, val in enumerate(self.state_vector):
            ttk.Entry(frame, textvariable=val, width=5).grid(row=1, column=i)

        # 4th Dimension Parameters
        ttk.Label(frame, text="4th Dimension Parameters").grid(row=2, column=0, columnspan=8)
        for i, param in enumerate(self.params_4th):
            ttk.Entry(frame, textvariable=param, width=5).grid(row=3, column=i)

        # 5th Dimension Parameters
        ttk.Label(frame, text="5th Dimension Parameters").grid(row=4, column=0, columnspan=10)
        for i, param in enumerate(self.params_5th):
            ttk.Entry(frame, textvariable=param, width=5).grid(row=5, column=i)

        # 6th Dimension Parameters
        ttk.Label(frame, text="6th Dimension Parameters").grid(row=6, column=0, columnspan=10)
        for i, param in enumerate(self.params_6th):
            ttk.Entry(frame, textvariable=param, width=5).grid(row=7, column=i)

        self.status_label = ttk.Label(frame, text="Ready")
        self.status_label.grid(row=8, column=0, columnspan=10)

        self.progress_bar = ttk.Progressbar(frame, orient='horizontal', mode='determinate')
        self.progress_bar.grid(row=9, column=0, columnspan=10, sticky=(tk.W, tk.E))

        ttk.Button(frame, text="Animate", command=self.animate).grid(row=10, column=0, columnspan=5)
        ttk.Button(frame, text="Save Config", command=self.save_config).grid(row=10, column=5, columnspan=5)
        ttk.Button(frame, text="Load Config", command=self.load_config).grid(row=11, column=0, columnspan=5)
        ttk.Button(frame, text="Reset", command=self.reset).grid(row=11, column=5, columnspan=5)

    def animate(self):
        """
        Gather parameters and initial state, then animate the state transition.
        """
        self.status_label.config(text="Starting animation...")
        self.root.update_idletasks()
        params_4th = [param.get() for param in self.params_4th]
        params_5th = [param.get() for param in self.params_5th]
        params_6th = [param.get() for param in self.params_6th]
        # Convert state vector to a NumPy array for numerical computations
        initial_state = np.array([val.get() for val in self.state_vector], dtype=complex)
        self.status_label.config(text="Animating...")
        self.progress_bar['value'] = 0
        animate_state_transition(initial_state, params_4th, params_5th, params_6th,
                                 status_label=self.status_label, progress_bar=self.progress_bar)
        self.status_label.config(text="Animation complete.")
        self.progress_bar['value'] = 100

    def save_config(self):
        """
        Save the current configuration (parameters and state vector) to a JSON file.
        """
        params_4th = [param.get() for param in self.params_4th]
        params_5th = [param.get() for param in self.params_5th]
        params_6th = [param.get() for param in self.params_6th]
        state_vector = [val.get() for val in self.state_vector]
        config = {
            'params_4th': params_4th,
            'params_5th': params_5th,
            'params_6th': params_6th,
            'state_vector': state_vector
        }
        try:
            with open('config.json', 'w') as f:
                json.dump(config, f)
            messagebox.showinfo("Save Config", "Configuration saved successfully!")
        except Exception as e:
            messagebox.showerror("Save Config", f"Failed to save configuration: {e}")

    def load_config(self):
        """
        Load the configuration from a JSON file and update the GUI.
        """
        try:
            with open('config.json', 'r') as f:
                config = json.load(f)
            for i, val in enumerate(config['params_4th']):
                self.params_4th[i].set(val)
            for i, val in enumerate(config['params_5th']):
                self.params_5th[i].set(val)
            for i, val in enumerate(config['params_6th']):
                self.params_6th[i].set(val)
            for i, val in enumerate(config['state_vector']):
                self.state_vector[i].set(val)
            messagebox.showinfo("Load Config", "Configuration loaded successfully!")
        except Exception as e:
            messagebox.showerror("Load Config", f"Failed to load configuration: {e}")

    def reset(self):
        """
        Reset all parameters and the state vector to their default values.
        """
        for param in self.params_4th + self.params_5th + self.params_6th:
            param.set(1.0)
        for i, val in enumerate(self.state_vector):
            val.set(1.0 if i == 0 else 0.0)
        self.status_label.config(text="Ready")
        self.progress_bar['value'] = 0

# =============================================================================
# Run the Application
# =============================================================================

if __name__ == "__main__":
    root = tk.Tk()
    app = QuantumApp(root)
    root.mainloop()