import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

# inspired by the work of Walter Russell and his wave cycle theories.

class Wave:
    def __init__(self, amplitude=1, frequency=1, phase=0, angle=0, interference_intensity=1.0):
        self.amplitude = amplitude
        self.frequency = frequency
        self.phase = phase
        self.angle = angle
        self.interference_intensity = interference_intensity

    def compute(self, t, frame):
        cathode_wave = self.amplitude * np.sin(self.frequency * t + frame * 0.1)
        anode_wave = -self.amplitude * np.sin(self.frequency * t + frame * 0.1)
        interference_base = self.amplitude * np.sin(self.frequency * t + frame * 0.1 + self.phase) * self.interference_intensity
        x_prime = interference_base * np.cos(self.angle)
        z_prime = interference_base * np.sin(self.angle)
        cathode_effect = (1 - np.abs(x_prime) / self.amplitude) * cathode_wave
        anode_effect = (1 - np.abs(z_prime) / self.amplitude) * anode_wave
        return cathode_effect, anode_effect, x_prime, z_prime

class WaveSimulation:
    T = np.linspace(-4 * np.pi, 4 * np.pi, 50)
    NUM_PARTICLES = 5000
    
    def __init__(self):
        self.fig, self.ax = self._setup_plot()
        self.particle_positions = np.linspace(-4 * np.pi, 4 * np.pi, self.NUM_PARTICLES)
        self.particles1 = self.ax.scatter(self.particle_positions, [0] * self.NUM_PARTICLES, [0] * self.NUM_PARTICLES, c='blue')
        self.particles2 = self.ax.scatter(self.particle_positions, [0] * self.NUM_PARTICLES, [0] * self.NUM_PARTICLES, c='red')
        self.particle_trails = []
        self.wave = Wave()
        self._setup_sliders()
        self._setup_animation()

    def _setup_plot(self):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(bottom=0.3)
        return fig, ax

    def _setup_sliders(self):
        axfreq = plt.axes([0.1, 0.01, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        axamp = plt.axes([0.1, 0.06, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        axphase = plt.axes([0.1, 0.11, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        axangle = plt.axes([0.1, 0.16, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        axintensity = plt.axes([0.1, 0.21, 0.65, 0.03], facecolor='lightgoldenrodyellow')
        axcoherence = plt.axes([0.1, 0.26, 0.65, 0.03], facecolor='lightgoldenrodyellow')

        self.sfreq = Slider(axfreq, 'Freq', 0.1, 2.0, valinit=1)
        self.samp = Slider(axamp, 'Amp', 0.1, 2.0, valinit=1)
        self.sphase = Slider(axphase, 'Phase', 0, 2 * np.pi, valinit=0)
        self.sangle = Slider(axangle, 'Angle', 0, 2 * np.pi, valinit=0)
        self.sintensity = Slider(axintensity, 'Intensity', 0, 1.0, valinit=0)
        self.scoherence = Slider(axcoherence, 'Coherence', 0, 1.0, valinit=1.0)

    def _setup_animation(self):
        self.ani = FuncAnimation(self.fig, self._update, frames=np.linspace(0, 2*np.pi, 10000), repeat=True, fargs=(self.particles1, self.particles2, self.particle_trails, self.sfreq, self.samp, self.sphase, self.sangle, self.sintensity, self.scoherence))

    def _update(self, frame, particles1, particles2, particle_trails, sfreq, samp, sphase, sangle, sintensity, scoherence):
        self.ax.clear()

        self.wave.amplitude = samp.val
        self.wave.frequency = sfreq.val
        self.wave.phase = sphase.val
        self.wave.angle = sangle.val
        self.wave.interference_intensity = sintensity.val

        cathode_wave, anode_wave, interference_x, interference_z = self.wave.compute(self.T, frame)

        coherence_alpha = scoherence.val

        self.ax.plot(self.T, cathode_wave, np.zeros_like(self.T), color='blue', label='Cathode Wave', alpha=coherence_alpha)
        self.ax.plot(self.T, np.zeros_like(self.T), anode_wave, color='red', label='Anode Wave', alpha=coherence_alpha)
        self.ax.plot(self.T, interference_x, interference_z, color='green', label='Interference', alpha=coherence_alpha)

        particle_cathode_y = [np.interp((len(self.T)/2) + self.wave.frequency * pos, np.arange(len(self.T)), cathode_wave) for pos in self.particle_positions]
        particle_anode_z = [np.interp((len(self.T)/2) + self.wave.frequency * pos, np.arange(len(self.T)), anode_wave) for pos in self.particle_positions]
        particle_interference_y = [np.interp((len(self.T)/2) + self.wave.frequency * pos, np.arange(len(self.T)), interference_x) for pos in self.particle_positions]
        particle_interference_z = [np.interp((len(self.T)/2) + self.wave.frequency * pos, np.arange(len(self.T)), interference_z) for pos in self.particle_positions]

        threshold = 1
        colors1 = ['yellow' if abs(particle_cathode_y[i] - np.interp((len(self.T)/2) + self.wave.frequency * self.particle_positions[i], np.arange(len(self.T)), cathode_wave)) < threshold else 'black' for i in range(self.NUM_PARTICLES)]
        colors2 = ['yellow' if abs(particle_anode_z[i] - np.interp((len(self.T)/2) + self.wave.frequency * self.particle_positions[i], np.arange(len(self.T)), anode_wave)) < threshold else 'black' for i in range(self.NUM_PARTICLES)]

        particles1.set_color(colors1)
        particles2.set_color(colors2)

        avg_particle_y = [(particle_cathode_y[i] + particle_interference_y[i]) / 2 for i in range(self.NUM_PARTICLES)]
        avg_particle_z = [(particle_anode_z[i] + particle_interference_z[i]) / 2 for i in range(self.NUM_PARTICLES)]

        particles1._offsets3d = (self.particle_positions, avg_particle_y, avg_particle_z)
        particles2._offsets3d = (self.particle_positions, avg_particle_y, avg_particle_z)

        if len(particle_trails) > 1:
            particle_trails.pop(0)
        particle_trails.append((self.particle_positions.copy(), avg_particle_y.copy(), avg_particle_z.copy()))
        for trail in particle_trails:
            self.ax.plot(trail[0], trail[1], trail[2], c='black', alpha=0.5)

        legend_elements = [Line2D([0], [0], color='blue', label='Cathode Wave'),
                           Line2D([0], [0], color='red', label='Anode Wave'),
                           Line2D([0], [0], color='green', label='Interference'),
                           Line2D([0], [0], color='black', label='Particle Stream')]

        self.ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(-0.7, 1))

    def run(self):
        plt.show()

if __name__ == "__main__":
    sim = WaveSimulation()
    sim.run()
