import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# 1. Definición de la señal de entrada
fs = 1000.0  # Frecuencia de muestreo (Hz)
t = np.arange(0, 1.0, 1.0/fs)  # Vector de tiempo de 1 segundo

# Señal compuesta: tono de 5 Hz (baja) + tono de 150 Hz (alta)
f_baja = 5.0
f_alta = 150.0
senal_pura = np.sin(2 * np.pi * f_baja * t) + 0.5 * np.sin(2 * np.pi * f_alta * t)

# Añadir ruido blanco aleatorio
np.random.seed(42)
ruido = 0.5 * np.random.normal(size=t.shape)
senal_ruidosa = senal_pura + ruido

# 2. Diseño del filtro (Ejemplo: Filtro Pasa Bajos Butterworth IIR)
# Queremos dejar pasar frecuencias bajas (ej. < 50 Hz) y eliminar el ruido/frecuencias altas
f_corte = 50.0  # Frecuencia de corte en Hz
orden = 4
nyq = 0.5 * fs
normal_cutoff = f_corte / nyq

# Obtener coeficientes del filtro Butterworth pasa-bajos
b, a = signal.butter(orden, normal_cutoff, btype='low', analog=False)

# 3. Aplicación del filtro a la señal
senal_filtrada = signal.lfilter(b, a, senal_ruidosa)

# 4. Visualización de los resultados
plt.figure(figsize=(12, 10))

# Gráfica 1: Señal en el dominio del tiempo
plt.subplot(3, 1, 1)
plt.plot(t[:200], senal_ruidosa[:200], label='Señal Ruidosa / Original', color='red', alpha=0.7)
plt.plot(t[:200], senal_filtrada[:200], label='Señal Filtrada (Pasa Bajos)', color='blue', linewidth=2)
plt.title('Comparación en el Dominio del Tiempo')
plt.xlabel('Tiempo [s]')
plt.ylabel('Amplitud')
plt.legend()
plt.grid(True)

# Gráfica 2: Respuesta en frecuencia del filtro (Diagrama de Bode / Magnitud)
w, h = signal.freqz(b, a, worN=800)
plt.subplot(3, 1, 2)
plt.plot(0.5 * fs * w / np.pi, np.abs(h), 'b', linewidth=2)
plt.title('Respuesta en Frecuencia del Filtro Pasa Bajos (Butterworth)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Ganancia')
plt.grid(True)

# Gráfica 3: Densidad espectral de potencia (Dominio de la frecuencia de las señales)
plt.subplot(3, 1, 3)
f_s, Pxx_ruido = signal.welch(senal_ruidosa, fs, nperseg=1024)
f_s, Pxx_filtrada = signal.welch(senal_filtrada, fs, nperseg=1024)
plt.semilogy(f_s, Pxx_ruido, label='Espectro Señal Ruidosa', color='red', alpha=0.7)
plt.semilogy(f_s, Pxx_filtrada, label='Espectro Señal Filtrada', color='blue', linewidth=2)
plt.title('Densidad Espectral de Potencia (Dominio de la Frecuencia)')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('PSD [V**2/Hz]')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()