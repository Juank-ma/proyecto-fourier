import numpy as np
import matplotlib.pyplot as plt

# Frecuencia de muestreo
fs = 1000

# Tiempo de 0 a 1 segundo
# endpoint=False asegura que el espaciado real sea exactamente 1/fs
t = np.linspace(0, 1, fs, endpoint=False)

# Señales
escalon = np.heaviside(t - 0.5, 1)
pulso = np.where((t >= 0.4) & (t <= 0.6), 1, 0)
seno = np.sin(2 * np.pi * 10 * t)

# Función para calcular la FFT
def calcular_fft(senal):
    fft = np.fft.fft(senal)
    frecuencia = np.fft.fftfreq(len(senal), 1 / fs)
    magnitud = np.abs(fft)
    fase = np.angle(fft)
    # fftshift reordena las frecuencias de forma continua (negativas -> 0 -> positivas)
    # para que la gráfica se vea correctamente centrada
    frecuencia = np.fft.fftshift(frecuencia)
    magnitud = np.fft.fftshift(magnitud)
    fase = np.fft.fftshift(fase)
    return frecuencia, magnitud, fase

# Transformadas
f1, m1, p1 = calcular_fft(escalon)
f2, m2, p2 = calcular_fft(pulso)
f3, m3, p3 = calcular_fft(seno)

# ---------------------------
# Dominio del tiempo
# ---------------------------
plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(t, escalon)
plt.title("Función Escalón")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")

plt.subplot(3, 1, 2)
plt.plot(t, pulso)
plt.title("Pulso Rectangular")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")

plt.subplot(3, 1, 3)
plt.plot(t, seno)
plt.title("Señal Senoidal")
plt.xlabel("Tiempo [s]")
plt.ylabel("Amplitud")

plt.tight_layout()
plt.show()

# ---------------------------
# Magnitud
# ---------------------------
plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(f1, m1)
plt.title("Magnitud FFT Escalón")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("|X(f)|")
plt.xlim(-50, 50)

plt.subplot(3, 1, 2)
plt.plot(f2, m2)
plt.title("Magnitud FFT Pulso")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("|X(f)|")
plt.xlim(-50, 50)

plt.subplot(3, 1, 3)
plt.plot(f3, m3)
plt.title("Magnitud FFT Senoidal")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("|X(f)|")
plt.xlim(-50, 50)

plt.tight_layout()
plt.show()

# ---------------------------
# Fase
# ---------------------------
plt.figure(figsize=(10, 8))

plt.subplot(3, 1, 1)
plt.plot(f1, p1)
plt.title("Fase Escalón")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Fase [rad]")
plt.xlim(-50, 50)

plt.subplot(3, 1, 2)
plt.plot(f2, p2)
plt.title("Fase Pulso")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Fase [rad]")
plt.xlim(-50, 50)

plt.subplot(3, 1, 3)
plt.plot(f3, p3)
plt.title("Fase Senoidal")
plt.xlabel("Frecuencia [Hz]")
plt.ylabel("Fase [rad]")
plt.xlim(-50, 50)

plt.tight_layout()
plt.show()