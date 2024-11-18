import numpy as np
import matplotlib.pyplot as plt
from IPython.display import Math
import sympy as sym

# Matriks Transisi: Kinerja Mahasiswa (A: Kinerja Tinggi, B: Kinerja Rendah)
M = np.array([[0.8, 0.2],  # A ke A (0.8), A ke B (0.2)
              [0.4, 0.6]])  # B ke A (0.4), B ke B (0.6)

# Kondisi awal: 80% Mahasiswa berkinerja tinggi, 20% berkinerja rendah
H1 = np.array([0.8, 0.2])

# Fungsi untuk menghitung distribusi kinerja di masa depan
def hitung_distribusi(hari, H1, M, threshold=1e-5):
    distribusi = [H1]  # Daftar untuk menyimpan distribusi setiap hari
    for _ in range(hari):
        H1 = np.dot(H1, M)  # Perkalian matriks untuk perhitungan distribusi masa depan
        distribusi.append(H1)
        
        # Cek jika perubahan antara hari sebelumnya dan hari sekarang sangat kecil (stabil)
        if np.all(np.abs(distribusi[-1] - distribusi[-2]) < threshold):
            stabil_hari = len(distribusi) - 1
            print(f"Titik Stabil dicapai pada Hari {stabil_hari}")
            break
    else:
        stabil_hari = hari  # Jika tidak stabil dalam batas hari yang diberikan
    return distribusi, stabil_hari

# Menyimulasikan hingga hari ke-20 untuk memastikan stabilitas
hari = 11
distribusi, stabil_hari = hitung_distribusi(hari, H1, M)

# Menampilkan distribusi kinerja mahasiswa setiap hari
for i, distribusi_hari in enumerate(distribusi):
    print(f"Hari {i}: {distribusi_hari}")

# Grafik distribusi kinerja mahasiswa (A dan B) dari hari ke-0 hingga ke-stabil
hari_label = [f"Hari {i}" for i in range(stabil_hari + 1)]  # Membatasi label hingga hari stabil
kinerja_tinggi = [distribusi_hari[0] for distribusi_hari in distribusi[:stabil_hari + 1]]  # Persentase kinerja tinggi
kinerja_rendah = [distribusi_hari[1] for distribusi_hari in distribusi[:stabil_hari + 1]]  # Persentase kinerja rendah

# Membuat grafik untuk memvisualisasikan distribusi kinerja
plt.figure(figsize=(10, 6))
plt.plot(hari_label, kinerja_tinggi, label="Kinerja Tinggi (A)", marker='o', color='b', linestyle='-', linewidth=2)
plt.plot(hari_label, kinerja_rendah, label="Kinerja Rendah (B)", marker='o', color='r', linestyle='--', linewidth=2)

# Menambahkan label dan judul
plt.xlabel('Hari')
plt.ylabel('Probabilitas')
plt.title('Simulasi Perubahan Kinerja Mahasiswa dalam Sistem Pembelajaran')
plt.legend()

# Menambahkan grid
plt.grid(True)

# Menampilkan grafik
plt.tight_layout()
plt.show()

# Menampilkan Steady State (Titik Keseimbangan)
# Steady state: D = D * M, sehingga D1 + D2 = 1 (jumlah total probabilitas)
I = np.eye(2)  # Matriks Identitas
M_minus_I = M - I  # Matriks transisi dikurangi identitas

# Sistem persamaan linear D * (M - I) = [0, 0], dengan D1 + D2 = 1
A = np.vstack([M_minus_I.T, np.array([1, 1])])  # Menambahkan persamaan D1 + D2 = 1
b = np.array([0, 0, 1])

# Menyelesaikan sistem persamaan linier untuk steady state D
steady_state = np.linalg.lstsq(A, b, rcond=None)[0]

# Menampilkan hasil steady state
print(f"\nTitik Keseimbangan (Steady State): {steady_state}")
if steady_state[0] > steady_state[1]:
    print("Mahasiswa dengan kinerja tinggi (A) dominan.")
else:
    print("Mahasiswa dengan kinerja rendah (B) dominan.")