import numpy as np
import matplotlib.pyplot as plt

def calc_spectrum(Y):
    return 20 * np.log10(np.abs(Y) + 1e-10)

N = 256

t = np.linspace(0, 1, N)
n1, n2 = np.meshgrid(t, t)
#-------------------------------------------------------
x1 = np.sin(2 * np.pi * n1 + 3 * np.pi * n2)
Y1 = np.fft.fft2(x1)
spectrum1 = calc_spectrum(Y1)
#-------------------------------------------------------
x2 = np.sin(4 * np.pi * n1) + np.cos(6 * np.pi * n2)
Y2 = np.fft.fft2(x2)
spectrum2 = calc_spectrum(Y2)
#-------------------------------------------------------
Y3 = np.zeros((N, N), dtype=complex)
Y3[0, 10] = 1
Y3[0, N - 10] = 1
x3 = np.real(np.fft.ifft2(Y3))
spectrum3 = calc_spectrum(Y3)
#-------------------------------------------------------
Y4 = np.zeros((N, N), dtype=complex)
Y4[5, 0] = 1
Y4[N - 5, 0] = 1
x4 = np.real(np.fft.ifft2(Y4))
spectrum4 = calc_spectrum(Y4)
#-------------------------------------------------------
Y5 = np.zeros((N, N), dtype=complex)
Y5[5, 5] = 1
Y5[N - 5, N - 5] = 1
x5 = np.real(np.fft.ifft2(Y5))
spectrum5 = calc_spectrum(Y5)
#-------------------------------------------------------

def plot_func(fig_name, x, spectrum, sup_title):
    plt.figure(figsize=(12, 6))
    plt.suptitle(sup_title)

    plt.subplot(1, 2, 1)
    plt.imshow(x, cmap=plt.cm.gray)
    plt.title("The image - original signal")
    plt.xlabel('n1')
    plt.ylabel('n2')

    plt.subplot(1, 2, 2)
    plt.imshow(spectrum)
    plt.title('Fourier Spectrum of the image')
    plt.colorbar()

    plt.savefig(fname=f"./plots/{fig_name}.pdf", format="pdf")
    plt.show()

plot_func("a", x1, spectrum1, "2D Signal -> $x_{n_1,n_2} = \sin(2\pi n_1 + 3\pi n_2)$")
plot_func("b", x2, spectrum2, "2D Signal -> $x_{n_1,n_2} = \sin(4\pi n_1) + \cos(6\pi n_2)$")
plot_func("c", x3, spectrum3, "2D Signal from Spectrum -> $Y_{0,5} = Y_{0,N-5} = 1$")
plot_func("d", x4, spectrum4, "2D Signal from Spectrum -> $Y_{5,0} = Y_{N-5,0} = 1$")
plot_func("e", x5, spectrum5, "2D Signal from Spectrum -> $Y_{5,5} = Y_{N-5,N-5} = 1$")
