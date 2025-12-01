import numpy as np
import matplotlib.pyplot as plt

B_list = [1, 0.4, 1.5, 3]

def sinc2_generator(B, t):
    return np.sinc(B * t) ** 2

start = -3
end = 3

t_continuous = np.linspace(start, end, 10_000)

fs_list = [1.0, 1.5, 2.0, 4.0]

for B in B_list:
    x_continuous = sinc2_generator(B, t_continuous)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    axes = axes.flatten() # for easier iteration

    for idx, fs in enumerate(fs_list):
        ts = 1 / fs # distance between samples

        n_indices = np.arange(np.floor(-end / ts), np.ceil(end / ts) + 1)
        t_samples = n_indices * ts
        x_samples = sinc2_generator(B, t_samples)

        x_reconstructed = np.zeros_like(t_continuous)

        for n, x_n in zip(n_indices, x_samples):
            x_reconstructed += x_n * np.sinc((t_continuous - n * ts) / ts)

        ax = axes[idx]
        ax.plot(t_continuous, x_continuous,
                label='Original $x(t)$',
                color='black',
                linestyle='-',
                linewidth=1,
                )
        
        ax.plot(t_continuous, x_reconstructed,
                label='Reconstructed $\hat{x}(t)$',
                c='#0A0',
                linestyle='--',
                linewidth=2
                )
        
        ax.stem(t_samples, x_samples,
                label="Samples $x[n]$",
                linefmt='red',
                markerfmt='ro',
                basefmt=" ",
                )
        
        reconstruction_status = "Failed" if fs < 2 * B else "Succes"
        ax.set_title(f"fs = {fs} Hz - Reconstruction Status: {reconstruction_status}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")

        ax.legend()
        ax.grid(True)


    fig.suptitle(f"$sinc^2(Bt)$ (B = {B})", fontsize=20)
    plt.tight_layout()
    plt.savefig(fname=f"./plots/ex1_B_{B}.pdf", format="pdf")
    plt.show()