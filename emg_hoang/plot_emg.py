import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_DIR = Path(__file__).parent
FS = 1000  # Hz

files = sorted(DATA_DIR.glob("*.csv"))
n = len(files)

fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=False)
fig.suptitle("EMG Sessions — Raw & Filtered", fontsize=13, fontweight='bold')

for ax, f in zip(axes, files):
    df = pd.read_csv(f)
    t  = df.index / FS  # giây

    ax.plot(t, df["Raw_ADC"] - 2048, color='orange', lw=0.6, alpha=0.6, label="Raw (centered)")
    ax.plot(t, df["Filtered_EMG"],   color='steelblue', lw=1.2, label="Filtered")
    ax.set_title(f.stem, fontsize=10, fontweight='bold')
    ax.set_ylabel("ADC / mV")
    ax.axhline(0, color='gray', lw=0.5, ls='--')
    ax.legend(loc='upper right', fontsize=8)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel("Thời gian (s)")
plt.tight_layout()
plt.savefig(DATA_DIR / "emg_plot.png", dpi=150, bbox_inches='tight')
plt.show()
