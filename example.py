import matplotlib.pyplot as plt
import numpy as np

# =========================
# CHANGE VALUES HERE
# =========================

formats = ["PDF", "DOCX", "XLSX", "PPTX", "Image", "CSV/JSON"]

latency = [2.4, 2.1, 2.8, 3.2, 4.1, 2.6]

accuracy = [88, 91, 86, 83, 79, 87]

# =========================
# GRAPH SETTINGS
# =========================

x = np.arange(len(formats))
width = 0.35

fig, ax1 = plt.subplots(figsize=(10, 6))

# Latency bars
bars1 = ax1.bar(
    x - width / 2,
    latency,
    width,
    label="Latency (s)"
)

ax1.set_xlabel("File Format")
ax1.set_ylabel("Latency (seconds)")
ax1.set_ylim(0, 10)

ax1.set_xticks(x)
ax1.set_xticklabels(formats, rotation=45)

# Accuracy axis
ax2 = ax1.twinx()

bars2 = ax2.bar(
    x + width / 2,
    accuracy,
    width,
    label="Accuracy (%)"
)

ax2.set_ylabel("Accuracy (%)")
ax2.set_ylim(60, 100)

# =========================
# DISPLAY VALUES
# =========================

for bar in bars1:
    height = bar.get_height()

    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        height + 0.15,
        f"{height:.1f}",
        ha="center"
    )

for bar in bars2:
    height = bar.get_height()

    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        height + 1,
        f"{height:.0f}",
        ha="center"
    )

# =========================
# TITLE AND LEGEND
# =========================

plt.title("System Performance (Latency & Accuracy)")

ax1.legend(
    bars1,
    ["Latency (s)"],
    loc="upper left"
)

ax2.legend(
    bars2,
    ["Accuracy (%)"],
    loc="upper right"
)

plt.tight_layout()

# Save image
plt.savefig("system_performance.png", dpi=300)

plt.show()