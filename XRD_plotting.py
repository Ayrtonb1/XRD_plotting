import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from PyQt5.QtWidgets import QApplication, QFileDialog

# === File Selection (supports multiple files reliably) ===
def select_files():
    app = QApplication(sys.argv)
    files, _ = QFileDialog.getOpenFileNames(
        None,
        "Select XRD data files",
        "",
        "XRD files (*.xye *.xy *.dat *.txt);;All files (*)",
        options=QFileDialog.Option.DontUseNativeDialog
    )
    if not files:
        raise ValueError("No files selected.")
    return files

# === File loader ===
def load_xrd_file(filepath):
    data = np.loadtxt(filepath)
    theta, intensity = data[:, 0], data[:, 1]
    return theta, intensity

# === Professional color palette ===
def get_colors(n):
    cmap = get_cmap("tab10")
    return [cmap(i % cmap.N) for i in range(n)]

# === FWHM calculation using linear interpolation ===
def fwhm_from_peak(theta, intensity):
    peak_idx = np.argmax(intensity)
    peak_intensity = intensity[peak_idx]
    half_max = peak_intensity / 2

    # Search left
    left_idx = peak_idx
    while left_idx > 0 and intensity[left_idx] > half_max:
        left_idx -= 1
    left_theta = np.interp(half_max, [intensity[left_idx], intensity[left_idx+1]],
                           [theta[left_idx], theta[left_idx+1]])

    # Search right
    right_idx = peak_idx
    while right_idx < len(intensity)-1 and intensity[right_idx] > half_max:
        right_idx += 1
    right_theta = np.interp(half_max, [intensity[right_idx-1], intensity[right_idx]],
                            [theta[right_idx-1], theta[right_idx]])

    return right_theta - left_theta

# === Scherrer crystallite size calculation ===
def scherrer_size(theta, intensity, wavelength=0.15406, K=0.9):
    peak_idx = np.argmax(intensity)
    peak_2theta = theta[peak_idx]
    FWHM_deg = fwhm_from_peak(theta, intensity)
    beta = np.deg2rad(FWHM_deg)
    theta_rad = np.deg2rad(peak_2theta / 2)
    D = (K * wavelength) / (beta * np.cos(theta_rad))
    return D, peak_2theta, FWHM_deg

# === Plotting functions ===
def plot_stacked_xrd_publication(processed, peak_info=None, save_path=None):
    plt.figure(figsize=(8, 6))
    colors = get_colors(len(processed))

    min_theta = min(np.min(theta) for _, theta, _ in processed)
    max_theta = max(np.max(theta) for _, theta, _ in processed)
    
    offset = 0
    for i, (label, theta, intensity) in enumerate(processed):
        plt.plot(theta, intensity + offset, label=label, color=colors[i], lw=1.5)
        if peak_info and label in peak_info:
            peak_2theta = peak_info[label]['peak']
            plt.axvline(peak_2theta, linestyle="--", color=colors[i], alpha=0.7)
        offset += np.max(intensity) * 1.1

    plt.xlabel(r"$2\theta$ (°)", fontsize=14)
    plt.ylabel("Intensity + offset", fontsize=14)
    plt.title("Stacked XRD Spectra", fontsize=16, weight="bold")
    plt.xlim(min_theta, max_theta)
    plt.legend(frameon=False, fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_unstacked_xrd_publication(processed, peak_info=None, save_path=None):
    plt.figure(figsize=(8, 6))
    colors = get_colors(len(processed))

    min_theta = min(np.min(theta) for _, theta, _ in processed)
    max_theta = max(np.max(theta) for _, theta, _ in processed)

    for i, (label, theta, intensity) in enumerate(processed):
        plt.plot(theta, intensity, label=label, color=colors[i], lw=1.5)
        if peak_info and label in peak_info:
            peak_2theta = peak_info[label]['peak']
            plt.axvline(peak_2theta, linestyle="--", color=colors[i], alpha=0.7)

    plt.xlabel(r"$2\theta$ (°)", fontsize=14)
    plt.ylabel("Normalized Intensity (a.u.)", fontsize=14)
    plt.title("Unstacked XRD Spectra", fontsize=16, weight="bold")
    plt.xlim(min_theta, max_theta)
    plt.legend(frameon=False, fontsize=9)
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_3d_xrd_publication(processed, save_path=None):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    colors = get_colors(len(processed))

    min_theta = min(np.min(theta) for _, theta, _ in processed)
    max_theta = max(np.max(theta) for _, theta, _ in processed)

    for i, (label, theta, intensity) in enumerate(processed, start=1):  # ✅ start index at 1
        ax.plot(theta, np.full_like(theta, i), intensity, label=label, color=colors[i-1])

    ax.set_xlabel(r"$2\theta$ (°)", fontsize=12)
    ax.set_ylabel("Sample index", fontsize=12)
    ax.set_zlabel("Intensity (a.u.)", fontsize=12)
    ax.set_title("3D XRD Spectra", fontsize=14, weight="bold")
    ax.set_xlim(min_theta, max_theta)
    ax.set_ylim(1, len(processed))  # ✅ y-axis starts at 1 now
    ax.view_init(elev=30, azim=-60)

    # Add legend
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


# === Main runner ===
if __name__ == "__main__":
    output_folder = os.path.join(os.getcwd(), "XRD_Output")
    os.makedirs(output_folder, exist_ok=True)

    # Load XRD files
    files = select_files()
    datasets = []
    for f in files:
        filename = os.path.basename(f)
        theta, intensity = load_xrd_file(f)
        intensity /= np.max(intensity)
        datasets.append((filename, theta, intensity))  # use filename as label

    # Restrict to overlapping region
    min_theta = max(np.min(ds[1]) for ds in datasets)
    max_theta = min(np.max(ds[1]) for ds in datasets)
    processed = []
    for label, theta, intensity in datasets:
        mask = (theta >= min_theta) & (theta <= max_theta)
        processed.append((label, theta[mask], intensity[mask]))

    # Calculate crystallite size for console output
    peak_info = {}
    for label, theta, intensity in processed:
        D, peak, FWHM = scherrer_size(theta, intensity)
        peak_info[label] = {'D': D, 'peak': peak, 'FWHM': FWHM}
        print(f"{label}: Most intense peak at 2θ={peak:.2f}°, FWHM={FWHM:.3f}°, Crystallite size ≈ {D:.2f} nm")

    # Plot
    plot_stacked_xrd_publication(processed, peak_info, save_path=os.path.join(output_folder, "stacked_xrd.png"))
    plot_unstacked_xrd_publication(processed, peak_info, save_path=os.path.join(output_folder, "unstacked_xrd.png"))
    plot_3d_xrd_publication(processed, save_path=os.path.join(output_folder, "3d_xrd.png"))
