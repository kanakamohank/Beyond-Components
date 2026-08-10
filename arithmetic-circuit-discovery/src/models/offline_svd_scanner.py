import torch
import numpy as np
import glob
from transformer_lens import HookedTransformer
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
def map_pair_to_frequencies(Vh, number_embeddings, valid_nums, k1, k2,
                            model_name="", layer=0, head=0, output_dir="."):
    """
    Runs Fourier analysis on the specific (k1, k2) singular vector pair
    identified by the geometric scan as forming a circular structure.

    Validates that both dimensions share the same dominant period and
    exhibit the ~pi/2 phase offset expected of a sine/cosine pair.
    Plots a side-by-side FFT spectrum for visual confirmation.
    """
    import numpy as np
    import matplotlib.pyplot as plt

    print(f"\n🔍 Fourier Analysis of Identified Circular Pair (Dims {k1}, {k2}):")

    def _analyze_dim(Vh, number_embeddings, valid_nums, dim_idx):
        signal = (number_embeddings @ Vh[dim_idx]).cpu().numpy()
        signal = signal - np.mean(signal)

        fft_values = np.fft.rfft(signal)
        fft_mags = np.abs(fft_values)
        fft_phases = np.angle(fft_values)

        freqs = np.fft.rfftfreq(len(valid_nums), d=1.0)

        pos_mask = freqs > 0.01
        clean_freqs = freqs[pos_mask]
        clean_mags = fft_mags[pos_mask]
        clean_phases = fft_phases[pos_mask]

        if len(clean_mags) == 0:
            return None

        best_idx = np.argmax(clean_mags)
        best_freq = clean_freqs[best_idx]
        best_period = 1.0 / best_freq
        best_phase = clean_phases[best_idx]
        signal_strength = clean_mags[best_idx] / np.sum(clean_mags) * 100

        return {
            "period": best_period,
            "frequency": best_freq,
            "phase": best_phase,
            "strength": signal_strength,
            "periods": 1.0 / clean_freqs,
            "magnitudes": clean_mags,
            "best_idx": best_idx,
        }

    result_k1 = _analyze_dim(Vh, number_embeddings, valid_nums, k1)
    result_k2 = _analyze_dim(Vh, number_embeddings, valid_nums, k2)

    if result_k1 is None or result_k2 is None:
        print("   ⚠️ Could not extract frequency for one or both dimensions.")
        return

    print(f"   Dim {k1:2d}: Period = {result_k1['period']:5.1f} | Strength = {result_k1['strength']:4.1f}% | Phase = {result_k1['phase']:+.3f} rad")
    print(f"   Dim {k2:2d}: Period = {result_k2['period']:5.1f} | Strength = {result_k2['strength']:4.1f}% | Phase = {result_k2['phase']:+.3f} rad")

    period_ratio = result_k1['period'] / result_k2['period']
    periods_match = 0.9 < period_ratio < 1.1

    phase_delta = abs(result_k1['phase'] - result_k2['phase'])
    phase_delta = min(phase_delta, 2 * np.pi - phase_delta)
    ideal_offset = np.pi / 2
    phase_close = abs(phase_delta - ideal_offset) < 0.3

    print(f"\n   📐 Period ratio (ideal ~1.0):  {period_ratio:.3f}  {'✅' if periods_match else '❌'}")
    print(f"   📐 Phase offset (ideal ~π/2):  {phase_delta:.3f} rad ({np.degrees(phase_delta):.1f}°)  {'✅' if phase_close else '❌'}")

    if periods_match and phase_close:
        avg_period = (result_k1['period'] + result_k2['period']) / 2
        print(f"\n   ✅ Confirmed sine/cosine pair! Shared period ≈ {avg_period:.1f} numbers per cycle.")
    elif periods_match:
        print(f"\n   ⚠️ Periods match but phase offset deviates from π/2 — may be a rotated basis.")
    else:
        print(f"\n   ❌ Periods diverge — this pair may not encode a single clean frequency.")

    # ==========================================
    # Plot side-by-side FFT spectra for the pair
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, dim_idx, result, color in [
        (ax1, k1, result_k1, 'teal'),
        (ax2, k2, result_k2, 'darkorange'),
    ]:
        periods = result['periods']
        mags = result['magnitudes']
        bi = result['best_idx']

        markerline, stemlines, baseline = ax.stem(periods, mags, basefmt=" ")
        plt.setp(stemlines, 'linewidth', 2, 'color', color)
        plt.setp(markerline, 'markersize', 8, 'color', 'darkblue')

        ax.annotate(
            f'$T \\approx {result["period"]:.1f}$\n({result["strength"]:.0f}%)',
            xy=(periods[bi], mags[bi]),
            xytext=(periods[bi] + 5, mags[bi] * 0.9),
            arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=6),
            fontweight='bold', color='red',
        )
        ax.set_xlabel('Period $T$ (Numbers per cycle)', fontweight='bold')
        ax.set_title(f'SVD Dim {dim_idx}  (φ = {result["phase"]:+.2f} rad)', fontweight='bold')
        ax.grid(True, alpha=0.3)

    ax1.set_ylabel('Fourier Magnitude', fontweight='bold')

    verdict = "✅ Sine/Cosine Pair" if (periods_match and phase_close) else "⚠️ Unconfirmed"
    fig.suptitle(
        f'Paired FFT Spectrum — {model_name} L{layer}H{head}, Dims ({k1},{k2})  [{verdict}]',
        fontweight='bold', fontsize=13,
    )
    plt.tight_layout()

    clean_model_name = model_name.replace("/", "_")
    save_path = os.path.join(output_dir, f"paired_fft_{clean_model_name}_L{layer}H{head}_dims{k1}_{k2}.png")
    plt.savefig(save_path, dpi=300)
    print(f"✅ Paired FFT spectrum saved to {save_path}")
    plt.close()

def map_svd_to_frequencies(Vh, number_embeddings, valid_nums, top_n_dims=10):
    import numpy as np

    print(f"\n🔍 Mapping SVD Dimensions to Fourier Frequencies (Top {top_n_dims} Dims):")

    # Project numbers onto the top N dimensions
    projections = number_embeddings @ Vh[:top_n_dims].T

    for dim in range(top_n_dims):
        # 1. Get the 1D signal for this specific SVD dimension
        signal = projections[:, dim].cpu().numpy()
        signal = signal - np.mean(signal) # Mean center the 1D wave

        # 2. Run 1D Real FFT
        fft_values = np.fft.rfft(signal)
        fft_mags = np.abs(fft_values)

        # 3. Get Frequencies and Periods
        # rfftfreq returns frequencies for real-valued signals
        freqs = np.fft.rfftfreq(len(valid_nums), d=1.0)

        # Ignore the DC component (0 Hz)
        pos_mask = freqs > 0.01
        clean_freqs = freqs[pos_mask]
        clean_mags = fft_mags[pos_mask]

        if len(clean_mags) > 0:
            best_idx = np.argmax(clean_mags)
            best_period = 1.0 / clean_freqs[best_idx]
            signal_strength = clean_mags[best_idx] / np.sum(clean_mags) * 100

            print(f"   - Dim {dim:2d}: Dominant Period = {best_period:5.1f} | Signal Strength = {signal_strength:4.1f}%")

def plot_sine_cosine_waves(coords_2d_np, valid_nums, k1, k2, period, z_poly, output_dir=".", prefix=""):
    """
    Plots the individual X and Y SVD dimensions as separate 1D waves,
    overlaid with the ideal mathematical Sine and Cosine waves.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

    # 1. Extract the actual data waves from the SVD plane
    x_coords = coords_2d_np[:, 0]
    y_coords = coords_2d_np[:, 1]

    # 2. Generate the "Ideal" waves
    # Amplitude (A) is the average distance from the center
    radii = np.linalg.norm(coords_2d_np, axis=1)
    amplitude = np.mean(radii)

    # Calculate the ideal angle for each number using our linear fit (z_poly = [slope, intercept])
    ideal_angles = z_poly[0] * np.array(valid_nums) + z_poly[1]

    # Mathematical ideal curves
    ideal_x = amplitude * np.cos(ideal_angles)
    ideal_y = amplitude * np.sin(ideal_angles)

    # ==========================================
    # VISUALIZATION (2 Subplots)
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # --- Subplot 1: Dimension 1 (Cosine) ---
    ax1.plot(valid_nums, x_coords, marker='o', linestyle='-', color='teal', linewidth=2, label=f'Model Data (Dim {k1})')
    ax1.plot(valid_nums, ideal_x, linestyle='--', color='gray', linewidth=2, alpha=0.8, label=f'Ideal Cosine Wave (T ≈ {period:.1f})')
    ax1.axhline(0, color='black', linewidth=1, alpha=0.3)
    ax1.set_ylabel('Activation Magnitude', fontweight='bold')
    ax1.set_title(f'X-Axis (SVD Dim {k1}) vs. Ideal Cosine', fontweight='bold', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Subplot 2: Dimension 2 (Sine) ---
    ax2.plot(valid_nums, y_coords, marker='s', linestyle='-', color='darkorange', linewidth=2, label=f'Model Data (Dim {k2})')
    ax2.plot(valid_nums, ideal_y, linestyle='--', color='gray', linewidth=2, alpha=0.8, label=f'Ideal Sine Wave (T ≈ {period:.1f})')
    ax2.axhline(0, color='black', linewidth=1, alpha=0.3)
    ax2.set_xlabel('Number Value', fontweight='bold')
    ax2.set_ylabel('Activation Magnitude', fontweight='bold')
    ax2.set_title(f'Y-Axis (SVD Dim {k2}) vs. Ideal Sine', fontweight='bold', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{prefix}_sine_cosine_unpacking_L_H_dims_{k1}_{k2}.png")
    plt.savefig(save_path, dpi=300)
    print(f"✅ Sine/Cosine unpacking plot saved to {save_path}")
    plt.close()

def plot_fourier_spectrum(coords_2d, valid_nums, model_name, layer, head, output_dir="."):
    """
    Runs a Fast Fourier Transform (FFT) on the 2D plane to extract exact frequencies/periods.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    # 1. Combine X and Y into a single complex signal: z = x + iy
    # This is the mathematically perfect way to find frequency in a 2D circular plane
    x = coords_2d[:, 0].cpu().numpy()
    y = coords_2d[:, 1].cpu().numpy()
    complex_signal = x + 1j * y

    # 2. Run the FFT
    fft_values = np.fft.fft(complex_signal)
    fft_magnitudes = np.abs(fft_values)

    # Get the frequencies (d=1.0 because our numbers step by 1: 10, 11, 12...)
    freqs = np.fft.fftfreq(len(valid_nums), d=1.0)

    # We only care about positive frequencies (ignoring the 0 DC offset)
    pos_mask = freqs > 0.01
    clean_freqs = freqs[pos_mask]
    clean_mags = fft_magnitudes[pos_mask]

    # Convert frequency (cycles per number) to Period (numbers per cycle)
    # e.g., freq = 0.1 -> Period = 10 (Base-10 math)
    periods = 1.0 / clean_freqs

    # 3. Plot the Spectrum
    plt.figure(figsize=(10, 5))

    # Plot as a bar chart (stem plot) to clearly see the spikes
    markerline, stemlines, baseline = plt.stem(periods, clean_mags, basefmt=" ")
    plt.setp(stemlines, 'linewidth', 2, 'color', 'teal')
    plt.setp(markerline, 'markersize', 8, 'color', 'darkblue')

    plt.xlabel('Period $T$ (Numbers per full rotation)', fontweight='bold')
    plt.ylabel('Fourier Magnitude (Strength of Signal)', fontweight='bold')
    plt.title(f'FFT Frequency Spectrum\n{model_name} - Layer {layer}, Head {head}', fontweight='bold')
    plt.grid(True, alpha=0.3)

    # Highlight the absolute strongest frequency
    if len(clean_mags) > 0:
        best_idx = np.argmax(clean_mags)
        best_period = periods[best_idx]
        plt.annotate(f'Dominant Period: $T \\approx {best_period:.1f}$',
                     xy=(best_period, clean_mags[best_idx]),
                     xytext=(best_period + 5, clean_mags[best_idx]),
                     arrowprops=dict(facecolor='red', shrink=0.05, width=1.5, headwidth=6),
                     fontweight='bold', color='red')

    plt.tight_layout()

    # Save the plot
    clean_model_name = model_name.replace("/", "_")
    save_path = os.path.join(output_dir, f"fft_spectrum_{clean_model_name}_L{layer}H{head}.png")
    plt.savefig(save_path, dpi=300)
    print(f"✅ FFT Spectrum saved to {save_path}")
    plt.close()

def plot_candidate_geometry(model_name, layer, head, k1, k2, Vh, number_embeddings, valid_nums, circuit_type, output_dir="."):
    """
    Generates a 2D projection, a Phase Progression plot, AND a 3D Top-Component projection.
    """
    # 1. Calculate the 2D plane we found
    v1, v2 = Vh[k1], Vh[k2]
    coords_2d = torch.stack([number_embeddings @ v1, number_embeddings @ v2], dim=1)
    center_2d = coords_2d.mean(dim=0)
    centered_2d = coords_2d - center_2d

    radii = centered_2d.norm(dim=1).cpu().numpy()
    angles = torch.atan2(centered_2d[:, 1], centered_2d[:, 0]).cpu().numpy()
    unwrapped_angles = np.unwrap(angles)

    cv = np.std(radii) / np.mean(radii)
    lin = abs(np.corrcoef(valid_nums, unwrapped_angles)[0, 1])

    # 2. Extract the Top 3 Dimensions for the 3D "True Subspace" View
    v_top1, v_top2, v_top3 = Vh[0], Vh[1], Vh[2]
    coords_3d = torch.stack([
        number_embeddings @ v_top1,
        number_embeddings @ v_top2,
        number_embeddings @ v_top3
    ], dim=1)
    center_3d = coords_3d.mean(dim=0)
    centered_3d = (coords_3d - center_3d).cpu().numpy()

    # ==========================================
    # VISUALIZATION (3 Subplots)
    # ==========================================
    fig = plt.figure(figsize=(24, 7))

    # Plot 1: The 2D SVD Projection (The "Shadow")
    ax1 = fig.add_subplot(131)
    scatter = ax1.scatter(
        centered_2d[:, 0].cpu().numpy(), centered_2d[:, 1].cpu().numpy(),
        c=valid_nums, cmap='viridis', s=80, alpha=0.8, edgecolors='black'
    )
    ax1.set_xlabel(f'SVD Direction {k1}', fontweight='bold')
    ax1.set_ylabel(f'SVD Direction {k2}', fontweight='bold')
    ax1.set_title(f'2D Shadow Projection (Dims {k1} & {k2})\nCV = {cv:.3f}', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(0, color='black', linewidth=0.5, alpha=0.5)
    ax1.axvline(0, color='black', linewidth=0.5, alpha=0.5)
    ax1.set_aspect('equal')

    # Plot 2: Angle Linearity (Phase Progression)
    ax2 = fig.add_subplot(132)
    ax2.scatter(valid_nums, unwrapped_angles, c=valid_nums, cmap='viridis', s=80, alpha=0.8, edgecolors='black')
    z = np.polyfit(valid_nums, unwrapped_angles, 1)
    p = np.poly1d(z)
    ax2.plot(valid_nums, p(valid_nums), 'r--', linewidth=2, alpha=0.8)
    period = abs(2 * np.pi / z[0]) if z[0] != 0 else float('inf')
    ax2.set_xlabel('Number Value', fontweight='bold')
    ax2.set_ylabel('Unwrapped Angle (rad)', fontweight='bold')
    ax2.set_title(f'Phase Progression\nLinearity = {lin:.3f} | Est. Period = {period:.1f}', fontweight='bold')
    ax2.grid(True, alpha=0.3)

    # Plot 3: The 3D Subspace (Top 3 Components)
    ax3 = fig.add_subplot(133, projection='3d')
    sc3d = ax3.scatter(
        centered_3d[:, 0], centered_3d[:, 1], centered_3d[:, 2],
        c=valid_nums, cmap='viridis', s=80, alpha=0.8, edgecolors='black'
    )
    ax3.plot(centered_3d[:, 0], centered_3d[:, 1], centered_3d[:, 2], 'gray', alpha=0.5, linestyle='--') # Connect the dots
    ax3.set_xlabel('Top SVD Dir 0', fontweight='bold')
    ax3.set_ylabel('Top SVD Dir 1', fontweight='bold')
    ax3.set_zlabel('Top SVD Dir 2', fontweight='bold')
    ax3.set_title(f'3D True Subspace (Top 3 Dims)', fontweight='bold')

    # Add a colorbar to the whole figure
    cbar = fig.colorbar(scatter, ax=[ax1, ax2, ax3], fraction=0.02, pad=0.04)
    cbar.set_label('Number Value', fontsize=12, fontweight='bold')

    # Save it
    circuit_suffix = "QK" if "QK" in circuit_type else "OV"
    clean_model_name = model_name.replace("/", "_")
    save_path = os.path.join(output_dir, f"geometry_{clean_model_name}_L{layer}H{head}_{circuit_suffix}.png")

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ 3D Geometry Plot saved successfully to {save_path}!")
    plt.close()

    # ==========================================
    # Fast Fourier Transform extraction
    # ==========================================
    # Call the new FFT function using the 2D centered coordinates
    plot_fourier_spectrum(centered_2d, valid_nums, model_name, layer, head, output_dir)

    # ==========================================
    # INVOKE THE NEW SINE/COSINE UNPACKING PLOT
    # ==========================================
    prefix = f"{clean_model_name}_L{layer}H{head}_{circuit_suffix}"
    plot_sine_cosine_waves(
        coords_2d_np=centered_2d.cpu().numpy(),
        valid_nums=valid_nums,
        k1=k1,
        k2=k2,
        period=period,
        z_poly=z,
        output_dir=output_dir,
        prefix=prefix
    )

    # Mapping circular geometry to frequencies
    map_svd_to_frequencies(Vh, number_embeddings, valid_nums)

    # New: focused analysis on the identified circular pair
    map_pair_to_frequencies(Vh, number_embeddings, valid_nums, k1, k2,
                            model_name=model_name, layer=layer, head=head,
                            output_dir=output_dir)


def scan_offline_matrices(model_name="gpt2-small", cache_dir="svd_cache"):
    print(f"Loading {model_name} embeddings...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = HookedTransformer.from_pretrained(model_name, device=device)
    W_E = model.W_E.detach() # Shape: [vocab_size, d_model]

    # Robustly get the raw token IDs for numbers 0 through 100
    valid_nums = []
    number_tokens = []
    for n in range(1, 100):
        try:
            # Tokenizers can be finicky. This ensures we only use numbers that are single tokens.
            tok = model.to_single_token(f" {n}")
            number_tokens.append(tok)
            valid_nums.append(n)
        except Exception:
            continue

    print(f"Successfully extracted {len(valid_nums)} single-token numbers from 0 to 100.")
    number_embeddings = W_E[number_tokens, :]

    # Find all OV and QK cache files
    ov_files = glob.glob(f"{cache_dir}/*_ov.pt")
    qk_files = glob.glob(f"{cache_dir}/*_qk.pt")
    pt_files = ov_files + qk_files
    print(f"Scanning {len(pt_files)} cached OV/QK matrices...")

    best_overall_head = None
    best_overall_score = float('inf')
    best_candidate_data = None  # Store data required for the plot

    for file_path in pt_files:
        circuit_type = "QK (Routing)" if "_qk.pt" in file_path else "OV (Computation)"

        svd_data = torch.load(file_path, map_location=device, weights_only=True)
        Vh = svd_data['Vh']
        layer = svd_data['layer']
        head = svd_data['head']

        # Skip matrices belonging to other models
        if Vh.shape[-1] != number_embeddings.shape[1]:
            continue

        # Test combinatorial pairs of the top 10 reading directions
        for k1, k2 in itertools.combinations(range(10), 2):
            v1, v2 = Vh[k1], Vh[k2]

            coords = torch.stack([number_embeddings @ v1, number_embeddings @ v2], dim=1)
            center = coords.mean(dim=0)
            coords = coords - center

            radii = coords.norm(dim=1)
            angles = torch.atan2(coords[:, 1], coords[:, 0]).cpu().numpy()

            radii_mean = radii.mean().item()
            radius_cv = (radii.std() / radii_mean).item() if radii_mean > 1e-6 else float('inf')

            unwrapped = np.unwrap(angles)
            if np.std(unwrapped) > 1e-6:
                angle_lin = abs(np.corrcoef(valid_nums, unwrapped)[0, 1])
                if np.isnan(angle_lin): angle_lin = 0.0
            else:
                angle_lin = 0.0

            # Criteria for raw embedding scanning
            if radius_cv < 0.4 and angle_lin > 0.95:
                score = radius_cv - angle_lin
                if score < best_overall_score:
                    best_overall_score = score
                    best_overall_head = f"Layer {layer}, Head {head} (Dims {k1},{k2}) | CV: {radius_cv:.3f}, Lin: {angle_lin:.3f}"
                    # Assuming svd_data contains 'S' (the singular values)
                    if 'S' in svd_data:
                        S = svd_data['S']
                        print(f"\n📏 Testing the Singular Value Hypothesis for Dims {k1} & {k2}:")
                        print(f"   - Sigma {k1}: {S[k1].item():.4f}")
                        print(f"   - Sigma {k2}: {S[k2].item():.4f}")
                        ratio = S[k1].item() / S[k2].item()
                        print(f"   - Ratio (closer to 1.0 is better): {ratio:.4f}")

                    print(f"\n🔍 Found better candidate! {best_overall_head} in {circuit_type}")

                    # Track the data to plot later
                    best_candidate_data = {
                        "model_name": model_name,
                        "layer": layer,
                        "head": head,
                        "k1": k1,
                        "k2": k2,
                        "Vh": Vh,
                        "number_embeddings": number_embeddings,
                        "valid_nums": valid_nums,
                        "circuit_type": circuit_type,
                        "output_dir": "."
                    }

    print("\n" + "="*50)
    if best_candidate_data:
        print(f"🏆 Best Geometric Representation Found:\n{best_overall_head}")
        print("📊 Generating 3D visualization for the best candidate...")
        plot_candidate_geometry(**best_candidate_data)
    else:
        # FALLBACK: If we didn't hit the strict threshold, just show us the best thing we found anyway!
        print("⚠️ Strict threshold not met. But let's look at the best structure anyway!")
        if best_candidate_data is not None:
            plot_candidate_geometry(**best_candidate_data)
        else:
            print("❌ Complete failure. Try lowering the thresholds further.")
    print("="*50)

if __name__ == "__main__":
    #gpt-Neo-2.7B
    scan_offline_matrices("gpt2-small", "svd_cache") # Update dir path if needed