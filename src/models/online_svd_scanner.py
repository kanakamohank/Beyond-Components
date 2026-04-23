import torch
import torch.nn.functional as F
import numpy as np
import glob
import os
import re
import gc
import random
import time
import itertools
import warnings

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

from transformer_lens import HookedTransformer
from transformers import BitsAndBytesConfig
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")

def map_svd_to_frequencies(Vh, number_embeddings, valid_nums, top_n_dims=10):
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

def scan_model_on_the_fly(model_name="google/gemma-2-2b"):
    # TODO un-comment following block when you want to use
    print(f"🚀 Initializing On-The-Fly Scanner for {model_name}...")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # 1. Load Model in 16-bit
    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        dtype=torch.bfloat16
    )

    # ####### Quantized model testing block start #######
    # from transformers import BitsAndBytesConfig
    # import torch
    #
    # # Swap to Llama 3.2 1B (or 3B if your Mac handles it)
    # model_name = "meta-llama/Llama-3.2-1B"
    # print(f"🚀 Initializing On-The-Fly Scanner for {model_name} in 4-BIT QUANTIZATION...")
    #
    # device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    #
    # # Configure 4-bit quantization
    # quant_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_compute_dtype=torch.float16
    # )
    #
    # # Load Model with quantization directly through HookedTransformer
    # model = HookedTransformer.from_pretrained(
    #     model_name,
    #     device=device,
    #     hf_kwargs={"quantization_config": quant_config}
    # )
    # ######## Quantized model testing block end, REVISIT THIS ONCE APPROVED BY HF #######

    # 2. Handle Multi-Token Numbers (The Shallow Pass)
    print("🧠 Extracting contextualized number embeddings...")
    valid_nums = list(range(1, 100))
    number_prompts = [f"{n}" for n in valid_nums] # Removed space to match modern tokenizer norms

    # Get tokens (automatically pads to max length in the batch)
    tokens = model.to_tokens(number_prompts)

    # Run a fast shallow pass just to Layer 1.
    # This allows Layer 0 to fuse multi-digit tokens (like '1' and '0') into a single concept.
    _, cache = model.run_with_cache(tokens, stop_at_layer=2)

    number_embeddings = []
    for i, prompt in enumerate(number_prompts):
        # Find exactly where the number ends (ignoring padding)
        # model.to_tokens(prompt) returns [batch, seq_len]. Length of seq_len - 1 is the last token.
        seq_len = model.to_tokens(prompt).shape[1]
        last_token_idx = seq_len - 1

        # Extract the representation from the residual stream right before Layer 1
        fused_vector = cache["blocks.1.hook_resid_pre"][i, last_token_idx, :]
        number_embeddings.append(fused_vector)

    # Cast to float32 for stable SVD math
    number_embeddings = torch.stack(number_embeddings).float()

    best_overall_head = None
    best_score = float('inf')

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    print(f"📊 Scanning {n_layers} layers and {n_heads} heads ({n_layers * n_heads} total matrices)...")

    # 3. On-The-Fly Loop
    for layer in range(n_layers):
        for head in range(n_heads):
            # Compute W_OV dynamically
            W_V = model.W_V[layer, head].detach().float()
            W_O = model.W_O[layer, head].detach().float()
            W_OV = W_V @ W_O

            # # ==========================================
            # # 🔨 4-BIT SIMULATED QUANTIZATION
            # # ==========================================
            # # 1. Find the absolute maximum value in the matrix
            # max_val = torch.max(torch.abs(W_OV))
            # # 2. Create a scale to map the matrix into 4-bit bounds (-8 to +7)
            # scale = max_val / 7.0
            # # 3. Divide, round to nearest integer (crushing the precision), and scale back
            # W_OV_quantized = torch.round(W_OV / scale) * scale
            # # ==========================================

            # Run SVD
            try:
                U, S, Vh = torch.linalg.svd(W_OV)
            except RuntimeError:
                continue

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

                # Strict criteria
                if radius_cv < 0.4 and angle_lin > 0.9:
                    score = radius_cv - angle_lin
                    if score < best_score:
                        best_score = score
                        best_overall_head = f"Layer {layer}, Head {head} (Dims {k1},{k2}) | CV: {radius_cv:.3f}, Lin: {angle_lin:.3f}"

                        # Assuming svd_data contains 'S' (the singular values)
                        print(f"\n📏 Testing the Singular Value Hypothesis for Dims {k1} & {k2}:")
                        print(f"   - Sigma {k1}: {S[k1].item():.4f}")
                        print(f"   - Sigma {k2}: {S[k2].item():.4f}")
                        ratio = S[k1].item() / S[k2].item()
                        print(f"   - Ratio (closer to 1.0 is better): {ratio:.4f}")

                        print(f"🔍 New Best Candidate: {best_overall_head}")

                        # ---> TRACK THE DATA FOR PLOTTING <---
                        best_candidate_data = {
                            "model_name": model_name,
                            "layer": layer,
                            "head": head,
                            "k1": k1,
                            "k2": k2,
                            "Vh": Vh,
                            "number_embeddings": number_embeddings,
                            "valid_nums": valid_nums,
                            "circuit_type": "OV (Computation)"
                        }

    print("\n" + "="*50)
    if best_overall_head:
        print(f"🏆 Best Geometric Representation Found:\n{best_overall_head}")
        print("📊 Generating visualizations and running FFT...")

        # 1. Map the Fourier Frequencies
        map_svd_to_frequencies(
            best_candidate_data["Vh"],
            best_candidate_data["number_embeddings"],
            best_candidate_data["valid_nums"]
        )

        # 2. Generate the 3D plots and the Sine/Cosine waves
        # (Make sure plot_candidate_geometry internally calls plot_sine_cosine_waves like we set up earlier)
        plot_candidate_geometry(**best_candidate_data)

    else:
        print("❌ No clear geometric structure found.")
    print("="*50)

def test_output_plane_computation(model_name="google/gemma-7b", layer=14, head=2, dim1=2, dim2=5):
    print(f"\n🚀 Initiating Phase 4: Output Plane Verification on {model_name}")
    print(f"Target: Layer {layer}, Head {head} | U-Matrix Dims: {dim1} & {dim2}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load model
    model = HookedTransformer.from_pretrained(model_name, device=device, dtype=torch.bfloat16)

    # Extract the U matrix (The Writing Directions)
    W_V = model.W_V[layer, head].detach().float()
    W_O = model.W_O[layer, head].detach().float()
    W_OV = W_V @ W_O
    U, S, Vh = torch.linalg.svd(W_OV)
    u1, u2 = U[:, dim1], U[:, dim2]

    # ==========================================
    # 🧪 SANITY CHECK: Does U geometrically match V?
    # Because W_OV = U * S * Vh, the SVD guarantees that the columns of U
    # map perfectly to the rows of Vh. If Vh formed a circle, U *MUST* # theoretically form a circle when multiplied by the correct vector.
    # ==========================================
    print("🧪 Running U-Matrix Sanity Check...")
    ratio = (S[dim1] / S[dim2]).item()
    print(f"   U-Matrix Energy Balance (Sigma {dim1} / Sigma {dim2}): {ratio:.4f}")
    if abs(ratio - 1.0) > 0.2:
        print("   ⚠️ WARNING: The U-Matrix dimensions do not have balanced energy!")
    else:
        print("   ✅ U-Matrix is mathematically balanced for rotation.")

    # Generate clean, no-carry addition pairs
    random.seed(42)
    test_cases = []
    while len(test_cases) < 30:
        a, b = random.randint(10, 40), random.randint(10, 40)
        if (a % 10) + (b % 10) < 10:
            test_cases.append((a, b))
    # Sort test cases by sum so np.unwrap() works smoothly!
    test_cases.sort(key=lambda x: x[0] + x[1])

    # Run the live prompts
    target_sums = []
    output_coords = []

    print("\n🧠 Intercepting Live Computations...")
    for a, b in test_cases:
        prompt = f"{a} + {b} ="
        target_sums.append(a + b)

        # Robust token string matching
        str_tokens = model.to_str_tokens(prompt)
        eq_idx = -1
        for i, tok in enumerate(str_tokens):
            if "=" in tok:
                eq_idx = i
                break

        if eq_idx == -1:
            print(f"Skipping {prompt}: Could not find '=' token.")
            continue

        tokens = model.to_tokens(prompt)
        _, cache = model.run_with_cache(tokens)

        # Grab hook_z (the raw attention activation BEFORE W_O is applied)
        # hook_z shape: [batch, pos, n_heads, d_head]
        head_z = cache[f"blocks.{layer}.attn.hook_z"][0, eq_idx, head, :].float()

        # Manually apply W_O to see exactly what this head writes to the residual stream
        head_out = head_z @ W_O

        # Project onto the U-plane!
        coord_x = (head_out @ u1).item()
        coord_y = (head_out @ u2).item()
        output_coords.append([coord_x, coord_y])

    output_coords = torch.tensor(output_coords)

    # Measure the Geometry of the Answers
    radii = output_coords.norm(dim=1)
    cv = (radii.std() / radii.mean()).item()

    angles = torch.atan2(output_coords[:, 1], output_coords[:, 0]).numpy()
    unwrapped_angles = np.unwrap(angles)
    linearity = abs(np.corrcoef(target_sums, unwrapped_angles)[0, 1])

    # Let's test everything!
    target_a = [a for a, b in test_cases]
    target_b = [b for a, b in test_cases]
    target_tens = [(a+b)//10 for a, b in test_cases]
    target_units = [(a+b)%10 for a, b in test_cases]

    lin_sum = abs(np.corrcoef(target_sums, unwrapped_angles)[0, 1])
    lin_a = abs(np.corrcoef(target_a, unwrapped_angles)[0, 1])
    lin_b = abs(np.corrcoef(target_b, unwrapped_angles)[0, 1])
    lin_tens = abs(np.corrcoef(target_tens, unwrapped_angles)[0, 1])
    lin_units = abs(np.corrcoef(target_units, unwrapped_angles)[0, 1])

    print("\n" + "="*50)
    print("📊 Diagnostic Plane Analysis:")
    print(f"Coefficient of Variation (Is it a circle?): {cv:.4f} (Perfect!)")
    print(f"Correlation with Sum (a+b): {lin_sum:.4f}")
    print(f"Correlation with Operand A: {lin_a:.4f}")
    print(f"Correlation with Operand B: {lin_b:.4f}")
    print(f"Correlation with Tens Digit: {lin_tens:.4f}")
    print(f"Correlation with Units Digit of Sum: {lin_units:.4f}")
    print("="*50)

    # We will not unwrap. We will force the angles into a clean [0, 2pi] range.
    raw_angles = np.mod(angles, 2 * np.pi)
    target_units = [(a+b)%10 for a, b in test_cases]
    target_averages = [(a+b)/2 for a, b in test_cases]

    plt.figure(figsize=(10, 5))

    # Plot 1: Is it writing the Units Digit?
    plt.subplot(1, 2, 1)
    plt.scatter(target_units, raw_angles, color='purple', alpha=0.7)
    plt.title("Output Angle vs. Expected Units Digit")
    plt.xlabel("True Units Digit of (a+b)")
    plt.ylabel("Raw Output Angle (Radians)")
    plt.grid(True, alpha=0.3)

    # Plot 2: Is it writing the Average? (Attention Vector Math)
    plt.subplot(1, 2, 2)
    plt.scatter(target_averages, raw_angles, color='teal', alpha=0.7)
    plt.title("Output Angle vs. Operand Average")
    plt.xlabel("Average: (a+b)/2")
    plt.ylabel("Raw Output Angle (Radians)")
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("phase4_output_diagnostic_micosoft_Phi-3-mini-4k-instruct.png")
    print("✅ Diagnostic plot saved to phase4_output_diagnostic_micosoft_Phi-3-mini-4k-instruct.png. Open it!")

    print("\n" + "="*50)
    print("📊 Output Plane Computation Results:")
    print(f"Coefficient of Variation (Is it a circle?): {cv:.4f} (Lower is better)")
    print(f"Linearity (Does the angle match the sum?):  {linearity:.4f} (Higher is better)")

    if linearity > 0.90 and cv < 0.4:
        print("✅ SUCCESS: The attention head writes the mathematical sum geometrically!")
    else:
        print("❌ FAILED: The output vectors do not align with the trigonometric sums.")
    print("="*50)

#This is phase-5 mlp algnment but has a huge matrix calcuation on bigger models
def test_mlp_subspace_alignment(model_name, layer, head, dim1, dim2, top_k_mlp=50):
    print(f"\n🚀 Initiating Phase 5: MLP Subspace Alignment on {model_name}")
    print(f"Targeting Connection: Layer {layer} Head {head} Output ---> Layer {layer} MLP Input")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model = HookedTransformer.from_pretrained(model_name, device=device, dtype=torch.bfloat16)

    # ==========================================
    # 1. Extract the Attention Head's "Writing" Subspace (U)
    # ==========================================
    W_V = model.W_V[layer, head].detach().float()
    W_O = model.W_O[layer, head].detach().float()
    W_OV = W_V @ W_O

    U_attn, S_attn, _ = torch.linalg.svd(W_OV.to("cpu"))
    U_attn = U_attn.to("mps"),
    S_attn =  S_attn.to("mps")

    # Isolate our 2D trigonometric clock face
    attn_plane = torch.stack([U_attn[:, dim1], U_attn[:, dim2]], dim=1) # Shape: [d_model, 2]

    # ==========================================
    # 2. Extract the MLP's "Reading" Subspace
    # ==========================================
    mlp_module = model.blocks[layer].mlp

    # Modern models (Gemma, Phi-3, LLaMA) use Gated MLPs.
    # Both W_in and W_gate read directly from the residual stream.
    if hasattr(mlp_module, 'W_gate') and hasattr(mlp_module, 'W_in'):
        W_mlp_read = torch.cat([mlp_module.W_gate.detach().float(), mlp_module.W_in.detach().float()], dim=1)
        print("   Detected Gated MLP (concatenating W_gate and W_in).")
    elif hasattr(mlp_module, 'W_in'):
        W_mlp_read = mlp_module.W_in.detach().float()
        print("   Detected Standard MLP (using W_in).")
    else:
        raise AttributeError(f"Unrecognized MLP architecture in {model_name}")

    # Find the principal dimensions the MLP is paying the most attention to
    print("   Computing SVD on the massive MLP Reading Matrix... (This takes a few seconds)")
    U_mlp, S_mlp, _ = torch.linalg.svd(W_mlp_read.to("cpu"), full_matrices=False)
    U_mlp = U_mlp.to('mps'),
    S_mlp  = S_mlp.to('mps')

    # Isolate the top 'k' most important dimensions the MLP reads
    mlp_subspace = U_mlp[:, :top_k_mlp] # Shape: [d_model, top_k_mlp]

    # ==========================================
    # 3. Measure Subspace Alignment (Cosine Similarity)
    # ==========================================
    # Normalize our vectors just to be safe
    attn_plane_norm = torch.nn.functional.normalize(attn_plane, dim=0)
    mlp_subspace_norm = torch.nn.functional.normalize(mlp_subspace, dim=0)

    # Compute the projection of the Attention Plane onto the MLP Subspace
    # alignment_matrix shape: [2, top_k_mlp]
    alignment_matrix = torch.matmul(attn_plane_norm.T, mlp_subspace_norm)

    # Find the maximum cosine similarity
    max_alignment = alignment_matrix.abs().max().item()
    mean_top_alignment = np.sort(alignment_matrix.abs().cpu().numpy().flatten())[-5:].mean()

    print("\n" + "="*50)
    print("📊 MLP Subspace Alignment Results:")
    print(f"Max Cosine Similarity: {max_alignment:.4f}")
    print(f"Mean Top-5 Similarity: {mean_top_alignment:.4f}")

    # Contextualizing the math
    d_model = model.cfg.d_model
    baseline_expected = np.sqrt(1 / d_model)
    print(f"\nExpected random alignment in {d_model}-D space: ~{baseline_expected:.4f}")

    if max_alignment > 0.40: # Threshold for high-dimensional space
        print("✅ SUCCESS: Strong coupling detected! The MLP is explicitly wired to read this Attention Head's clock face.")
    else:
        print("❌ FAILED: Weak coupling. The MLP might be ignoring this specific geometric plane.")
    print("="*50)

#This is kind skipping svd calculation directly projecting 2d attention plane on mlp neurons
def test_safe_mlp_alignment(model_name="google/gemma-7b", layer=14, head=2, dim1=2, dim2=5):
    print(f"\n🚀 Initiating Phase 5: SAFE MLP Alignment on {model_name}")
    print(f"Targeting Connection: Layer {layer} Head {head} Output ---> Layer {layer} MLP Input")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Load model safely
    print("   Loading model...")
    model = HookedTransformer.from_pretrained(model_name, device=device, dtype=torch.bfloat16)
    d_model = model.cfg.d_model

    # ==========================================
    # 1. Extract Attention Subspace (Safe SVD on CPU)
    # ==========================================
    print("   Extracting Attention Head Writing Matrix...")
    W_V = model.W_V[layer, head].detach().float().cpu()
    W_O = model.W_O[layer, head].detach().float().cpu()
    W_OV = W_V @ W_O # Shape: [d_model, d_model]

    # Run SVD on CPU to prevent MPS crashes
    U_attn, _, _ = torch.linalg.svd(W_OV)

    # Isolate our 2D trigonometric clock face vectors
    u1 = U_attn[:, dim1].to(device)
    u2 = U_attn[:, dim2].to(device)

    # ==========================================
    # 2. Extract the MLP Reading Matrix
    # ==========================================
    print("   Extracting MLP Neurons...")
    mlp_module = model.blocks[layer].mlp

    if hasattr(mlp_module, 'W_gate') and hasattr(mlp_module, 'W_in'):
        # Concatenate gated neurons along the output dimension
        W_mlp = torch.cat([mlp_module.W_gate.detach(), mlp_module.W_in.detach()], dim=1).float()
    elif hasattr(mlp_module, 'W_in'):
        W_mlp = mlp_module.W_in.detach().float()
    else:
        raise AttributeError(f"Unrecognized MLP architecture in {model_name}")

    num_neurons = W_mlp.shape[1]
    print(f"   Testing alignment against {num_neurons} MLP neurons...")

    # ==========================================
    # 3. Direct Cosine Similarity (No SVD needed!)
    # ==========================================
    # W_mlp shape: [d_model, num_neurons]
    # u1, u2 shape: [d_model]

    # Normalize the vectors
    u1_norm = torch.nn.functional.normalize(u1, dim=0)
    u2_norm = torch.nn.functional.normalize(u2, dim=0)
    W_mlp_norm = torch.nn.functional.normalize(W_mlp, dim=0)

    # Project the 2D plane onto all MLP neurons simultaneously
    # This tells us exactly how much each neuron "listens" to the attention vectors
    sim_1 = torch.matmul(u1_norm, W_mlp_norm).abs() # Shape: [num_neurons]
    sim_2 = torch.matmul(u2_norm, W_mlp_norm).abs() # Shape: [num_neurons]

    # Calculate the combined 2D coupling magnitude for each neuron
    # Using Pythagorean theorem because u1 and u2 are orthogonal
    combined_coupling = torch.sqrt(sim_1**2 + sim_2**2)

    max_coupling = combined_coupling.max().item()
    top_10_mean = torch.topk(combined_coupling, 10).values.mean().item()

    print("\n" + "="*50)
    print("📊 Phase 5: MLP Subspace Alignment Results:")
    print(f"Max 2D Coupling (Strongest Neuron): {max_coupling:.4f}")
    print(f"Top 10 Neurons Average Coupling:    {top_10_mean:.4f}")

    # Contextualizing the math
    baseline_expected = np.sqrt(2 / d_model) # Expected magnitude of a 2D projection in random high-D space
    print(f"\nExpected random alignment in {d_model}-D space: ~{baseline_expected:.4f}")

    if max_coupling > baseline_expected * 5: # If it's 5x stronger than random noise
        print("✅ SUCCESS: Strong coupling detected! Specific MLP neurons are explicitly wired to this clock face.")
    else:
        print("❌ FAILED: Weak coupling. The MLP might be ignoring this specific geometric plane.")
    print("="*50)

def test_causal_phase_shift(model_name="google/gemma-7b", layer=14, head=2, dim1=2, dim2=5):
    print(f"\n🚀 Initiating Phase 6: Causal Phase-Shift on {model_name}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = HookedTransformer.from_pretrained(
        model_name, device=device, dtype=torch.bfloat16,
        center_writing_weights=False,
    )

    # 1. SVD on CPU for stability, extract Vh (the directions the scan found the circle in)
    print("   Extracting Vh Plane (SVD on CPU)...")
    W_V = model.W_V[layer, head].detach().float().cpu()
    W_O = model.W_O[layer, head].detach().float().cpu()
    W_OV = W_V @ W_O
    _, _, Vh = torch.linalg.svd(W_OV, full_matrices=False)

    v1 = Vh[dim1].to(device)
    v2 = Vh[dim2].to(device)

    # 2. Generate no-carry addition test cases, filter to ones the model gets right
    print("   Generating and filtering test cases...")
    random.seed(42)
    candidates = []
    while len(candidates) < 50:
        a, b = random.randint(10, 40), random.randint(10, 40)
        if (a % 10) + (b % 10) < 10:
            candidates.append((a, b))

    test_cases = []
    for a, b in candidates:
        prompt = f"{a} + {b} ="
        tokens = model.to_tokens(prompt)
        out = model.generate(tokens, max_new_tokens=3, verbose=False)
        pred = model.to_string(out[0][tokens.shape[1]:]).strip()
        expected = str(a + b)
        if pred == expected:
            test_cases.append({"a": a, "b": b, "clean_sum": a + b, "prompt": prompt})
        if len(test_cases) >= 20:
            break

    print(f"   Found {len(test_cases)} prompts where model gets the clean answer right.")
    if len(test_cases) < 5:
        print("   ❌ Too few valid test cases. Aborting.")
        return

    # 3. Measure angular period from live residual stream at operand B position
    #    Project resid_pre onto Vh rows — the SAME projection the scan used to find the circle
    print("   Measuring angular period from live activations (Vh projection at operand B)...")
    operand_b_values = []
    for case in test_cases:
        tokens = model.to_tokens(case["prompt"])
        str_tokens = model.to_str_tokens(case["prompt"])

        # Find operand B position: last digit token before '='
        b_idx = -1
        for i, tok in enumerate(str_tokens):
            if "=" in tok:
                for j in range(i - 1, -1, -1):
                    if any(c.isdigit() for c in str_tokens[j]):
                        b_idx = j
                        break
                break
        if b_idx == -1:
            continue

        _, cache = model.run_with_cache(tokens, stop_at_layer=layer + 1)
        resid = cache[f"blocks.{layer}.hook_resid_pre"][0, b_idx, :].float()
        c1 = torch.dot(resid, v1).item()
        c2 = torch.dot(resid, v2).item()
        angle = np.arctan2(c2, c1)
        operand_b_values.append((case["b"], case["clean_sum"], angle))

    if len(operand_b_values) >= 5:
        b_vals = [x[0] for x in operand_b_values]
        sums_list = [x[1] for x in operand_b_values]
        angles_list = [x[2] for x in operand_b_values]

        # Sort by operand B for unwrap
        sorted_by_b = sorted(zip(b_vals, angles_list), key=lambda x: x[0])
        b_sorted = [x[0] for x in sorted_by_b]
        ang_sorted = [x[1] for x in sorted_by_b]
        unwrapped = np.unwrap(ang_sorted)
        slope, _ = np.polyfit(b_sorted, unwrapped, 1)
        measured_period = abs(2 * np.pi / slope)
        linearity = abs(np.corrcoef(b_sorted, unwrapped)[0, 1])
        print(f"   Measured angular period (vs operand B): {measured_period:.2f}")
        print(f"   Angle-vs-B linearity: {linearity:.4f}")

        # Also check vs sum
        sorted_by_sum = sorted(zip(sums_list, angles_list), key=lambda x: x[0])
        s_sorted = [x[0] for x in sorted_by_sum]
        a_sorted = [x[1] for x in sorted_by_sum]
        unwrapped_s = np.unwrap(a_sorted)
        slope_s, _ = np.polyfit(s_sorted, unwrapped_s, 1)
        period_sum = abs(2 * np.pi / slope_s)
        lin_sum = abs(np.corrcoef(s_sorted, unwrapped_s)[0, 1])
        print(f"   Measured angular period (vs sum): {period_sum:.2f}")
        print(f"   Angle-vs-sum linearity: {lin_sum:.4f}")
    else:
        print("   ⚠️ Too few valid measurements for period estimation.")

    # 4. Run causal interventions with shifts of +1, +2, +3
    shifts_to_test = [1, 2, 3]
    print("\n" + "="*50)

    for case in test_cases[:10]:
        prompt = case["prompt"]
        clean_sum = case["clean_sum"]
        tokens = model.to_tokens(prompt)
        str_tokens = model.to_str_tokens(prompt)

        # Find operand B position: last digit token before '='
        b_idx = -1
        for i, tok in enumerate(str_tokens):
            if "=" in tok:
                for j in range(i - 1, -1, -1):
                    if any(c.isdigit() for c in str_tokens[j]):
                        b_idx = j
                        break
                break
        if b_idx == -1:
            print(f"   ❌ Skipping {prompt}: Could not find operand B token")
            continue

        print(f"   Operand B token: '{str_tokens[b_idx]}' at position {b_idx}")

        for shift_amount in shifts_to_test:
            expected_hacked = clean_sum + shift_amount
            theta = (2 * np.pi * shift_amount) / 10.0
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)

            print(f"🎯 {prompt} | Shift +{shift_amount} | Expect: {clean_sum} -> {expected_hacked}")

            magnitude_check_passed = True
            _b_idx = b_idx

            def rotation_hook(resid_pre, hook):
                nonlocal magnitude_check_passed
                # Guard: during KV-cache generation, seq length drops to 1
                if resid_pre.shape[1] <= _b_idx:
                    return resid_pre

                vec = resid_pre[0, _b_idx, :].clone().float()

                # Project onto the Vh plane (same basis the scan found the circle in)
                c1 = torch.dot(vec, v1)
                c2 = torch.dot(vec, v2)
                original_magnitude = torch.sqrt(c1**2 + c2**2).item()

                # Rotate
                new_c1 = c1 * cos_t - c2 * sin_t
                new_c2 = c1 * sin_t + c2 * cos_t

                new_magnitude = torch.sqrt(new_c1**2 + new_c2**2).item()
                if abs(original_magnitude - new_magnitude) > 1e-4:
                    magnitude_check_passed = False

                # Reconstruct: replace only the Vh-plane component
                vec = vec - (c1 * v1 + c2 * v2) + (new_c1 * v1 + new_c2 * v2)
                resid_pre[0, _b_idx, :] = vec.to(resid_pre.dtype)
                return resid_pre

            # Hook on residual stream BEFORE the target layer reads it
            with model.hooks(fwd_hooks=[(f"blocks.{layer}.hook_resid_pre", rotation_hook)]):
                patched_out = model.generate(tokens, max_new_tokens=3, verbose=False)

            patched_pred = model.to_string(patched_out[0][tokens.shape[1]:]).strip()

            status = "✅" if patched_pred == str(expected_hacked) else "❌"
            print(f"   {status} Clean: {clean_sum} | Hacked: '{patched_pred}' (wanted {expected_hacked})")
            if not magnitude_check_passed:
                print("   ⚠️ Warning: Vector magnitude was NOT preserved during rotation!")
    print("-" * 50)

def run_rigorous_logit_lens(model_name="google/gemma-7b"):
    print(f"\n🔬 Initiating Rigorous Logit Lens on {model_name}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = HookedTransformer.from_pretrained(model_name, device=device, dtype=torch.bfloat16)

    # The Prompt
    prompt = "12 + 14 ="

    # Gemma's tokenizer is weird. For "26", the next token it predicts is usually "2" or " 26".
    # We will track the single-digit "2" (Tens digit) or "6" (Units digit) depending on how it tokenizes.
    # Let's track the actual tokens for "2", "1" (from 12), and "4" (from 14).
    target_token_str = "2"   # The correct first digit of 26
    distractor_str_1 = "1"   # The first digit of Operand A (12)
    distractor_str_2 = "4"   # The units digit of Operand B (14)

    # Safely get token IDs # these following 3 lines worked for gemma but failed on Phi-3 with
    #AssertionError: Input string: 2 is not a single token!
    # target_id = model.to_single_token(target_token_str)
    # dist1_id = model.to_single_token(distractor_str_1)
    # dist2_id = model.to_single_token(distractor_str_2)
    # Safely get token IDs (bypasses Phi-3 and Llama tokenizer quirks)
    def get_safe_id(token_str):
        # prepend_bos=False stops the tokenizer from adding sequence start tokens
        # [0, -1] ensures we grab the actual digit even if a dummy space sneaks in
        return model.to_tokens(token_str, prepend_bos=False)[0, -1].item()

    target_id = get_safe_id(target_token_str)
    dist1_id = get_safe_id(distractor_str_1)
    dist2_id = get_safe_id(distractor_str_2)

    tokens = model.to_tokens(prompt)

    print(f"Prompt: '{prompt}'")
    print(f"Tracking Target: '{target_token_str}' (ID {target_id})")
    print(f"Tracking Distractor A: '{distractor_str_1}' (ID {dist1_id})")
    print(f"Tracking Distractor B: '{distractor_str_2}' (ID {dist2_id})")
    print("-" * 70)

    # Run model and cache all layers
    logits, cache = model.run_with_cache(tokens)

    # Get the residual stream at the '=' token across all layers
    # Shape: [num_layers, batch, pos, d_model]
    # Manually stack the residual stream from every layer
    n_layers = model.cfg.n_layers
    resid_stack = torch.stack([cache[f"blocks.{l}.hook_resid_post"] for l in range(n_layers)])

    # Isolate the '=' token (index -1) across all layers
    resid_at_eq = resid_stack[:, 0, -1, :]

    # Apply Unembedding to all layers at once
    unembedded_logits = model.unembed(model.ln_final(resid_at_eq))
    probs = unembedded_logits.softmax(dim=-1)

    # Calculate Ranks and Probabilities
    def get_stats(token_id):
        tok_probs = probs[:, token_id].detach().cpu().float().numpy()
        # argsort is slow, so we do it once and find the rank
        sorted_indices = unembedded_logits.argsort(dim=-1, descending=True)
        tok_ranks = (sorted_indices == token_id).nonzero(as_tuple=True)[1].detach().cpu().numpy()
        return tok_probs, tok_ranks

    target_probs, target_ranks = get_stats(target_id)
    d1_probs, d1_ranks = get_stats(dist1_id)
    d2_probs, d2_ranks = get_stats(dist2_id)

    print(f"{'Layer':<7} | {'Target Rank (2)':<17} | {'Distractor 1 Rank':<17} | {'Distractor 4 Rank':<17}")
    print("-" * 70)

    for layer_idx in range(len(target_ranks)):
        t_rank = target_ranks[layer_idx]
        d1_rank = d1_ranks[layer_idx]
        d2_rank = d2_ranks[layer_idx]

        # Add visual markers for when the target breaches the Top 10
        marker = "🔥 TOP 10!" if t_rank < 10 else ""

        print(f"Layer {layer_idx:<2} | Rank {t_rank:<6} ({target_probs[layer_idx]*100:5.2f}%) | Rank {d1_rank:<6} ({d1_probs[layer_idx]*100:5.2f}%) | Rank {d2_rank:<6} ({d2_probs[layer_idx]*100:5.2f}%) {marker}")

    print("-" * 70)
    print("Sanity Check 1 (Distractor): Did the Target rank significantly higher than the Distractors at the end?")
    print("Sanity Check 2 (Trajectory): Did the Target rank start high (memorization) or spike in the middle (computation)?")

def run_universal_logit_lens(model_name="microsoft/Phi-3-mini-4k-instruct"):
    print(f"\n🔬 Initiating Universal Logit Lens on {model_name}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = HookedTransformer.from_pretrained(model_name, device=device, dtype=torch.bfloat16)

    # 1. FEW-SHOT PROMPT to anchor instruct models
    prompt = "Math:\n10 + 10 = 20\n21 + 13 = 34\n12 + 14 ="

    # 2. Track EVERY possible way the tokenizer might write the answer
    target_strings = ["26", " 26", "2", " 2"]
    distractor_strings = ["12", " 12", "14", " 14", "1", " 1"]

    def get_token_id_safe(s):
        # Grab the very last token ID to bypass tokenizer prefix quirks
        return model.to_tokens(s, prepend_bos=False)[0, -1].item()

    target_ids = [get_token_id_safe(s) for s in target_strings]
    dist_ids = [get_token_id_safe(s) for s in distractor_strings]

    tokens = model.to_tokens(prompt)
    print(f"Prompt:\n{prompt}")
    print("-" * 70)

    # Run model and cache
    logits, cache = model.run_with_cache(tokens)

    # Verify what the model actually chose as the next token
    final_token_id = logits[0, -1, :].argmax().item()
    print(f"🎯 Model's ACTUAL Top Prediction: '{model.to_string([final_token_id])}' (ID: {final_token_id})")

    # Stack residual streams manually for version safety
    n_layers = model.cfg.n_layers
    resid_stack = torch.stack([cache[f"blocks.{l}.hook_resid_post"] for l in range(n_layers)])
    resid_at_eq = resid_stack[:, 0, -1, :]

    unembedded_logits = model.unembed(model.ln_final(resid_at_eq))
    probs = unembedded_logits.softmax(dim=-1)

    print("\nTracking the Highest Rank among all Target variants (26, 2, etc.)")
    print(f"{'Layer':<7} | {'Best Target Rank':<20} | {'Best Distractor Rank':<20}")
    print("-" * 70)

    for layer_idx in range(n_layers):
        layer_logits = unembedded_logits[layer_idx]
        sorted_indices = layer_logits.argsort(descending=True)

        # Find the best rank among all our target variations
        best_target_rank = float('inf')
        for t_id in target_ids:
            rank = (sorted_indices == t_id).nonzero(as_tuple=True)[0].item()
            if rank < best_target_rank: best_target_rank = rank

        # Find the best rank among distractors
        best_dist_rank = float('inf')
        for d_id in dist_ids:
            rank = (sorted_indices == d_id).nonzero(as_tuple=True)[0].item()
            if rank < best_dist_rank: best_dist_rank = rank

        marker = "🔥 TOP 10!" if best_target_rank < 10 else ""
        print(f"Layer {layer_idx:<2} | Rank {best_target_rank:<15} | Rank {best_dist_rank:<15} {marker}")

def run_causal_brain_transplant(model_name="google/gemma-7b", layer=14, head=2):
    print(f"\n🧠 Initiating Causal Brain Transplant on {model_name}")
    print(f"Targeting Arithmetic Router: Layer {layer}, Head {head}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = HookedTransformer.from_pretrained(model_name, device=device, dtype=torch.bfloat16)


    # 1. Setup Prompts
    clean_prompt = "Math:\n10 + 10 = 20\n21 + 13 = 34\n12 + 14 ="
    corrupted_prompt = "Math:\n10 + 10 = 20\n21 + 13 = 34\n12 + 15 ="

    clean_tokens = model.to_tokens(clean_prompt)
    corrupted_tokens = model.to_tokens(corrupted_prompt)

    # --- SANITY CHECK 1: Token Alignment ---
    clean_str_toks = model.to_str_tokens(clean_prompt)
    corr_str_toks = model.to_str_tokens(corrupted_prompt)

    print("\n🔍 Token Alignment Check:")
    print(f"Clean Length: {len(clean_str_toks)} | Corrupted Length: {len(corr_str_toks)}")
    if len(clean_str_toks) != len(corr_str_toks):
        print(f"Clean: {clean_str_toks[-5:]}")
        print(f"Corrp: {corr_str_toks[-5:]}")
        raise ValueError("CRASH PREVENTED: Prompts tokenized to different lengths. The patch would misalign.")

    # 2. Extract the "Clean Thought"
    hook_name = f"blocks.{layer}.attn.hook_z"
    _, clean_cache = model.run_with_cache(clean_tokens, names_filter=hook_name)

    # hook_z shape: [batch, pos, head_index, d_head]
    clean_head_output = clean_cache[hook_name][0, -1, head, :].clone()

    # 3. Define the Transplant Hook
    def brain_transplant_hook(z, hook):
        # --- SANITY CHECK 2: Safe Casting ---
        # Overwrite ONLY Head 2 at the very last token ('=') with the clean thought
        z[0, -1, head, :] = clean_head_output.to(z.dtype)
        return z

    # 4. Helper to print Top 5 probabilities cleanly
    def print_top_5(logits, run_name):
        # Look at the logits for the very last token in the sequence
        next_token_logits = logits[0, -1, :]
        probs = next_token_logits.softmax(dim=-1)
        top_probs, top_indices = probs.topk(5)

        print(f"\n[{run_name}] Top 5 Subconscious Thoughts:")
        for p, idx in zip(top_probs, top_indices):
            token_str = model.to_string([idx.item()]).replace('\n', '\\n')
            print(f"   '{token_str}': {p.item()*100:5.2f}%")

    # 5. Execute the Runs!
    print("\n" + "="*50)
    print("🏃 RUN 1: CLEAN BASELINE (12 + 14)")
    clean_logits = model(clean_tokens)
    print_top_5(clean_logits, "Clean (Expect '2' or ' 26')")

    print("-" * 50)
    print("🏃 RUN 2: CORRUPTED BASELINE (12 + 15)")
    corrupted_logits = model(corrupted_tokens)
    print_top_5(corrupted_logits, "Corrupted (Expect '2' or ' 27')")

    print("-" * 50)
    print("🧠 RUN 3: THE BRAIN TRANSPLANT")
    print(f"   Feeding the model 12 + 15, but forcing Head {layer}.{head} to think about 14...")

    intervened_logits = model.run_with_hooks(
        corrupted_tokens,
        fwd_hooks=[(hook_name, brain_transplant_hook)]
    )
    print_top_5(intervened_logits, "Transplanted (Does it flip back to 26?)")
    print("="*50)

def run_sledgehammer_transplant(model_name="google/gemma-7b", layer=14):
    print(f"\n🔨 Initiating FULL LAYER Sledgehammer Transplant on {model_name}")
    print(f"Targeting Entire Residual Stream after Layer {layer}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = HookedTransformer.from_pretrained(model_name, device=device, dtype=torch.bfloat16)

    # --- FIX 1: The Space Barrier ---
    # Notice the trailing space after the equals sign!
    # This forces the model to skip formatting and predict the actual digit.
    # We feed the model the tens digit so it is forced to predict the units digit next!
    clean_prompt = "Math:\n10 + 10 = 20\n21 + 13 = 34\n12 + 14 = 2"
    corrupted_prompt = "Math:\n10 + 10 = 20\n21 + 13 = 34\n12 + 15 = 2"

    clean_tokens = model.to_tokens(clean_prompt)
    corrupted_tokens = model.to_tokens(corrupted_prompt)

    # Token Alignment Check
    clean_str_toks = model.to_str_tokens(clean_prompt)
    corr_str_toks = model.to_str_tokens(corrupted_prompt)

    print("\n🔍 Token Alignment Check:")
    print(f"Clean Length: {len(clean_str_toks)} | Corrupted Length: {len(corr_str_toks)}")
    if len(clean_str_toks) != len(corr_str_toks):
        print(f"Clean: {clean_str_toks[-5:]}")
        print(f"Corrp: {corr_str_toks[-5:]}")
        raise ValueError("CRASH PREVENTED: Prompts tokenized to different lengths.")

    # --- FIX 2: The Sledgehammer (hook_resid_post) ---
    # Instead of hook_z (one head), we grab the entire highway after the block finishes
    hook_name = f"blocks.{layer}.hook_resid_post"
    _, clean_cache = model.run_with_cache(clean_tokens, names_filter=hook_name)

    # clean_layer_output shape: [d_model]
    clean_layer_output = clean_cache[hook_name][0, -1, :].clone()

    def sledgehammer_hook(resid, hook):
        # Overwrite the ENTIRE residual stream for the last token
        resid[0, -1, :] = clean_layer_output.to(resid.dtype)
        return resid

    def print_top_5(logits, run_name):
        next_token_logits = logits[0, -1, :]
        probs = next_token_logits.softmax(dim=-1)
        top_probs, top_indices = probs.topk(5)

        print(f"\n[{run_name}] Top 5 Subconscious Thoughts:")
        for p, idx in zip(top_probs, top_indices):
            token_str = model.to_string([idx.item()]).replace('\n', '\\n')
            print(f"   '{token_str}': {p.item()*100:5.2f}%")

    print("\n" + "="*50)
    print("🏃 RUN 1: CLEAN BASELINE (12 + 14 = )")
    clean_logits = model(clean_tokens)
    print_top_5(clean_logits, "Clean (Expect '2' or '26')")

    print("-" * 50)
    print("🏃 RUN 2: CORRUPTED BASELINE (12 + 15 = )")
    corrupted_logits = model(corrupted_tokens)
    print_top_5(corrupted_logits, "Corrupted (Expect '2' or '27')")

    print("-" * 50)
    print("🔨 RUN 3: THE SLEDGEHAMMER TRANSPLANT")
    print(f"   Feeding 12 + 15, but aggressively overwriting Layer {layer}'s entire output...")

    intervened_logits = model.run_with_hooks(
        corrupted_tokens,
        fwd_hooks=[(hook_name, sledgehammer_hook)]
    )
    print_top_5(intervened_logits, "Transplanted (Does it flip back to 26?)")
    print("="*50)

def plot_arithmetic_clock_face(model_name="google/gemma-7b", layer=14, head=2, dim1=2, dim2=5):
    print(f"\n🎨 Generating 2D SVD Clock Face Visualization for {model_name}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = HookedTransformer.from_pretrained(model_name, device=device, dtype=torch.bfloat16)

    # 1. Extract the Golden Head's Reading Plane via SVD
    print("   Extracting Subspace...")
    W_V = model.W_V[layer, head].detach().float().cpu()
    W_O = model.W_O[layer, head].detach().float().cpu()
    W_OV = W_V @ W_O

    # SVD to find the principal directions
    _, _, Vh = torch.linalg.svd(W_OV, full_matrices=False)
    v1 = Vh[dim1]
    v2 = Vh[dim2]

    # 2. Extract Token Embeddings for numbers 0-9
    print("   Projecting Numbers onto the 2D Plane...")
    digits = [str(i) for i in range(10)]

    # Gemma tokenizer quirk: sometimes numbers have a leading space in context.
    # We will grab both variants, but plot the clean digits.
    token_ids = [model.to_tokens(d, prepend_bos=False)[0, -1].item() for d in digits]

    # Get the raw embedding vectors for these digits
    W_E = model.W_E.detach().float().cpu()
    digit_embeddings = W_E[token_ids] # Shape: [10, d_model]

    # 3. Project the high-dimensional embeddings onto our 2D clock face
    x_coords = torch.matmul(digit_embeddings, v1).numpy()
    y_coords = torch.matmul(digit_embeddings, v2).numpy()

    # Center the coordinates for a beautiful plot
    x_coords -= x_coords.mean()
    y_coords -= y_coords.mean()

    # ==========================================
    # 4. Create the Publication-Ready Plot
    # ==========================================
    print("   Rendering Plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 10))

    # Color map to show the sequential progression of numbers
    colors = cm.viridis(np.linspace(0, 1, len(digits)))

    # Plot the points
    ax.scatter(x_coords, y_coords, c=colors, s=200, edgecolor='black', zorder=3)

    # Annotate the points with their digits
    for i, txt in enumerate(digits):
        ax.annotate(txt, (x_coords[i], y_coords[i]),
                    fontsize=20, weight='bold',
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', va='bottom')

    # Draw the implicit "Clock Face" circle
    radius = np.mean(np.sqrt(x_coords**2 + y_coords**2))
    circle = plt.Circle((0, 0), radius, color='gray', fill=False, linestyle='--', linewidth=2, alpha=0.5, zorder=1)
    ax.add_patch(circle)

    # Draw an arrow representing our Phase 6 causal intervention (e.g., shifting '4' to '6')
    idx_start, idx_end = 4, 6
    ax.annotate("",
                xy=(x_coords[idx_end], y_coords[idx_end]),
                xytext=(x_coords[idx_start], y_coords[idx_start]),
                arrowprops=dict(arrowstyle="->", color="red", lw=3, connectionstyle="arc3,rad=0.2"),
                zorder=2)

    # Add a text box explaining the intervention
    ax.text(0.05, 0.95, f"Phase 6 Intervention:\nManually rotating vector\nfrom '4' to '6' (+2 Shift)\nyields identical final sum.",
            transform=ax.transAxes, fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))

    # Formatting
    ax.set_aspect('equal')
    ax.set_title(f"Arithmetic Clock Face\n(Projected onto Layer {layer}, Head {head} SVD Plane)", fontsize=18, weight='bold', pad=20)
    ax.set_xlabel(f"Principal Component {dim1}", fontsize=14)
    ax.set_ylabel(f"Principal Component {dim2}", fontsize=14)

    # Remove ticks for a cleaner abstract math look
    ax.set_xticks([])
    ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("arithmetic_clock_face_msphi.png", dpi=300, bbox_inches='tight')
    print("✅ SUCCESS! Saved visualization to 'arithmetic_clock_face_msphi.png'")
    plt.show()

def sweep_svd_planes(model_name="google/gemma-7b", layer=14, head=2):
    print(f"\n🔬 Sweeping Top 10 SVD Reading Planes for {model_name} (Layer {layer}, Head {head})")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = HookedTransformer.from_pretrained(model_name, device=device, dtype=torch.bfloat16)

    W_V = model.W_V[layer, head].detach().float().cpu()
    W_O = model.W_O[layer, head].detach().float().cpu()
    W_OV = W_V @ W_O

    # FIX 1: Use Left Singular Vectors (U) to map the INPUT reading space!
    U, _, _ = torch.linalg.svd(W_OV, full_matrices=False)

    digits = [str(i) for i in range(10)]
    token_ids = [model.to_tokens(d, prepend_bos=False)[0, -1].item() for d in digits]
    W_E = model.W_E.detach().float().cpu()
    digit_embeddings = W_E[token_ids]

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(f"SVD PC Sweep (Input Reading Space U): Layer {layer}, Head {head}", fontsize=20, weight='bold')
    colors = cm.viridis(np.linspace(0, 1, 10))

    for i in range(9):
        ax = axes[i // 3, i % 3]
        pc_x, pc_y = i, i + 1

        # Grab columns of U (the input dimensions)
        v1, v2 = U[:, pc_x], U[:, pc_y]

        x_coords = torch.matmul(digit_embeddings, v1).numpy()
        y_coords = torch.matmul(digit_embeddings, v2).numpy()

        x_coords -= x_coords.mean()
        y_coords -= y_coords.mean()

        ax.scatter(x_coords, y_coords, c=colors, s=100, edgecolor='black')
        for j, txt in enumerate(digits):
            ax.annotate(txt, (x_coords[j], y_coords[j]), fontsize=12, weight='bold',
                        xytext=(0, 5), textcoords='offset points', ha='center')

        ax.set_title(f"PC {pc_x} vs PC {pc_y}")
        ax.set_xticks([])
        ax.set_yticks([])

    # FIX 2: Prevent the tight_layout from squishing into the suptitle
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("svd_pc_sweep_gemma.png", dpi=300)
    print("✅ Saved sweep to 'svd_pc_sweep_gemma.png'")

def plot_dynamic_clock_face(model_name="google/gemma-7b", layer=14, head=2, dim1=2, dim2=5):
    print(f"\n🌊 Extracting DYNAMIC Contextualized Clock Face for {model_name}")
    print(f"Targeting active residual stream at Layer {layer}, Head {head} before attention...")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = HookedTransformer.from_pretrained(model_name, device=device, dtype=torch.bfloat16)

    W_V = model.W_V[layer, head].detach().float().cpu()
    W_O = model.W_O[layer, head].detach().float().cpu()
    W_OV = W_V @ W_O

    # FIX 1: Use U to map the dimensions the head physically reads
    U, _, _ = torch.linalg.svd(W_OV, full_matrices=False)
    v1 = U[:, dim1]
    v2 = U[:, dim2]

    digits = [str(i) for i in range(10)]
    prompts = [f"Math:\n10 + 10 = 20\n21 + 13 = 34\n12 + {d} =" for d in digits]

    x_coords = []
    y_coords = []

    print("   Running equations and capturing live thoughts...")
    hook_name = f"blocks.{layer}.hook_resid_pre"

    for i, prompt in enumerate(prompts):
        tokens = model.to_tokens(prompt)
        str_toks = model.to_str_tokens(prompt)

        target_pos = -1
        # FIX 2: Strict match to prevent grabbing the '1' in '12' or '21'
        for idx in range(len(str_toks)-1, -1, -1):
            if str_toks[idx] == digits[i] or str_toks[idx] == f" {digits[i]}":
                target_pos = idx
                break

        if target_pos == -1:
            print(f"⚠️ Warning: Could not align token for digit {digits[i]} in prompt: {str_toks}")
            continue

        _, cache = model.run_with_cache(tokens, names_filter=hook_name)

        live_vector = cache[hook_name][0, target_pos, :].detach().float().cpu()

        x = torch.dot(live_vector, v1).item()
        y = torch.dot(live_vector, v2).item()

        x_coords.append(x)
        y_coords.append(y)

    x_coords = np.array(x_coords)
    y_coords = np.array(y_coords)
    x_coords -= x_coords.mean()
    y_coords -= y_coords.mean()

    print("   Rendering Plot...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 10))

    colors = cm.viridis(np.linspace(0, 1, len(digits)))

    ax.scatter(x_coords, y_coords, c=colors, s=200, edgecolor='black', zorder=3)

    for i, txt in enumerate(digits):
        ax.annotate(txt, (x_coords[i], y_coords[i]),
                    fontsize=20, weight='bold',
                    xytext=(0, 10), textcoords='offset points',
                    ha='center', va='bottom')

    radius = np.mean(np.sqrt(x_coords**2 + y_coords**2))
    circle = plt.Circle((0, 0), radius, color='gray', fill=False, linestyle='--', linewidth=2, alpha=0.5, zorder=1)
    ax.add_patch(circle)

    for i in range(len(digits) - 1):
        ax.plot([x_coords[i], x_coords[i+1]], [y_coords[i], y_coords[i+1]],
                color='gray', alpha=0.3, linestyle='-', zorder=1)

    ax.set_aspect('equal')
    ax.set_title(f"Dynamic Contextualized Arithmetic Manifold\n(Live Activations at Layer {layer}, Head {head})", fontsize=16, weight='bold', pad=20)
    ax.set_xlabel(f"Principal Component {dim1}", fontsize=14)
    ax.set_ylabel(f"Principal Component {dim2}", fontsize=14)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.text(0.05, 0.95, "Unlike static embeddings, these vectors\nrepresent the active 'thought' of the model\nafter 13 layers of mathematical context-mixing.",
            transform=ax.transAxes, fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))

    plt.tight_layout()
    plt.savefig("dynamic_clock_face_gemma.png", dpi=300, bbox_inches='tight')
    print("✅ SUCCESS! Saved visualization to 'dynamic_clock_face_gemma.png'")
    plt.show()

def plot_arithmetic_distance_matrices(model_name="google/gemma-7b", layer=14):
    print(f"\n📏 Calculating Distance Metrics for Contextualized Digits at Layer {layer}...")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = HookedTransformer.from_pretrained(model_name, device=device, dtype=torch.bfloat16)

    digits = [str(i) for i in range(10)]
    prompts = [f"Math:\n10 + 10 = 20\n21 + 13 = 34\n12 + {d} =" for d in digits]

    # Store the full high-dimensional vectors (No SVD reduction!)
    live_vectors = []
    hook_name = f"blocks.{layer}.hook_resid_pre"

    for i, prompt in enumerate(prompts):
        tokens = model.to_tokens(prompt)
        str_toks = model.to_str_tokens(prompt)

        target_pos = -1
        for idx in range(len(str_toks)-1, -1, -1):
            if str_toks[idx] == digits[i] or str_toks[idx] == f" {digits[i]}":
                target_pos = idx
                break

        _, cache = model.run_with_cache(tokens, names_filter=hook_name)
        # Keep the full d_model vector (e.g., 3072 dimensions)
        vec = cache[hook_name][0, target_pos, :].detach().float().cpu()
        live_vectors.append(vec)

    # Stack into a single tensor [10, d_model]
    V = torch.stack(live_vectors)

    # 1. Calculate Cosine Similarity Matrix
    cos_sim_matrix = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            cos_sim_matrix[i, j] = F.cosine_similarity(V[i].unsqueeze(0), V[j].unsqueeze(0)).item()

    # 2. Calculate L2 Euclidean Distance Matrix
    l2_dist_matrix = torch.cdist(V, V, p=2).numpy()

    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(f"High-Dimensional Arithmetic Structure (Layer {layer})", fontsize=18, weight='bold')

    sns.heatmap(cos_sim_matrix, annot=True, fmt=".2f", cmap="YlGnBu", ax=axes[0],
                xticklabels=digits, yticklabels=digits)
    axes[0].set_title("Cosine Similarity (Higher = Pointing same direction)")

    sns.heatmap(l2_dist_matrix, annot=True, fmt=".1f", cmap="rocket_r", ax=axes[1],
                xticklabels=digits, yticklabels=digits)
    axes[1].set_title("Euclidean (L2) Distance (Lower = Physically closer)")

    plt.tight_layout()
    plt.savefig(f"{model_name.split('/')[1]}_distance_matrices_layer_{layer}.png", dpi=300)
    print(f" saved matrices to {model_name.split('/')[1]}_distance_matrices_layer_{layer}.png")

def run_professors_automated_circle_scanner(model_name="google/gemma-7b", layer=14, head=2, max_n=10):
    print(f"\n👨‍🏫 Running Professor's Automated 2D Circle Scanner on {model_name}...")
    print(f"Targeting Layer {layer}, Head {head} over numbers 0 to {max_n-1}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = HookedTransformer.from_pretrained(model_name, device=device, dtype=torch.bfloat16)

    acts = []
    valid_ns = []
    hook_name = f"blocks.{layer}.hook_resid_pre"

    print("   Extracting contextualized activations...")
    for n in range(max_n):
        prompt = f"Math:\n10 + 10 = 20\n21 + 13 = 34\n12 + {n} ="
        tokens = model.to_tokens(prompt)
        str_tokens = model.to_str_tokens(prompt)

        # --- SANITY CHECK: Strict Token Alignment ---
        target_pos = -1
        for idx in range(len(str_tokens)-1, -1, -1):
            if str_tokens[idx] == str(n) or str_tokens[idx] == f" {n}":
                target_pos = idx
                break

        if target_pos == -1:
            print(f"   ⚠️ Skipping n={n}: Could not isolate exact token in {str_tokens}")
            continue

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_name)

        resid = cache[hook_name][0, target_pos, :].cpu()
        acts.append(resid)
        valid_ns.append(n)

    acts_tensor = torch.stack(acts).float() # Shape: [N, d_model]

    # --- THE CRUCIAL MATH FIX: Use U instead of Vt ---
    W_V = model.W_V[layer, head].detach().float().cpu()
    W_O = model.W_O[layer, head].detach().float().cpu()
    W_OV = W_V @ W_O
    U, S, _ = torch.linalg.svd(W_OV, full_matrices=False)

    print("   Testing all 45 top-10 2D dimension pairs...")
    best_circle = None
    best_score  = -1

    for k1, k2 in itertools.combinations(range(10), 2):
        # Using U (Left Singular Vectors) for reading space
        v1, v2 = U[:, k1], U[:, k2]

        coords = torch.stack([
            acts_tensor @ v1,
            acts_tensor @ v2
        ], dim=1)

        # Mean center
        coords = coords - coords.mean(dim=0)

        radii  = coords.norm(dim=1)
        angles = torch.atan2(coords[:, 1], coords[:, 0]).numpy()

        radius_mean = radii.mean().item()
        radius_cv = (radii.std() / (radius_mean + 1e-8)).item() if radius_mean > 1e-6 else float('inf')

        mean_delta = np.abs(np.diff(angles)).mean()
        if mean_delta < np.pi * 0.75:
            unwrapped = np.unwrap(angles)
            linearity = abs(np.corrcoef(valid_ns, unwrapped)[0, 1])
            if np.isnan(linearity): linearity = 0.0
        else:
            linearity = 0.0

        # maximize linearity, minimize spread
        score = linearity - radius_cv

        if score > best_score:
            best_score  = score
            best_circle = (k1, k2, radius_cv, linearity)

    k1, k2, cv, lin = best_circle
    print("\n" + "="*50)
    print("🏆 Automated 2D Circle Search Results:")
    print(f"Best SVD Plane: Dimensions ({k1}, {k2})")
    print(f"Radius CV (Spread): {cv:.3f} (Lower is better, ideal < 0.2)")
    print(f"Linearity (Circle): {lin:.3f} (Higher is better, ideal > 0.95)")
    print(f"Combined Score:     {best_score:.3f}")

    if best_score < 0.6:
        print("\n❌ VERDICT: Low score. The 2D circle hypothesis fails to adequately explain the manifold in this frontier model.")
    print("="*50)


def test_vector_translation_math(model_name="google/gemma-7b", layer=21):
    print(f"\n🧮 Testing High-Dimensional Vector Translation at Layer {layer}...")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = HookedTransformer.from_pretrained(model_name, device=device, dtype=torch.bfloat16)

    digits = [str(i) for i in range(10)]
    prompts = [f"Math:\n10 + 10 = 20\n21 + 13 = 34\n12 + {d} =" for d in digits]

    live_vectors = []
    hook_name = f"blocks.{layer}.hook_resid_pre"

    for i, prompt in enumerate(prompts):
        tokens = model.to_tokens(prompt)
        str_toks = model.to_str_tokens(prompt)

        # --- SANITY CHECK ---
        target_pos = -1
        for idx in range(len(str_toks)-1, -1, -1):
            if str_toks[idx] == digits[i] or str_toks[idx] == f" {digits[i]}":
                target_pos = idx
                break

        if target_pos == -1:
            raise ValueError(f"CRASH PREVENTED: Could not align token for digit {digits[i]} in prompt {str_toks}")

        _, cache = model.run_with_cache(tokens, names_filter=hook_name)
        vec = cache[hook_name][0, target_pos, :].detach().float().cpu()
        live_vectors.append(vec)

    if len(live_vectors) != 10:
        raise ValueError("CRASH PREVENTED: Did not extract exactly 10 vectors.")

    V = torch.stack(live_vectors)

    # 1. Calculate the high-dimensional "+1" direction vector
    step_vectors = []
    for i in range(9):
        step_vectors.append(V[i+1] - V[i])

    d_plus_1 = torch.stack(step_vectors).mean(dim=0)

    # 2. THE ULTIMATE MATH TEST: V_4 + d_+1 = ?
    v_4 = V[4]
    predicted_v_5 = v_4 + d_plus_1

    print("\n🎯 Translation Results: V_4 + d_+1")
    print("-" * 50)

    best_match_idx = -1
    best_sim = -1

    for i in range(10):
        sim = F.cosine_similarity(predicted_v_5.unsqueeze(0), V[i].unsqueeze(0)).item()
        if sim > best_sim:
            best_sim = sim
            best_match_idx = i

        marker = "⭐ (Target)" if i == 5 else ""
        print(f"Similarity to True V_{i}: {sim:.4f} {marker}")

    print("-" * 50)
    if best_match_idx == 5:
        print("✅ SUCCESS: The highest similarity is V_5! Vector translation is proven.")
    else:
        print(f"❌ FAILED: The translated vector is closest to V_{best_match_idx}.")

def run_unified_helix_scanner(model_name="google/gemma-7b", layer=14, head=2, max_n=100):
    print(f"\n🧬 Running Unified Helix Scanner (Fix A + B) on {model_name}")
    print(f"Targeting Layer {layer}, Head {head} over numbers 0 to {max_n-1}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    print("   Loading model...")
    # Loading in bfloat16 for speed and memory efficiency
    model = HookedTransformer.from_pretrained(model_name, device=device, dtype=torch.bfloat16)

    acts = []
    valid_ns = []
    hook_name = f"blocks.{layer}.hook_resid_pre"

    print(f"   Extracting contextualized activations for {max_n} numbers...")
    for n in range(max_n):
        prompt = f"What is {n} + 5?"
        tokens = model.to_tokens(prompt)
        str_tokens = model.to_str_tokens(prompt)

        # --- THE FIX: The Pre-Plus Locator ---
        # Instead of matching the string "42", we find the operator "+"
        # and grab the token immediately preceding it. This guarantees we get
        # the fully fused concept of the operand, regardless of fragmentation!
        target_pos = -1
        for idx, tok in enumerate(str_tokens):
            if '+' in tok:
                target_pos = idx - 1
                break

        if target_pos <= 0:
            print(f"   ⚠️ Skipping n={n}: Could not safely locate the '+' operator in {str_tokens}")
            continue

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_name)

        resid = cache[hook_name][0, target_pos, :].cpu()
        acts.append(resid)
        valid_ns.append(n)

    if len(valid_ns) < 10:
        print("❌ CRASH PREVENTED: Not enough valid vectors captured.")
        return

    acts_tensor = torch.stack(acts).float()
    print(f"   Successfully captured {len(valid_ns)} fused, contextualized vectors.")

    # --- FIX B: PCA within the Top SVD Subspace ---
    W_V = model.W_V[layer, head].detach().float().cpu()
    W_O = model.W_O[layer, head].detach().float().cpu()
    W_OV = W_V @ W_O

    # Extract Reading Directions (U)
    U, S, _ = torch.linalg.svd(W_OV, full_matrices=False)

    # Project activations into the Top-10 SVD Subspace
    top10_basis = U[:, :10]
    acts_in_top10 = (acts_tensor @ top10_basis).numpy()

    # Run PCA to find the true tilted rotation plane
    pca = PCA(n_components=10)
    acts_pca = pca.fit_transform(acts_in_top10)

    print("   Testing all PCA pairs for pure Modulo-10 Geometry...")
    best_score = -1
    best_result = None

    for k1, k2 in itertools.combinations(range(10), 2):
        coords = torch.tensor(acts_pca[:, [k1, k2]]).float()

        # Mean center to evaluate true circularity
        coords = coords - coords.mean(dim=0)

        radii = coords.norm(dim=1)
        angles = torch.atan2(coords[:, 1], coords[:, 0]).numpy()

        radius_mean = radii.mean().item()
        radius_cv = (radii.std() / (radius_mean + 1e-8)).item() if radius_mean > 1e-6 else float('inf')

        unwrapped = np.unwrap(angles)

        # Correlate continuous angles with continuous `n`
        linearity = abs(np.corrcoef(valid_ns, unwrapped)[0, 1])
        if np.isnan(linearity): linearity = 0.0

        # Calculate the explicit Period (T)
        slope, _ = np.polyfit(valid_ns, unwrapped, 1)
        period = abs(2 * np.pi / slope) if slope != 0 else float('inf')

        # Score heavily penalizes bad periods, bad radii, and bad linearity
        period_penalty = abs(period - 10.0) / 10.0
        score = linearity - radius_cv - period_penalty

        if score > best_score:
            best_score = score
            best_result = {
                'k1': k1, 'k2': k2,
                'cv': radius_cv,
                'lin': linearity,
                'period': period,
                'score': score
            }

    print("\n" + "="*60)
    print("🏆 UNIFIED PCA-within-SVD RESULTS:")
    print("="*60)
    print(f"Best Corrected Plane:  PCA Components ({best_result['k1']}, {best_result['k2']})")
    print(f"Radius CV (Spread):    {best_result['cv']:.3f} (Aiming for < 0.20)")
    print(f"Linearity (Circle):    {best_result['lin']:.3f} (Aiming for > 0.95)")
    print(f"Detected Period (T):   {best_result['period']:.2f} (Aiming for exactly 10.00)")
    print("-" * 60)

    # We allow a slightly wider CV (0.25) to account for remaining tokenizer noise
    if best_result['cv'] < 0.25 and best_result['lin'] > 0.90 and 9.5 < best_result['period'] < 10.5:
        print("✅ PROFESSOR WAS RIGHT: The true T=10 clock face was hidden by elliptical tilt and sparse data!")
    else:
        print("❌ VERDICT: Even after PCA correction and 100 numbers, the perfect 2D circle fails.")
        print("   This confirms the High-Dimensional Vector Translation theory is dominant.")
    print("="*60)

# ──────────────────────────────────────────────────────────────────
# CORE CORRECTION: Clarify U vs Vt for TransformerLens
# ──────────────────────────────────────────────────────────────────
# TransformerLens uses ROW VECTORS:  y = x @ W_OV = x @ (U Σ Vt)
# x interacts with U first, therefore:
#   U columns = READING  directions (input  space)  ← CORRECT
#   Vt rows   = WRITING  directions (output space)  ← DISCARDED
# ──────────────────────────────────────────────────────────────────

def collect_activations(model, layer, max_n=100, fixed_b=5, position="operand"):
    hook_name = f"blocks.{layer}.hook_resid_pre"
    acts, valid_ns = [], []

    for n in range(max_n):
        prompt     = f"What is {n} + {fixed_b}?"
        tokens     = model.to_tokens(prompt)
        str_tokens = model.to_str_tokens(prompt)

        if position == "operand":
            target_pos = -1
            for idx, tok in enumerate(str_tokens):
                if '+' in tok:
                    target_pos = idx - 1
                    break
            if target_pos <= 0:
                continue
        elif position == "final":
            target_pos = len(str_tokens) - 1

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_name)

        resid = cache[hook_name][0, target_pos, :].cpu().float()
        acts.append(resid)
        valid_ns.append(n)

    return torch.stack(acts), valid_ns

def analyze_helix(acts_tensor, valid_ns, label=""):
    ones_digits = np.array([n % 10 for n in valid_ns])

    acts_np       = acts_tensor.numpy()
    acts_centered = acts_np - acts_np.mean(0, keepdims=True)

    # ── SANITY CHECK FIX: Dynamic PCA Components ──
    # Prevents crash when analyzing pre-reduced SVD spaces
    n_samples, n_features = acts_centered.shape
    max_pca_comps = min(15, n_samples, n_features)

    pca      = PCA(n_components=max_pca_comps)
    acts_pca = pca.fit_transform(acts_centered)

    best_score, best_result = -np.inf, None
    all_results = []

    # Dynamically cap the loop so we don't look for PC11 in a 10-PC space
    max_loop_comps = min(12, max_pca_comps)

    for k1, k2 in itertools.combinations(range(max_loop_comps), 2):
        coords = torch.tensor(acts_pca[:, [k1, k2]]).float()
        coords = coords - coords.mean(0)

        radii  = coords.norm(dim=1)
        angles = torch.atan2(coords[:, 1], coords[:, 0]).numpy()

        rm = radii.mean().item()
        if rm < 1e-8:
            continue
        cv = (radii.std() / rm).item()

        mean_delta = np.abs(np.diff(angles)).mean()
        if mean_delta >= np.pi * 0.75:
            continue

        unwrapped = np.unwrap(angles)

        # ── Circular Phase Alignment ──
        ideal_angles = (ones_digits / 10.0) * 2 * np.pi
        lin_ones = np.mean(np.cos(angles - ideal_angles))

        # ── Linearity against raw n ──
        lin_raw = abs(np.corrcoef(valid_ns, unwrapped)[0, 1])
        if np.isnan(lin_raw):
            lin_raw = 0.0

        # ── Period from slope against raw n ──
        slope, _ = np.polyfit(valid_ns, unwrapped, 1)
        period   = abs(2 * np.pi / slope) if abs(slope) > 1e-8 else np.inf

        # ── Score ──
        period_error = abs(period - 10.0) / 10.0
        score = lin_raw - cv - 0.3 * period_error

        result = dict(k1=k1, k2=k2, cv=cv, lin_ones=lin_ones,
                      lin_raw=lin_raw, period=period, score=score,
                      coords=coords)
        all_results.append(result)

        if score > best_score:
            best_score, best_result = score, result

    if best_result is None:
        print(f"  {label}: No valid pairs found.")
        return None

    r = best_result
    print(f"\n  {label}")
    print(f"  Best PC pair      : ({r['k1']}, {r['k2']})")
    print(f"  Radius CV         : {r['cv']:.4f}   (target < 0.20)")
    print(f"  Phase Match (ones): {r['lin_ones']:.4f}   (target > 0.90)")
    print(f"  Lin (raw n)       : {r['lin_raw']:.4f}   (comparison)")
    print(f"  Period T          : {r['period']:.2f}   (target ~10.0)")

    if r['cv'] < 0.20 and r['lin_ones'] > 0.90 and 9 < r['period'] < 11:
        print("  ✅ CLEAN T=10 HELIX: Clock algorithm confirmed.")
    elif r['cv'] < 0.35 and r['lin_ones'] > 0.75:
        print("  ⚠️  PARTIAL HELIX: Helical structure present but impure.")
    elif r['lin_raw'] > 0.70 and not (9 < r['period'] < 11):
        print(f"  🔶 WRONG PERIOD: Angular structure exists but T={r['period']:.1f} ≠ 10.")
    else:
        print("  ❌ NO CLEAN HELIX at this position/layer.")

    return best_result

def svd_reading_directions(model, layer, head, acts_tensor, valid_ns):
    print(f"\n  SVD Reading-Direction Analysis: Layer {layer}, Head {head}")

    W_V  = model.W_V[layer, head].detach().float().cpu()
    W_O  = model.W_O[layer, head].detach().float().cpu()
    W_OV = W_V @ W_O

    U, S, Vt = torch.linalg.svd(W_OV, full_matrices=False)

    top10_U       = U[:, :10]
    acts_in_U     = (acts_tensor @ top10_U).numpy()

    pca      = PCA(n_components=10)
    acts_pca = pca.fit_transform(acts_in_U)

    result = analyze_helix(
        torch.tensor(acts_pca).float(),
        valid_ns,
        label=f"SVD-Reading-PCA L{layer}H{head}"
    )
    return result

def sweep_layers(model, layer_range, max_n=100, fixed_b=5):
    print("\n" + "="*60)
    print("LAYER SWEEP: Finding Cleanest Helix Layer")
    print("="*60)

    print(f"\n  {'Layer':>5}  {'Position':>9}  {'CV':>6}  "
          f"{'Phase(ones)':>11}  {'Period':>8}  Verdict")
    print("  " + "-"*67)

    layer_results = {}

    for layer in layer_range:
        for pos_name in ["operand", "final"]:
            try:
                acts, valid_ns = collect_activations(
                    model, layer, max_n=max_n,
                    fixed_b=fixed_b, position=pos_name
                )
            except Exception as e:
                continue

            if len(valid_ns) < 10:
                continue

            ones = np.array([n % 10 for n in valid_ns])
            acts_c = acts.numpy() - acts.numpy().mean(0, keepdims=True)
            pca    = PCA(n_components=10)
            pca_c  = pca.fit_transform(acts_c)

            best_cv, best_phase, best_T = 1.0, 0.0, np.inf

            for k1, k2 in itertools.combinations(range(8), 2):
                coords = torch.tensor(pca_c[:, [k1, k2]]).float()
                coords = coords - coords.mean(0)
                radii  = coords.norm(dim=1)
                angles = torch.atan2(coords[:, 1], coords[:, 0]).numpy()

                rm     = radii.mean().item()
                if rm < 1e-8:
                    continue
                cv = (radii.std() / rm).item()

                mean_delta = np.abs(np.diff(angles)).mean()
                if mean_delta < np.pi * 0.75:
                    unwrapped   = np.unwrap(angles)
                    ideal_angles = (ones / 10.0) * 2 * np.pi
                    phase_ones  = np.mean(np.cos(angles - ideal_angles))
                    slp, _      = np.polyfit(valid_ns, unwrapped, 1)
                    T           = abs(2*np.pi/slp) if abs(slp) > 1e-8 else np.inf
                else:
                    phase_ones, T = 0.0, np.inf

                if phase_ones - cv > best_phase - best_cv:
                    best_cv, best_phase, best_T = cv, phase_ones, T

            if best_cv < 0.20 and best_phase > 0.90 and 9 < best_T < 11:
                verdict = "✅ STRONG HELIX"
            elif best_cv < 0.35 and best_phase > 0.75:
                verdict = "⚠️  Partial"
            elif best_phase > 0.60:
                verdict = f"🔶 T={best_T:.1f}≠10"
            else:
                verdict = "❌ None"

            print(f"  {layer:>5}  {pos_name:>9}  {best_cv:>6.3f}  "
                  f"{best_phase:>11.3f}  {best_T:>8.2f}  {verdict}")

            layer_results[(layer, pos_name)] = dict(cv=best_cv, phase=best_phase, period=best_T)

    best_key = min(layer_results, key=lambda k: layer_results[k]['cv'] - layer_results[k]['phase'])
    br = layer_results[best_key]
    print(f"\n  Best: Layer {best_key[0]}, position='{best_key[1]}'  "
          f"CV={br['cv']:.3f}, Phase={br['phase']:.3f}, T={br['period']:.1f}")

    return layer_results, best_key

def run_corrected_pipeline(model_name="google/gemma-7b", layer=14, head=2):
    device = ("mps"  if torch.backends.mps.is_available()  else
              "cuda" if torch.cuda.is_available()          else "cpu")

    print(f"Loading {model_name}...")
    model = HookedTransformer.from_pretrained(model_name, device=device, dtype=torch.bfloat16)
    model.eval()

    print("\nCollecting operand-position activations at Layer", layer)
    acts_op, ns_op = collect_activations(model, layer, position="operand")
    r1 = analyze_helix(acts_op, ns_op, f"Direct PCA — operand pos, L{layer}")

    print("\nCollecting final-token activations at Layer", layer)
    acts_fin, ns_fin = collect_activations(model, layer, position="final")
    r2 = analyze_helix(acts_fin, ns_fin, f"Direct PCA — final pos, L{layer}")

    r3 = svd_reading_directions(model, layer, head, acts_op, ns_op)

    layer_results, best_key = sweep_layers(model, layer_range=range(8, 26), max_n=100)

    print("\n" + "═"*60)
    print("SUMMARY")
    print("═"*60)
    print(f"  Direct PCA operand  L{layer}: " + (f"CV={r1['cv']:.3f}, Phase={r1['lin_ones']:.3f}, T={r1['period']:.1f}" if r1 else "  failed"))
    print(f"  Direct PCA final    L{layer}: " + (f"CV={r2['cv']:.3f}, Phase={r2['lin_ones']:.3f}, T={r2['period']:.1f}" if r2 else "  failed"))
    print(f"  SVD-Reading L{layer}H{head}:  " + (f"CV={r3['cv']:.3f}, Phase={r3['lin_ones']:.3f}, T={r3['period']:.1f}" if r3 else "  failed"))
    print(f"  Best layer overall: Layer {best_key[0]}, position='{best_key[1]}'")

# ─────────────────────────────────────────────────────────────
# CORRECTED PHASE METRIC (Phase Invariance)
# ─────────────────────────────────────────────────────────────

def phase_match_optimized(angles_rad: np.ndarray,
                          ones_digits: np.ndarray,
                          n_offsets:   int = 720) -> tuple:
    ideal_base = (ones_digits / 10.0) * 2 * np.pi
    best_alignment = -1.0
    best_phi       = 0.0

    for phi in np.linspace(0, 2 * np.pi, n_offsets, endpoint=False):
        alignment = np.mean(np.cos(angles_rad - ideal_base - phi))
        if alignment > best_alignment:
            best_alignment = alignment
            best_phi       = phi

    return best_alignment, best_phi

def phase_match_original(angles_rad: np.ndarray,
                         ones_digits: np.ndarray) -> float:
    ideal_angles = (ones_digits / 10.0) * 2 * np.pi
    return float(np.mean(np.cos(angles_rad - ideal_angles)))

# ─────────────────────────────────────────────────────────────
# COLLECT ACTIVATIONS
# ─────────────────────────────────────────────────────────────

def collect_activations(model, layer, max_n=100,
                        fixed_b=5, position="operand"):
    hook_name = f"blocks.{layer}.hook_resid_pre"
    acts, valid_ns = [], []

    for n in range(max_n):
        prompt     = f"What is {n} + {fixed_b}?"
        tokens     = model.to_tokens(prompt)
        str_tokens = model.to_str_tokens(prompt)

        if position == "operand":
            target_pos = -1
            for idx, tok in enumerate(str_tokens):
                if '+' in tok:
                    target_pos = idx - 1
                    break
            if target_pos <= 0:
                continue
        else:  # "final"
            target_pos = len(str_tokens) - 1

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_name)

        resid = cache[hook_name][0, target_pos, :].cpu().float()
        acts.append(resid)
        valid_ns.append(n)

    return torch.stack(acts), valid_ns

# ─────────────────────────────────────────────────────────────
# CORE HELIX ANALYZER
# ─────────────────────────────────────────────────────────────

def analyze_helix_corrected(acts_tensor: torch.Tensor,
                            valid_ns:    list,
                            label:       str = "",
                            n_pca:       int  = 15,
                            n_pairs:     int  = 12) -> dict:
    ones_arr = np.array([n % 10 for n in valid_ns])

    acts_np  = acts_tensor.numpy()
    acts_c   = acts_np - acts_np.mean(0, keepdims=True)

    # Safety Cap for PCA
    n_samples, n_features = acts_c.shape
    safe_n_pca = min(n_pca, n_samples, n_features)

    pca      = PCA(n_components=safe_n_pca)
    acts_pca = pca.fit_transform(acts_c)

    best_score, best_result = -np.inf, None

    # Safety Cap for combinations
    safe_n_pairs = min(n_pairs, safe_n_pca)

    for k1, k2 in itertools.combinations(range(safe_n_pairs), 2):
        coords = torch.tensor(acts_pca[:, [k1, k2]]).float()
        coords = coords - coords.mean(0)

        radii  = coords.norm(dim=1)
        angles = torch.atan2(coords[:, 1], coords[:, 0]).numpy()

        rm = radii.mean().item()
        if rm < 1e-8:
            continue
        cv = (radii.std() / rm).item()

        mean_delta = np.abs(np.diff(angles)).mean()
        if mean_delta >= np.pi * 0.75:
            continue

        unwrapped = np.unwrap(angles)

        lin_raw = abs(float(np.corrcoef(valid_ns, unwrapped)[0, 1]))

        slope, _ = np.polyfit(valid_ns, unwrapped, 1)
        period   = abs(2 * np.pi / slope) if abs(slope) > 1e-8 else np.inf

        phase_orig = phase_match_original(angles, ones_arr)
        phase_corr, best_phi = phase_match_optimized(angles, ones_arr)

        period_err = abs(period - 10.0) / 10.0
        score = phase_corr - cv - 0.3 * period_err

        if score > best_score:
            best_score  = score
            best_result = dict(
                k1=k1, k2=k2, cv=cv,
                lin_raw=lin_raw,
                phase_orig=phase_orig,
                phase_corr=phase_corr,
                phase_offset_deg=np.degrees(best_phi),
                period=period,
                score=score,
                coords=coords,
                angles=angles,
                ones_arr=ones_arr,
            )

    if best_result is None:
        print(f"  {label}: No valid pairs found.")
        return None

    r = best_result
    print(f"\n  ── {label} ──")
    print(f"  Best PC pair       : ({r['k1']}, {r['k2']})")
    print(f"  Radius CV          : {r['cv']:.4f}   (target < 0.20)")
    print(f"  Lin (raw n)        : {r['lin_raw']:.4f}   (monotone order)")
    print(f"  Phase ORIGINAL     : {r['phase_orig']:.4f}   (BUGGY — for reference)")
    print(f"  Phase CORRECTED    : {r['phase_corr']:.4f}   (target > 0.85)  ← KEY")
    print(f"  Phase offset found : {r['phase_offset_deg']:.1f}°")
    print(f"  Period T           : {r['period']:.2f}   (target ~10.0)")

    _interpret(r)
    return best_result

def _interpret(r: dict):
    cv       = r['cv']
    pc       = r['phase_corr']
    period   = r['period']
    lin_raw  = r['lin_raw']

    t_ok  = 9.0 < period < 11.0
    cv_ok = cv < 0.20
    pc_ok = pc > 0.85

    print()
    if cv_ok and pc_ok and t_ok:
        print("  ✅ CLEAN T=10 CLOCK FACE CONFIRMED")
    elif pc_ok and t_ok and not cv_ok:
        print("  ⚠️  T=10 CLOCK FACE PRESENT but radius is impure (CV too high).")
    elif pc_ok and not t_ok:
        print(f"  🔶 HELIX PRESENT but WRONG PERIOD (T={period:.1f} ≠ 10.0)")
    elif not pc_ok and lin_raw > 0.90:
        print(f"  🔷 MONOTONE HELIX (not modular): Lin(raw n)={lin_raw:.3f}")
        print(f"     → This is the 'Vector Translation' mechanism.")
    else:
        print(f"  ❌ NO CLEAN HELIX at this position/layer.")

# ─────────────────────────────────────────────────────────────
# LAYER SWEEP
# ─────────────────────────────────────────────────────────────

def sweep_layers_corrected(model, layer_range, max_n=100, fixed_b=5):
    print("\n" + "="*65)
    print("LAYER SWEEP (Corrected Phase Metric)")
    print("="*65)
    print(f"\n  {'L':>3}  {'Pos':>7}  {'CV':>6}  "
          f"{'Phase_corr':>10}  {'Phase_orig':>10}  {'T':>7}  Verdict")
    print("  " + "-"*72)

    results = {}

    for layer in layer_range:
        for pos in ["operand", "final"]:
            try:
                acts, ns = collect_activations(
                    model, layer, max_n=max_n,
                    fixed_b=fixed_b, position=pos
                )
            except Exception:
                continue
            if len(ns) < 15:
                continue

            ones = np.array([n % 10 for n in ns])
            ac   = acts.numpy() - acts.numpy().mean(0, keepdims=True)

            # Safety Cap
            safe_n_pca = min(10, ac.shape[0], ac.shape[1])
            pca  = PCA(n_components=safe_n_pca)
            pca_c = pca.fit_transform(ac)

            best_cv, best_pc, best_po, best_T = 1.0, -1.0, 0.0, np.inf

            safe_n_pairs = min(8, safe_n_pca)
            for k1, k2 in itertools.combinations(range(safe_n_pairs), 2):
                coords = torch.tensor(pca_c[:, [k1, k2]]).float()
                coords = coords - coords.mean(0)
                radii  = coords.norm(dim=1)
                angles = torch.atan2(coords[:,1], coords[:,0]).numpy()

                rm = radii.mean().item()
                if rm < 1e-8:
                    continue
                cv = (radii.std() / rm).item()

                if np.abs(np.diff(angles)).mean() >= np.pi * 0.75:
                    continue

                unwrapped = np.unwrap(angles)
                slp, _ = np.polyfit(ns, unwrapped, 1)
                T      = abs(2*np.pi/slp) if abs(slp) > 1e-8 else np.inf

                pc, _  = phase_match_optimized(angles, ones)
                po     = phase_match_original(angles, ones)

                if pc - cv > best_pc - best_cv:
                    best_cv, best_pc, best_po, best_T = cv, pc, po, T

            if   best_cv < 0.20 and best_pc > 0.85 and 9 < best_T < 11:
                verdict = "✅ CLOCK"
            elif best_cv < 0.35 and best_pc > 0.70:
                verdict = "⚠️  Partial"
            elif best_pc > 0.60:
                verdict = f"🔶 T={best_T:.1f}"
            elif best_po < 0.15 and best_pc > 0.60:
                verdict = "🔷 PhaseShift"
            elif best_pc <= 0.60:
                # NEW: Catch the Vector Translation signature
                verdict = "❌ (Linear Translation)"
            else:
                verdict = "❌"

            print(f"  {layer:>3}  {pos:>7}  {best_cv:>6.3f}  "
                  f"{best_pc:>10.3f}  {best_po:>10.3f}  "
                  f"{best_T:>7.2f}  {verdict}")

            results[(layer, pos)] = dict(
                cv=best_cv, phase_corr=best_pc,
                phase_orig=best_po, period=best_T
            )

    if results:
        best = max(results, key=lambda k: (
                results[k]['phase_corr'] - results[k]['cv']
        ))
        br = results[best]
        print(f"\n  Best: L{best[0]} {best[1]}  "
              f"Phase_corr={br['phase_corr']:.3f}  "
              f"CV={br['cv']:.3f}  T={br['period']:.2f}")

    return results

# ─────────────────────────────────────────────────────────────
# MAIN RUNNER
# ─────────────────────────────────────────────────────────────

def run_phase_corrected_pipeline(model_name, layer, head):
    device = ("mps"  if torch.backends.mps.is_available() else
              "cuda" if torch.cuda.is_available()         else "cpu")

    print(f"\n{'='*65}")
    print(f"Phase-Corrected Helix Analysis: {model_name}")
    print(f"{'='*65}")

    model = HookedTransformer.from_pretrained(
        model_name, device=device, dtype=torch.bfloat16
    )
    model.eval()

    for pos in ["operand", "final"]:
        acts, ns = collect_activations(model, layer,
                                       position=pos, max_n=100)
        analyze_helix_corrected(
            acts, ns,
            label=f"{model_name.split('/')[-1]} L{layer} {pos}"
        )

    sweep_layers_corrected(model, layer_range=range(8, 26))

# ─────────────────────────────────────────────────────────────
# FOURIER ISOLATION
# ─────────────────────────────────────────────────────────────

def fourier_isolate_TN(acts_tensor: torch.Tensor,
                       valid_ns:    list,
                       period:      float) -> torch.Tensor:
    """Isolates the T=N Fourier component from the residual stream."""
    ns    = np.array(valid_ns, dtype=np.float32)
    cosT  = np.cos(2 * np.pi * ns / period)
    sinT  = np.sin(2 * np.pi * ns / period)
    basis = np.stack([cosT, sinT, np.ones(len(ns))], axis=1)

    acts_np = acts_tensor.numpy()
    A, _, _, _ = np.linalg.lstsq(basis, acts_np, rcond=None)
    acts_T   = basis[:, :2] @ A[:2, :]

    ss_total = np.sum((acts_np - acts_np.mean(0))**2)
    r2       = np.sum(acts_T**2) / (ss_total + 1e-8)
    print(f"  T={period:.1f} component explains {r2:.1%} of total variance")

    return torch.tensor(acts_T, dtype=torch.float32), r2

def check_after_isolation(acts_isolated: torch.Tensor,
                          valid_ns:      list,
                          period:        float,
                          label:         str = "") -> dict:
    ones = np.array([n % int(round(period)) for n in valid_ns])

    acts_np = acts_isolated.numpy()
    acts_c  = acts_np - acts_np.mean(0, keepdims=True)

    # Isolated matrix is rank 2 by definition. Safety cap PCA.
    pca      = PCA(n_components=min(5, acts_c.shape[0], acts_c.shape[1]))
    pca_out  = pca.fit_transform(acts_c)

    best_score, best = -np.inf, None
    n_pairs = min(4, pca_out.shape[1])

    for k1, k2 in itertools.combinations(range(n_pairs), 2):
        coords = torch.tensor(pca_out[:, [k1, k2]]).float()
        coords = coords - coords.mean(0)
        radii  = coords.norm(dim=1)
        angles = torch.atan2(coords[:, 1], coords[:, 0]).numpy()

        rm = radii.mean().item()
        if rm < 1e-8: continue
        cv = (radii.std() / rm).item()

        if np.abs(np.diff(angles)).mean() >= np.pi * 0.75:
            continue

        best_phase, best_phi = -1.0, 0.0
        for phi in np.linspace(0, 2*np.pi, 360, endpoint=False):
            p = np.mean(np.cos(angles - (ones/period)*2*np.pi - phi))
            if p > best_phase:
                best_phase, best_phi = p, phi

        unwrapped = np.unwrap(angles)
        slp, _ = np.polyfit(valid_ns, unwrapped, 1)
        T_det  = abs(2*np.pi/slp) if abs(slp) > 1e-8 else np.inf

        score  = best_phase - cv
        if score > best_score:
            best_score = score
            best = dict(k1=k1, k2=k2, cv=cv,
                        phase_corr=best_phase,
                        phase_offset_deg=np.degrees(best_phi),
                        period=T_det,
                        coords=coords, ones=ones)

    if best is None:
        print(f"  {label}: no valid pairs")
        return {}

    r = best
    print(f"\n  {label} — AFTER FOURIER ISOLATION:")
    print(f"  CV            : {r['cv']:.4f}   ({'✅ <0.20' if r['cv']<0.20 else '⚠️ still high'})")
    print(f"  Phase_corr    : {r['phase_corr']:.4f}   ({'✅ >0.85' if r['phase_corr']>0.85 else ''})")
    print(f"  Period T      : {r['period']:.2f}")

    if r['cv'] < 0.20 and r['phase_corr'] > 0.85:
        print("  ✅ CLEAN CLOCK FACE after isolation. Superposition resolved.")
    elif r['cv'] < r['cv'] * 0.6:
        print("  ⚠️  CV improved — superposition partially explains it.")
    else:
        print("  ❌ CV unchanged. High CV is intrinsic, not a superposition artifact.")

    return r

# ─────────────────────────────────────────────────────────────
# ROBUST CAUSAL PHASE SHIFT (Tokenizer Bug Fixed)
# ─────────────────────────────────────────────────────────────

def causal_phase_shift_test(model, layer, head, valid_ns, period: float = 10.0, n_tests: int = 20, seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)

    # Use U columns (reading directions)
    W_V  = model.W_V[layer, head].detach().float().cpu()
    W_O  = model.W_O[layer, head].detach().float().cpu()
    W_OV = W_V @ W_O
    U, _, _ = torch.linalg.svd(W_OV, full_matrices=False)
    v1, v2 = U[:, 0], U[:, 1]

    print(f"\n  Causal Phase-Shift Test: Layer {layer}, Head {head}")
    hook_name = f"blocks.{layer}.hook_resid_pre"

    no_carry, carry = [], []
    for _ in range(n_tests * 6):
        if len(no_carry) >= n_tests and len(carry) >= n_tests: break
        a, b = random.randint(5, 45), random.randint(5, 45)
        delta = random.choice([1, 2])

        ones_a   = a % 10
        ones_sum = ones_a + (b % 10)
        if ones_a + delta > 9: continue

        case = (a, b, delta)
        if ones_sum >= 10 and len(carry) < n_tests: carry.append(case)
        elif ones_sum < 10 and len(no_carry) < n_tests: no_carry.append(case)

    results = {}
    for case_type, cases in [("no_carry", no_carry), ("carry", carry)]:
        successes = 0
        details   = []

        for a, b, delta in cases:
            # We use a structured prompt to prevent zero-shot format hallucinations
            prompt     = f"Math:\n10 + 10 = 20\n21 + 13 = 34\n{a} + {b} ="
            str_tokens = model.to_str_tokens(prompt)
            tokens     = model.to_tokens(prompt)

            target_pos = -1
            for idx, tok in enumerate(str_tokens):
                if '+' in tok:
                    target_pos = idx - 1
            if target_pos <= 0: continue

            with torch.no_grad():
                _, cache = model.run_with_cache(tokens, names_filter=hook_name)
            h_orig = cache[hook_name][0, target_pos, :].cpu().float()

            # Rotate
            theta = 2.0 * np.pi * delta / period
            c1, c2 = (h_orig @ v1).item(), (h_orig @ v2).item()
            ct, st = np.cos(theta), np.sin(theta)
            h_rot = (h_orig - c1 * v1 - c2 * v2 + (ct*c1 - st*c2) * v1 + (st*c1 + ct*c2) * v2)

            # ── SANITY CHECK FIX: Safe Hook & Generation ──
            def hook(value, hook):
                # Only apply to the first pass, prevent KV-cache indexing crashes
                if value.shape[1] > target_pos:
                    value[0, target_pos, :] = h_rot.to(value.dtype)
                return value

            with torch.no_grad():
                # We must use the context manager for generation hooks
                with model.hooks(fwd_hooks=[(hook_name, hook)]):
                    out_tokens = model.generate(tokens, max_new_tokens=4, prepend_bos=False, verbose=False)

            # Extract just the newly generated text
            out_str = model.tokenizer.decode(out_tokens[0, tokens.shape[1]:])

            # Regex to find the first number in the output
            nums = re.findall(r'\d+', out_str)
            pred_o = int(nums[0]) % 10 if nums else -1

            expected_ones = (a + b + delta) % 10
            success       = (pred_o == expected_ones)
            successes    += int(success)

            details.append(f"    {a}+{b} rot+{delta}: pred={pred_o}, exp={expected_ones} {'✓' if success else '✗'} (Raw Out: '{out_str.strip()}')")

        rate = successes / max(len(cases), 1)
        results[case_type] = rate

        print(f"\n  [{case_type.upper()}]")
        for d in details[:8]: print(d)
        if len(details) > 8: print(f"    ... ({len(details)-8} more)")
        print(f"  Success: {successes}/{len(cases)} = {rate:.1%}")

    return results

# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────
def collect_activations_simple(model, layer, max_n=100, fixed_b=5):
    hook_name = f"blocks.{layer}.hook_resid_pre"
    acts, valid_ns = [], []
    for n in range(max_n):
        prompt = f"What is {n} + {fixed_b}?"
        tokens = model.to_tokens(prompt)
        str_tokens = model.to_str_tokens(prompt)
        target_pos = -1
        for idx, tok in enumerate(str_tokens):
            if '+' in tok:
                target_pos = idx - 1
                break
        if target_pos <= 0: continue
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_name)
        resid = cache[hook_name][0, target_pos, :].cpu().float()
        acts.append(resid)
        valid_ns.append(n)
    return torch.stack(acts), valid_ns


def run_final_pipeline(model_name: str, layer: int, head: int, period: float, best_layer: int):
    device = ("mps"  if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*65}\nFINAL PIPELINE: {model_name}\nTarget: L{best_layer}, Head: {head}, T: {period}\n{'='*65}")

    model = HookedTransformer.from_pretrained(model_name, device=device, dtype=torch.bfloat16)
    model.eval()

    acts, valid_ns = collect_activations_simple(model, best_layer, max_n=100)

    print(f"\n  Step 1: Fourier-isolating T={period:.1f} component...")
    acts_iso, r2 = fourier_isolate_TN(acts, valid_ns, period)

    print(f"\n  Step 2: Circle test on isolated component...")
    result = check_after_isolation(acts_iso, valid_ns, period, label=f"{model_name.split('/')[-1]}_L{best_layer}")

    print(f"\n  Step 3: Causal phase-shift at L{best_layer} H{head}...")
    causal = causal_phase_shift_test(model, best_layer, head, valid_ns=valid_ns, period=period, n_tests=20)

    print(f"\n{'='*65}\nFINAL RESULTS SUMMARY\n{'='*65}")
    print(f"  T={period:.1f} explains {r2:.1%} of activation variance")
    if result:
        print(f"  CV after isolation   : {result.get('cv', 'N/A'):.3f}")
        print(f"  Phase after isolation: {result.get('phase_corr', 'N/A'):.3f}")
    print(f"  Causal no-carry      : {causal.get('no_carry', 0):.1%}")


def get_ones_token_safe(model, ans: int):
    """
    Extracts the exact token ID for the ones digit of the answer.
    """
    ans_str = str(ans)
    if len(ans_str) != 2:
        return None, False
    ones_digit = ans_str[1]

    try:
        # Get the token for just the ones digit (e.g., "7")
        tok_ids = model.to_tokens(ones_digit, prepend_bos=False)[0]
        return tok_ids[-1].item(), True
    except Exception:
        return None, False

def build_fewshot_pairs(model, n_pairs: int, seed: int = 42):
    """
    Builds pairs where the prompt provides the tens digit,
    forcing the model to predict the ones digit as the next token.
    """
    random.seed(seed)
    pairs = []

    attempts = 0
    while len(pairs) < n_pairs and attempts < n_pairs * 10:
        attempts += 1
        a   = random.randint(10, 49)
        b   = random.randint(10, 49)
        ans = a + b

        tok, valid = get_ones_token_safe(model, ans)
        if not valid:
            continue

        tens_digit = str(ans)[0]

        # Notice we append the tens digit here!
        prompt = (f"Math:\n"
                  f"10 + 10 = 20\n"
                  f"21 + 13 = 34\n"
                  f"{a} + {b} = {tens_digit}")

        pairs.append({
            'a': a, 'b': b, 'ans': ans,
            'prompt': prompt,
            'ans_tok': tok,
            'carry': int((a % 10 + b % 10) >= 10),
            'ones_sum': (a + b) % 10,
        })

    if len(pairs) < n_pairs:
        print(f"  ⚠️  Only built {len(pairs)}/{n_pairs} pairs.")

    return pairs

def find_arithmetic_neurons_fixed(model, layer_range, n_pairs: int = 50, n_candidates: int = 100, seed: int = 42):
    print(f"\n  Building {n_pairs} single-token-answer pairs...")
    pairs = build_fewshot_pairs(model, n_pairs, seed)

    if len(pairs) < 5:
        print("  ❌ Not enough valid pairs. Cannot run analysis.")
        return {}, pairs

    all_prompts = [p['prompt'] for p in pairs]
    all_tokens  = model.to_tokens(all_prompts)
    ans_toks    = torch.tensor([p['ans_tok'] for p in pairs], device=model.cfg.device)

    print(f"  Running ablation analysis on {len(pairs)} pairs...")
    neuron_effects = {}

    for layer in layer_range:
        hook_name = f"blocks.{layer}.mlp.hook_post" # Correct TransformerLens path

        # Verify hook exists
        _, test_cache = model.run_with_cache(all_tokens[:1], names_filter=hook_name)
        if hook_name not in test_cache:
            continue

        print(f"\n  Layer {layer}...", flush=True)
        start = time.time()

        with torch.no_grad():
            base_logits, cache = model.run_with_cache(all_tokens, names_filter=hook_name)

            # FIX: Use LOGITS not probabilities.
            # Get the logit of the correct answer token at the final sequence position
            base_ans_logits = base_logits[torch.arange(len(pairs)), -1, ans_toks]

            mlp_acts   = cache[hook_name][:, -1, :]
            mean_acts  = mlp_acts.abs().mean(dim=0)
            candidates = mean_acts.topk(n_candidates).indices.tolist()

        effects = torch.zeros(model.cfg.d_mlp)

        for i, n_idx in enumerate(candidates):
            if i % 25 == 0 and i > 0:
                print(f"    ...{i}/{n_candidates}", flush=True)

            def make_hook(nidx):
                def hook(val, hook):
                    val[:, -1, nidx] = 0.0
                    return val
                return hook

            with torch.no_grad():
                abl_logits = model.run_with_hooks(
                    all_tokens,
                    fwd_hooks=[(hook_name, make_hook(n_idx))]
                )

                # FIX: Logit difference
                abl_ans_logits = abl_logits[torch.arange(len(pairs)), -1, ans_toks]
                logit_drop = (base_ans_logits - abl_ans_logits).mean().item()
                effects[n_idx] = logit_drop

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        neuron_effects[layer] = effects
        top5     = effects.topk(5)
        duration = time.time() - start

        print(f"  Layer {layer} done in {duration:.0f}s")
        print(f"    Top neurons:   {top5.indices.tolist()}")
        print(f"    Logit drops:   {top5.values.numpy().round(4).tolist()}")

        max_effect = top5.values[0].item()
        if max_effect > 1.0:
            print(f"    ✅ STRONG signal (>{max_effect:.1f} logit units)")
        elif max_effect > 0.2:
            print(f"    ⚠️  MODERATE signal ({max_effect:.3f} logit units)")
        elif max_effect > 0.05:
            print(f"    🔶 WEAK signal ({max_effect:.3f} logit units)")
        else:
            print(f"    ❌ NEGLIGIBLE signal ({max_effect:.4f} logit units)")
            print(f"       → Arithmetic at this layer is highly distributed.")

    return neuron_effects, pairs

def characterize_neurons(model, neuron_effects, pairs, top_n: int = 5):
    print("\n" + "="*60)
    print("NEURON CHARACTERIZATION")
    print("="*60)

    all_prompts = [p['prompt'] for p in pairs]
    all_tokens  = model.to_tokens(all_prompts)
    carries     = torch.tensor([p['carry']    for p in pairs], dtype=torch.float32)
    ones_sums   = torch.tensor([p['ones_sum'] for p in pairs], dtype=torch.float32)
    raw_sums    = torch.tensor([p['ans']      for p in pairs], dtype=torch.float32)

    for layer, effects in neuron_effects.items():
        top_indices = effects.topk(top_n).indices.tolist()
        hook_name   = f"blocks.{layer}.mlp.hook_post"

        with torch.no_grad():
            _, cache = model.run_with_cache(all_tokens, names_filter=hook_name)
        mlp_acts = cache[hook_name][:, -1, :].cpu().float()

        print(f"\nLayer {layer}:")

        for n_idx in top_indices:
            eff  = effects[n_idx].item()
            acts = mlp_acts[:, n_idx]

            carry_corr  = torch.corrcoef(torch.stack([acts, carries]))[0, 1].item()
            ones_corr   = torch.corrcoef(torch.stack([acts, ones_sums]))[0, 1].item()
            sum_corr    = torch.corrcoef(torch.stack([acts, raw_sums]))[0, 1].item()

            # Handle NaNs if a neuron is dead
            carry_corr = 0.0 if np.isnan(carry_corr) else carry_corr
            ones_corr = 0.0 if np.isnan(ones_corr) else ones_corr
            sum_corr = 0.0 if np.isnan(sum_corr) else sum_corr

            print(f"\n  Neuron {n_idx}  (logit drop = {eff:.4f})")
            print(f"    Corr w/ carry   : {carry_corr:+.3f}")
            print(f"    Corr w/ ones_sum: {ones_corr:+.3f}")
            print(f"    Corr w/ raw_sum : {sum_corr:+.3f}")

            if   carry_corr >  0.4:
                print("    → CARRY DETECTOR: fires when ones digits sum ≥ 10")
            elif carry_corr < -0.4:
                print("    → NO-CARRY DETECTOR: fires when no carry")
            elif abs(ones_corr) > 0.4:
                print("    → ONES-SUM ROUTER: tracks ones digit of result")
            elif abs(sum_corr) > 0.5:
                print("    → MAGNITUDE TRACKER: responds to size of sum")
            else:
                print("    → COMPLEX HEURISTIC: no single-variable explanation")

            top3_idx = acts.topk(3).indices
            bot3_idx = acts.topk(3, largest=False).indices
            print(f"    Fires strongest: {[(pairs[i]['a'], pairs[i]['b']) for i in top3_idx.tolist()]}")
            print(f"    Fires weakest:   {[(pairs[i]['a'], pairs[i]['b']) for i in bot3_idx.tolist()]}")

def run_analysis(model_name: str, layer_range, n_pairs: int = 50):
    device = ("mps"  if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"ARITHMETIC NEURON ANALYSIS: {model_name}")
    print(f"{'='*60}")

    model = HookedTransformer.from_pretrained(model_name, device=device, dtype=torch.bfloat16)
    model.eval()

    effects, pairs = find_arithmetic_neurons_fixed(model, layer_range, n_pairs=n_pairs, n_candidates=100)
    if effects:
        characterize_neurons(model, effects, pairs, top_n=5)


def extract_step_vector_at_output_layer(model, causal_layer, max_n=40, fixed_b=5):
    """
    Extracts d_+1 at the CAUSALLY PROVEN LAYER (e.g. Layer 25),
    not at the input representation layer.

    Critical distinction:
      Layer 21 d_+1: difference in how inputs are encoded
      Layer 25 d_+1: difference in how the answer is represented
                     immediately before output

    The second is what we need for causal vector injection.
    We use resid_POST at the final token (the answer assembly site).
    """
    hook_name = f"blocks.{causal_layer}.hook_resid_post"
    print(f"  Extracting d_+1 at causal Layer {causal_layer} (answer assembly site)...")

    live_vectors = []

    for n in range(max_n):
        prompt = (f"Math:\n10 + 10 = 20\n21 + 13 = 34\n"
                  f"{n} + {fixed_b} =")
        tokens = model.to_tokens(prompt)

        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, names_filter=hook_name)

        # Final token = '=' = where answer is assembled
        vec = cache[hook_name][0, -1, :].cpu().float()
        live_vectors.append(vec)

    V = torch.stack(live_vectors)   # [max_n, d_model]

    # d_+1 = average step in answer-representation space
    step_vectors = []
    for i in range(len(V) - 1):
        step_vectors.append(V[i+1] - V[i])

    d_plus_1 = torch.stack(step_vectors).mean(dim=0)
    mag      = torch.norm(d_plus_1).item()

    # Measure consistency: if VT is correct, all step vectors should be parallel
    norms    = torch.stack(step_vectors)
    norms    = norms / (norms.norm(dim=1, keepdim=True) + 1e-8)
    sim_mat  = norms @ norms.T
    n_steps  = len(step_vectors)
    off_diag = sim_mat[~torch.eye(n_steps, dtype=torch.bool)]
    mean_cos = off_diag.mean().item()

    print(f"  d_+1 magnitude: {mag:.4f}")
    print(f"  Step vector consistency (mean cosine sim): {mean_cos:.4f}")
    print(f"  {'✅ Consistent direction → VT plausible' if mean_cos > 0.5 else '⚠️  Inconsistent → VT unlikely at this layer'}")

    return d_plus_1.to(model.cfg.device, dtype=model.cfg.dtype), mean_cos


def causal_injection_at_proven_layer(model, causal_layer, d_plus_1,
                                     n_tests=20, seed=42):
    """
    Injects delta * d_+1 at the causally proven layer.

    Three injection strategies tested:

    Strategy A: Inject at final token of PROMPT (same as before, but correct layer)
    Strategy B: Inject at ALL generation steps (persistent injection)
    Strategy C: Inject scaled by layer norm magnitude (adaptive scaling)

    If Vector Translation is the mechanism, Strategy A or B should work at Layer 25.
    If neither works even at Layer 25, VT is not the mechanism.
    """
    random.seed(seed)
    torch.manual_seed(seed)

    hook_name   = f"blocks.{causal_layer}.hook_resid_post"
    prompt_len  = None

    print(f"\n  Causal Injection Test at Layer {causal_layer} (causally proven layer)")

    for strategy in ["A", "B"]:
        successes = 0
        details   = []

        print(f"\n  Strategy {strategy}: "
              f"{'inject at = token only' if strategy == 'A' else 'inject persistently on all steps'}")

        for _ in range(n_tests):
            a     = random.randint(15, 40)
            b     = random.randint(15, 40)
            delta = random.choice([1, 2])

            if (a % 10 + b % 10 + delta) >= 10:
                delta = 1
                if (a % 10 + b % 10 + 1) >= 10:
                    continue

            prompt = (f"Math:\n10 + 10 = 20\n21 + 13 = 34\n"
                      f"{a} + {b} =")
            tokens    = model.to_tokens(prompt)
            prompt_len = tokens.shape[1]
            expected  = a + b + delta

            def make_hook(strat, toks_len, d, dvec):
                def hook(value, hook): # FIXED: Changed hook_obj to hook
                    cur_len = value.shape[1]
                    if strat == "A":
                        if cur_len == toks_len:
                            value[0, -1, :] = (value[0, -1, :].float()
                                               + d * dvec.float()
                                               ).to(value.dtype)
                    elif strat == "B":
                        value[0, -1, :] = (value[0, -1, :].float()
                                           + d * dvec.float()
                                           ).to(value.dtype)
                    return value
                return hook

            with torch.no_grad():
                # FIXED: Wrapped in model.hooks context manager
                with model.hooks(fwd_hooks=[(hook_name, make_hook(strategy, prompt_len, delta, d_plus_1))]):
                    out_tokens = model.generate(
                        tokens,
                        max_new_tokens=4,
                        prepend_bos=False,
                        verbose=False
                    )

            out_str  = model.tokenizer.decode(out_tokens[0, tokens.shape[1]:])
            nums     = re.findall(r'\d+', out_str)
            pred_ans = int(nums[0]) if nums else -1

            original = a + b
            success  = (pred_ans == expected)
            successes += int(success)

            details.append(
                f"    {a}+{b}={original} | +{delta} | "
                f"pred={pred_ans} exp={expected} "
                f"{'✅' if success else '❌'} "
                f"(raw: '{out_str.strip()[:15]}')"
            )

        rate = successes / max(n_tests, 1)
        print(f"\n  Results ({strategy}):")
        for d in details[:8]:
            print(d)
        print(f"  Success: {successes}/{n_tests} = {rate:.1%}")

        if rate > 0.60:
            print(f"  ✅ VECTOR TRANSLATION CONFIRMED at Layer {causal_layer}")
        elif rate > 0.30:
            print(f"  ⚠️  PARTIAL EFFECT ({rate:.0%}) — VT direction found but impure.")
        else:
            print(f"  ❌ Injection failed even at causally proven layer.")

    return successes / n_tests


def compare_injection_layers(model, d_vectors_by_layer, n_tests=15, seed=99):
    """
    Injects d_+1 at multiple layers and compares success rates.
    This directly maps the 'correction window' — how many layers
    can undo a perturbation.

    Expected result:
      Late layers (close to output): high success
      Early layers (far from output): low success (corrected by subsequent layers)

    The layer where success drops to ~0% is where the
    'arithmetic attractor' overpowers the injection.
    """
    print("\n" + "="*60)
    print("LAYER COMPARISON: Where Does Injection Work?")
    print("="*60)

    random.seed(seed)
    test_cases = []
    while len(test_cases) < n_tests:
        a, b  = random.randint(15, 35), random.randint(15, 35)
        delta = 1
        if (a % 10 + b % 10 + delta) < 10:
            test_cases.append((a, b, delta))

    results = {}

    for layer, d_vec in sorted(d_vectors_by_layer.items()):
        hook_name = f"blocks.{layer}.hook_resid_post"
        successes = 0

        for a, b, delta in test_cases:
            prompt   = (f"Math:\n10 + 10 = 20\n21 + 13 = 34\n"
                        f"{a} + {b} =")
            tokens   = model.to_tokens(prompt)
            expected = a + b + delta
            plen     = tokens.shape[1]

            def make_hook(dvec, tlen):
                def hook(value, hook): # FIXED: Changed hook_obj to hook
                    if value.shape[1] == tlen:
                        value[0, -1, :] = (value[0, -1, :].float()
                                           + delta * dvec.float()
                                           ).to(value.dtype)
                    return value
                return hook

            with torch.no_grad():
                # FIXED: Wrapped in model.hooks context manager
                with model.hooks(fwd_hooks=[(hook_name, make_hook(d_vec, plen))]):
                    out = model.generate(
                        tokens, max_new_tokens=4,
                        prepend_bos=False, verbose=False
                    )

            out_str  = model.tokenizer.decode(out[0, tokens.shape[1]:])
            nums     = re.findall(r'\d+', out_str)
            pred     = int(nums[0]) if nums else -1
            successes += int(pred == expected)

        rate = successes / len(test_cases)
        results[layer] = rate
        bar  = "█" * int(rate * 30)
        print(f"  Layer {layer:2d}: {rate:.1%}  {bar}")

    layers_sorted = sorted(results.keys())
    for i, l in enumerate(layers_sorted[:-1]):
        if results[l] < 0.2 and results[layers_sorted[i+1]] > 0.4:
            print(f"\n  Transition: Layer {l} → {layers_sorted[i+1]}")
            break

    return results


def run_vector_translation_proof(model_name: str, causal_layer: int):
    device = ("mps"  if torch.backends.mps.is_available() else
              "cuda" if torch.cuda.is_available()         else "cpu")

    print(f"\n{'='*65}")
    print(f"VECTOR TRANSLATION PROOF: {model_name}")
    print(f"Target: Layer {causal_layer} (causally proven)")
    print(f"{'='*65}")

    model = HookedTransformer.from_pretrained(
        model_name, device=device, dtype=torch.bfloat16
    )
    model.eval()

    # Step 1: Extract d_+1 at the causally proven layer
    d_vec, consistency = extract_step_vector_at_output_layer(
        model, causal_layer=causal_layer
    )

    # Step 2: Inject at Layer 25 (where you proved causality)
    rate = causal_injection_at_proven_layer(
        model, causal_layer, d_vec, n_tests=20
    )

    # Step 3: If Layer 25 works, check other layers to map the window
    if rate > 0.30:
        print("\n  Mapping correction window across layers...")
        d_vectors = {}
        for l in range(max(0, causal_layer-6), causal_layer+1):
            dv, _ = extract_step_vector_at_output_layer(model, l)
            d_vectors[l] = dv

        compare_injection_layers(model, d_vectors)

    # Final verdict
    print(f"\n{'='*65}")
    print(f"VERDICT: {model_name.split('/')[-1]}")
    print(f"{'='*65}")
    print(f"  d_+1 consistency at Layer {causal_layer}: {consistency:.3f}")
    print(f"  Injection success at Layer {causal_layer}: {rate:.1%}")

    if consistency < 0.3:
        print("\n  ❌ STEP VECTORS ARE INCONSISTENT")
        print("     d_+1 points in different directions for different n.")
        print("     This disproves Vector Translation as a mechanism.")
        print("     The arithmetic manifold is curved, not linear.")
    elif rate > 0.5:
        print("\n  ✅ VECTOR TRANSLATION CONFIRMED")
        print("     A consistent +1 direction exists AND causally controls output.")
    elif consistency > 0.5 and rate < 0.2:
        print("\n  🔷 CONSISTENT DIRECTION, ZERO CAUSAL EFFECT")
        print("     d_+1 is geometrically consistent (smooth manifold)")
        print("     but injecting it does not change the answer.")
        print("     This is the same pattern as the helix:")
        print("     The linear manifold is REPRESENTATIONAL, not COMPUTATIONAL.")
        print("     The model stores numbers on a line but does not")
        print("     compute by walking along it.")
    else:
        print("\n  ⚠️  AMBIGUOUS — interpret layer comparison results")

def deep_characterize_neuron(model, layer: int, neuron_idx: int,
                             n_grid: int = 30):
    """
    Complete characterization of a single high-impact neuron.
    Tests every combination of ones digits (0-9 × 0-9 = 100 pairs).
    """
    # FIXED: Correct TransformerLens hook name
    hook_name = f"blocks.{layer}.mlp.hook_post"

    print(f"\nDeep characterization: Layer {layer}, Neuron {neuron_idx}")
    print("Testing all 100 ones-digit combinations (a_ones × b_ones)...")

    activation_grid = np.zeros((10, 10))

    for a_ones in range(10):
        for b_ones in range(10):
            a = 10 + a_ones
            b = 10 + b_ones
            prompt = (f"Math:\n10 + 10 = 20\n21 + 13 = 34\n"
                      f"{a} + {b} =")
            tokens = model.to_tokens(prompt)

            with torch.no_grad():
                _, cache = model.run_with_cache(
                    tokens, names_filter=hook_name
                )
            act = cache[hook_name][0, -1, neuron_idx].item()
            activation_grid[a_ones, b_ones] = act

    print("\n  Activation grid (rows=a_ones, cols=b_ones):")
    print("       " + "  ".join(f"{j:4d}" for j in range(10)))
    print("     " + "-" * 50)
    for i in range(10):
        row = "  ".join(f"{activation_grid[i,j]:4.1f}" for j in range(10))
        print(f"  {i} |  {row}")

    mean_act   = activation_grid.mean()
    std_act    = activation_grid.std()
    threshold  = mean_act + 0.5 * std_act

    print(f"\n  Mean activation: {mean_act:.3f}")
    print(f"  Std:             {std_act:.3f}")
    print(f"  Strong-firing combinations (act > {threshold:.2f}):")

    strong_pairs = []
    for i in range(10):
        for j in range(10):
            if activation_grid[i, j] > threshold:
                ones_sum = (i + j) % 10
                carry    = int(i + j >= 10)
                strong_pairs.append((i, j, activation_grid[i,j],
                                     ones_sum, carry))

    strong_pairs.sort(key=lambda x: x[2], reverse=True)
    for a_o, b_o, act, ones_s, carry in strong_pairs[:15]:
        print(f"    a_ones={a_o}, b_ones={b_o}: "
              f"act={act:.2f}, sum_ones={ones_s}, carry={carry}")

    print("\n  Hypothesis testing:")
    sum_activations = {}
    for i in range(10):
        for j in range(10):
            s = i + j
            if s not in sum_activations:
                sum_activations[s] = []
            sum_activations[s].append(activation_grid[i, j])

    print("  Mean activation by (a_ones + b_ones) value:")
    for s in range(19):
        if s in sum_activations:
            mean_s = np.mean(sum_activations[s])
            bar    = "█" * max(0, int((mean_s - mean_act) * 3))
            print(f"    sum={s:2d}: {mean_s:6.2f}  {bar}")

    flat_acts  = activation_grid.flatten()
    carries    = np.array([int(i + j >= 10) for i in range(10) for j in range(10)], dtype=float)
    ones_sums  = np.array([(i + j) % 10 for i in range(10) for j in range(10)], dtype=float)
    raw_sums   = np.array([i + j for i in range(10) for j in range(10)], dtype=float)

    corr_carry    = np.corrcoef(flat_acts, carries)[0, 1]
    corr_ones_sum = np.corrcoef(flat_acts, ones_sums)[0, 1]
    corr_raw_sum  = np.corrcoef(flat_acts, raw_sums)[0, 1]

    print(f"\n  Correlations over full 10x10 grid:")
    print(f"    With carry flag  : {corr_carry:+.3f}")
    print(f"    With ones_sum    : {corr_ones_sum:+.3f}")
    print(f"    With raw sum(0-18): {corr_raw_sum:+.3f}")

    print("\n  What answer token does this neuron PROMOTE?")
    _find_promoted_token(model, layer, neuron_idx)

    return activation_grid


def _find_promoted_token(model, layer: int, neuron_idx: int):
    """
    Direct Logit Attribution (DLA).
    """
    try:
        W_out = model.blocks[layer].mlp.W_out   # [d_mlp, d_model]
        neuron_direction = W_out[neuron_idx, :]  # [d_model]

        W_U   = model.W_U                        # [d_model, vocab]
        token_logits = (neuron_direction.float() @ W_U.float())

        top_promoted   = token_logits.topk(10)
        top_suppressed = token_logits.topk(10, largest=False)

        print("  Top promoted tokens:")
        for tok_id, logit in zip(top_promoted.indices.tolist(),
                                 top_promoted.values.tolist()):
            tok_str = model.tokenizer.decode([tok_id])
            print(f"    '{tok_str}' (id={tok_id}): {logit:+.2f}")

        print("\n  Top suppressed tokens:")
        for tok_id, logit in zip(top_suppressed.indices.tolist(),
                                 top_suppressed.values.tolist()):
            tok_str = model.tokenizer.decode([tok_id])
            print(f"    '{tok_str}' (id={tok_id}): {logit:+.2f}")

    except Exception as e:
        print(f"  Could not compute token promotion: {e}")

def run_final_characterization(model_name: str, key_layer: int, key_neuron: int):
    device = ("mps"  if torch.backends.mps.is_available() else
              "cuda" if torch.cuda.is_available()         else "cpu")

    model = HookedTransformer.from_pretrained(
        model_name, device=device, dtype=torch.bfloat16
    )
    model.eval()

    grid = deep_characterize_neuron(model, key_layer, key_neuron)
    return grid

# # ─────────────────────────────────────────────────────────────
# # CORE TOOL 1: Direct Logit Attribution (DLA)
# # ─────────────────────────────────────────────────────────────
#
# def direct_logit_attribution(model, prompts_and_answers: list, layer_range=None):
#     if layer_range is None:
#         layer_range = range(model.cfg.n_layers)
#
#     W_U = model.W_U
#     n_layers = model.cfg.n_layers
#     n_heads  = model.cfg.n_heads
#
#     attn_dla = torch.zeros(n_layers, n_heads)
#     mlp_dla  = torch.zeros(n_layers)
#     embed_dla = 0.0
#     n_valid = 0
#
#     for prompt, correct_token in prompts_and_answers:
#         tokens = model.to_tokens(prompt)
#
#         with torch.no_grad():
#             logits, cache = model.run_with_cache(tokens)
#
#         pred = logits[0, -1, :].argmax().item()
#         if pred != correct_token:
#             del cache, logits
#             if torch.backends.mps.is_available(): torch.mps.empty_cache()
#             continue
#
#         n_valid += 1
#
#         # Keep the target column on the MPS device for fast math
#         W_U_col = W_U[:, correct_token].float()
#
#         def dla_score(vector):
#             # Do the dot product on the GPU, .item() safely returns the scalar
#             return (vector.float() @ W_U_col).item()
#
#         embed_out = cache["hook_embed"][0, -1, :]
#         embed_dla += dla_score(embed_out)
#
#         for layer in layer_range:
#             # hook_z contains the head activations before the output projection
#             z = cache[f"blocks.{layer}.attn.hook_z"][0, -1, :, :]
#             for head in range(n_heads):
#                 # Calculate head_result on the fly: z * W_O
#                 head_out = z[head] @ model.blocks[layer].attn.W_O[head]
#                 attn_dla[layer, head] += dla_score(head_out)
#
#             mlp_out = cache[f"blocks.{layer}.hook_mlp_out"]
#             mlp_dla[layer] += dla_score(mlp_out[0, -1, :])
#
#         del cache, logits
#         if torch.backends.mps.is_available(): torch.mps.empty_cache()
#
#     if n_valid == 0:
#         print("  ⚠️ No correct predictions found. Cannot compute DLA.")
#         return None, None, 0
#
#     return attn_dla / n_valid, mlp_dla / n_valid, embed_dla / n_valid


# def build_arithmetic_prompts(model, n_pairs=100, seed=42):
#     print("  Generating prompts...")
#     random.seed(seed)
#     pairs = []
#     attempts = 0
#
#     while len(pairs) < n_pairs:
#         attempts += 1
#         if attempts > 5000:
#             print(f"  ⚠️ Stopping early: Could only generate {len(pairs)} valid pairs.")
#             break
#
#         a   = random.randint(10, 49)
#         b   = random.randint(10, 49)
#         ans = a + b
#         prompt  = f"Math:\n10 + 10 = 20\n21 + 13 = 34\n{a} + {b} ="
#         ans_str = f" {ans}"
#
#         try:
#             tok_ids = model.to_tokens(ans_str, prepend_bos=False)[0]
#             # Gemma splits two-digit numbers. We take the FIRST token it generates.
#             # Example: " 34" -> " 3", which is enough to prove it knows the math path.
#             if len(tok_ids) > 0:
#                 pairs.append((prompt, tok_ids[0].item()))
#         except Exception:
#             continue
#
#     return pairs

# # ─────────────────────────────────────────────────────────────
# # CORE TOOL 2: SVD Computational Signal Strength
# # ─────────────────────────────────────────────────────────────
#
# def svd_computational_signal_strength(model, prompts_and_answers: list, layer_range=None):
#     if layer_range is None:
#         layer_range = range(model.cfg.n_layers)
#
#     print("\n  SVD Computational Signal Strength per Layer:")
#
#     answers = []
#     layer_outputs = {l: [] for l in layer_range}
#
#     for prompt, tok in prompts_and_answers[:80]:
#         tokens = model.to_tokens(prompt)
#         with torch.no_grad():
#             logits, cache = model.run_with_cache(tokens)
#
#         if logits[0, -1, :].argmax().item() != tok:
#             del cache, logits
#             continue
#
#         # Extract the actual numbers from the prompt string directly
#         # The prompt always ends with "{a} + {b} ="
#         last_line = prompt.strip().split('\n')[-1]
#         equation = last_line.replace('=', '').split('+')
#         actual_answer = float(equation[0].strip()) + float(equation[1].strip())
#
#         answers.append(actual_answer)
#
#         for layer in layer_range:
#             mlp_delta = cache[f"blocks.{layer}.hook_mlp_out"][0, -1, :].cpu().float()
#             # hook_attn_out is natively cached and equals the sum of all head outputs
#             attn_delta = cache[f"blocks.{layer}.hook_attn_out"][0, -1, :].cpu().float()
#             layer_outputs[layer].append(mlp_delta + attn_delta)
#
#         # Free cache
#         del cache, logits
#         if torch.backends.mps.is_available():
#             torch.mps.empty_cache()
#
#     if len(answers) < 5:
#         print("  Not enough correct predictions.")
#         return {}
#
#     ans_np = np.array(answers)
#     results = {}
#
#     for layer in layer_range:
#         if not layer_outputs[layer]: continue
#
#         L = torch.stack(layer_outputs[layer]).numpy()
#         U_l, S_l, Vt_l = np.linalg.svd(L - L.mean(0), full_matrices=False)
#         r2_by_k = {}
#
#         for k in [1, 2, 5, 10]:
#             if k > len(S_l): break
#             proj   = U_l[:, :k] @ np.diag(S_l[:k])
#             w, _, _, _ = np.linalg.lstsq(proj, ans_np, rcond=None)
#             pred   = proj @ w
#             ss_res = np.sum((ans_np - pred)**2)
#             ss_tot = np.sum((ans_np - ans_np.mean())**2)
#             r2_by_k[k] = max(0, 1 - ss_res / (ss_tot + 1e-8))
#
#         results[layer] = r2_by_k
#         r2_10 = r2_by_k.get(10, 0)
#         bar   = "█" * int(r2_10 * 40)
#         verdict = "✅ STRONG" if r2_10 > 0.5 else "⚠️ Moderate" if r2_10 > 0.2 else "❌ Weak"
#         print(f"  Layer {layer:2d}: R²(k=10)={r2_10:.3f}  {bar}  {verdict}")
#
#     return results
#
# # ─────────────────────────────────────────────────────────────
# # CORE TOOL 3: Backtracking From Layer 25
# # ─────────────────────────────────────────────────────────────
#
# def backtrack_from_causal_layer(model, prompts_and_answers: list, causal_layer: int = 25):
#     hook_l25 = f"blocks.{causal_layer}.hook_resid_post"
#     l25_acts, correct_logits_at_25 = [], []
#     valid_prompts = []
#
#     # Store ONLY the necessary CPU tensors, not the full cache
#     extracted_caches = []
#
#     for prompt, tok in prompts_and_answers[:30]: # Limit to 30 for memory safety
#         tokens = model.to_tokens(prompt)
#         with torch.no_grad():
#             logits, cache = model.run_with_cache(tokens)
#
#         if logits[0, -1, :].argmax().item() != tok:
#             del cache, logits
#             continue
#
#         l25_act_cpu = cache[hook_l25][0, -1, :].cpu().float()
#         l25_acts.append(l25_act_cpu)
#
#         W_U_col = model.W_U[:, tok].cpu().float()
#         correct_logits_at_25.append((l25_act_cpu @ W_U_col).item())
#         valid_prompts.append((prompt, tok))
#
#         # Extract only what we need for backtracking to CPU immediately
#         mini_cache = {}
#         for layer in range(min(causal_layer + 1, model.cfg.n_layers)):
#             mini_cache[f"MLP{layer}"] = cache[f"blocks.{layer}.hook_mlp_out"][0, -1, :].cpu().float()
#             # Extract the smaller 'z' tensor instead
#             mini_cache[f"ATTN_Z_{layer}"] = cache[f"blocks.{layer}.attn.hook_z"][0, -1, :, :].cpu().float()
#
#         extracted_caches.append(mini_cache)
#
#         # Kill full cache
#         del cache, logits
#         if torch.backends.mps.is_available():
#             torch.mps.empty_cache()
#
#     if len(l25_acts) < 5:
#         print("  Not enough valid examples")
#         return
#
#     A25  = torch.stack(l25_acts)
#     y25  = torch.tensor(correct_logits_at_25)
#
#     A25_c  = A25 - A25.mean(0)
#     U, S, Vt = torch.linalg.svd(A25_c, full_matrices=False)
#
#     dir_r2_scores = []
#     for k in range(min(20, len(S))):
#         proj   = (A25_c @ Vt[k]).numpy()
#         corr   = abs(np.corrcoef(proj, y25.numpy())[0, 1])
#         dir_r2_scores.append((k, corr, Vt[k]))
#
#     dir_r2_scores.sort(key=lambda x: x[1], reverse=True)
#     top_answer_dirs = [d[2] for d in dir_r2_scores[:5]]
#     top_r2s         = [d[1] for d in dir_r2_scores[:5]]
#
#     component_scores = {}
#     n_heads = model.cfg.n_heads
#
#     for mini_cache in extracted_caches:
#         for layer in range(min(causal_layer + 1, model.cfg.n_layers)):
#             mlp_out = mini_cache[f"MLP{layer}"]
#             # Use Cosine Similarity to normalize magnitudes
#             mlp_score = sum(abs(torch.nn.functional.cosine_similarity(mlp_out, d.cpu(), dim=0).item()) * r for d, r in zip(top_answer_dirs, top_r2s))
#             component_scores.setdefault(f"MLP{layer}", []).append(mlp_score)
#
#             z = mini_cache[f"ATTN_Z_{layer}"]
#             for head in range(n_heads):
#                 # Get this specific head's W_O matrix and move to CPU to match 'z'
#                 W_O = model.blocks[layer].attn.W_O[head].cpu().float()
#                 head_out = z[head] @ W_O
#
#                 # Normalize head outputs as well
#                 head_score = sum(abs(torch.nn.functional.cosine_similarity(head_out, d.cpu(), dim=0).item()) * r for d, r in zip(top_answer_dirs, top_r2s))
#                 component_scores.setdefault(f"L{layer}H{head}", []).append(head_score)
#
#     avg_scores = {k: np.mean(v) for k, v in component_scores.items()}
#     ranked = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
#
#     print(f"\n  RANKED COMPONENTS by contribution to answer at L{causal_layer}:")
#     for comp, score in ranked[:15]:
#         bar = "█" * min(int(score * 15), 40)
#         print(f"  {comp:>12}: {score:8.4f}  {bar}")
#
#     return ranked

# ─────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────

# def run_laser_focus_analysis(model_name: str, causal_layer: int = 25):
#     device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
#
#     print(f"\n{'='*65}")
#     print(f"LASER-FOCUS ARITHMETIC ANALYSIS: {model_name} on {device.upper()}")
#     print(f"{'='*65}")
#
#     # Explicitly use float16. M1 does not natively accelerate bfloat16.
#     model = HookedTransformer.from_pretrained(
#         model_name, device=device, dtype=torch.float16
#     )
#     model.eval()
#
#     pairs = build_arithmetic_prompts(model, n_pairs=120)
#     print(f"  Built {len(pairs)} valid single-token arithmetic pairs")
#
#     print("\n" + "─"*50)
#     print("TOOL 1: Direct Logit Attribution")
#     print("─"*50)
#
#     # Adjusted layer range based on model size to avoid out-of-bounds
#     max_layer = model.cfg.n_layers
#     start_layer = max(0, max_layer - 13)
#
#     # We need to capture the returned values!
#     attn_dla, mlp_dla, embed_dla = direct_logit_attribution(model, pairs, layer_range=range(start_layer, max_layer))
#
#     # And we need to print them!
#     if attn_dla is not None:
#         print(f"\n  MLP DLA by layer (direct answer logit contribution):")
#         for layer in range(start_layer, max_layer):
#             val = mlp_dla[layer].item()
#             # Visualize the magnitude with a simple bar
#             bar = ("+" if val > 0 else "-") * min(int(abs(val)*3), 30)
#             print(f"    Layer {layer:2d} MLP: {val:+.4f}  {bar}")
#
#         print(f"\n  Top attention heads by DLA:")
#         head_scores = [
#             (layer, head, attn_dla[layer, head].item())
#             for layer in range(start_layer, max_layer)
#             for head in range(model.cfg.n_heads)
#         ]
#         # Sort to find the heads that directly write the most correct logits
#         head_scores.sort(key=lambda x: x[2], reverse=True)
#         for layer, head, score in head_scores[:10]:
#             bar = "+" * min(int(score*3), 30)
#             print(f"    L{layer:2d}H{head:2d}: {score:+.4f}  {bar}")
#
#     print("\n" + "─"*50)
#     print("TOOL 2: SVD Computational Signal Strength")
#     print("─"*50)
#     svd_computational_signal_strength(model, pairs, layer_range=range(start_layer, max_layer))
#
#     print("\n" + "─"*50)
#     print("TOOL 3: Backtracking from Causal Layer")
#     print("─"*50)
#     backtrack_from_causal_layer(model, pairs, causal_layer=min(causal_layer, max_layer - 1))
#
#     # Clear memory at the end of the run
#     del model
#     gc.collect()
#     if torch.backends.mps.is_available():
#         torch.mps.empty_cache()

# def generate_clean_corrupted_pairs(model, n_pairs=20, seed=100):
#     """Generates AB/AC pairs for edge patching."""
#     random.seed(seed)
#     pairs = []
#     attempts = 0
#
#     while len(pairs) < n_pairs:
#         attempts += 1
#         if attempts > 2000: break
#
#         a1, b1 = random.randint(10, 49), random.randint(10, 49)
#         ans_clean = a1 + b1
#
#         # Corrupted pair uses different numbers but same format
#         a2, b2 = random.randint(10, 49), random.randint(10, 49)
#         # Ensure answers are actually different
#         if ans_clean == a2 + b2: continue
#
#         prompt_clean = f"Math:\n10 + 10 = 20\n{a1} + {b1} ="
#         prompt_corr  = f"Math:\n10 + 10 = 20\n{a2} + {b2} ="
#
#         try:
#             # Safely get the target token IDs
#             clean_toks = model.to_tokens(f" {ans_clean}", prepend_bos=False)[0]
#             corr_toks  = model.to_tokens(f" {a2 + b2}", prepend_bos=False)[0]
#
#             if len(clean_toks) > 0 and len(corr_toks) > 0:
#                 pairs.append({
#                     'clean': prompt_clean,
#                     'corr': prompt_corr,
#                     'clean_ans': clean_toks[0].item() # The target we want to recover
#                 })
#         except Exception:
#             continue
#
#     return pairs

# import torch
# import gc
# from transformer_lens import HookedTransformer
#
# def run_group_edge_patching(model, prompt_pairs, head_group, candidate_mlps):
#     """
#     Tests if a GROUP of attention heads collectively routes information to target MLPs.
#     """
#     print(f"\n{'='*65}")
#     print("RUNNING GROUP EDGE PATCHING (DISTRIBUTED CIRCUIT TEST)")
#     print(f"{'='*65}")
#
#     results = {mlp: 0.0 for mlp in candidate_mlps}
#
#     unique_layers = list(set([l for l, h in head_group]))
#     max_src_layer = max(unique_layers) if unique_layers else -1
#
#     print(f"  Testing Group of {len(head_group)} Heads: {head_group}")
#
#     for pair in prompt_pairs[:20]:
#         clean_toks = model.to_tokens(pair['clean'])
#         corr_toks  = model.to_tokens(pair['corr'])
#         clean_ans  = pair['clean_ans']
#
#         # ── Step 1: Baselines ──
#         with torch.no_grad():
#             clean_prob = torch.softmax(model(clean_toks)[0, -1, :], dim=-1)[clean_ans].item()
#             corr_prob  = torch.softmax(model(corr_toks)[0, -1, :], dim=-1)[clean_ans].item()
#
#         if torch.backends.mps.is_available(): torch.mps.empty_cache()
#
#         # ── Step 2: Collect all head activations efficiently ──
#         clean_z_cache = {}
#         corr_z_cache = {}
#
#         # THE FIX: Changed 'hook_obj' to '**kwargs' to safely absorb the TransformerLens keyword arguments
#         def make_clean_hook(layer_idx):
#             def hook_fn(z, **kwargs):
#                 clean_z_cache[layer_idx] = z[0, -1, :, :].cpu().clone()
#                 return z
#             return hook_fn
#
#         def make_corr_hook(layer_idx):
#             def hook_fn(z, **kwargs):
#                 corr_z_cache[layer_idx] = z[0, -1, :, :].cpu().clone()
#                 return z
#             return hook_fn
#
#         clean_hooks = [(f"blocks.{l}.attn.hook_z", make_clean_hook(l)) for l in unique_layers]
#         corr_hooks  = [(f"blocks.{l}.attn.hook_z", make_corr_hook(l)) for l in unique_layers]
#
#         with torch.no_grad():
#             model.run_with_hooks(clean_toks, fwd_hooks=clean_hooks)
#             model.run_with_hooks(corr_toks, fwd_hooks=corr_hooks)
#
#         # ── Step 3: Combine the group into a single vector ──
#         clean_group_vec = torch.zeros(model.cfg.d_model, device=model.cfg.device)
#         corr_group_vec  = torch.zeros(model.cfg.d_model, device=model.cfg.device)
#
#         for l, h in head_group:
#             W_O = model.blocks[l].attn.W_O[h]
#             clean_z_head = clean_z_cache[l][h].to(model.cfg.device)
#             corr_z_head  = corr_z_cache[l][h].to(model.cfg.device)
#
#             clean_group_vec += clean_z_head @ W_O
#             corr_group_vec  += corr_z_head @ W_O
#
#         del clean_z_cache, corr_z_cache
#         if torch.backends.mps.is_available(): torch.mps.empty_cache()
#
#         # ── Step 4: Patch the combined group into the target MLPs ──
#         for tgt_layer in candidate_mlps:
#             if tgt_layer <= max_src_layer:
#                 continue
#
#             mlp_in_hook_name = f"blocks.{tgt_layer}.hook_resid_pre"
#
#             # THE FIX: Changed 'hook_obj' to '**kwargs' here too
#             def group_patch_hook(resid_pre, **kwargs):
#                 resid_pre[0, -1, :] = resid_pre[0, -1, :] - corr_group_vec + clean_group_vec
#                 return resid_pre
#
#             with torch.no_grad():
#                 patched_logits = model.run_with_hooks(
#                     corr_toks,
#                     fwd_hooks=[(mlp_in_hook_name, group_patch_hook)]
#                 )
#
#             patched_prob = torch.softmax(patched_logits[0, -1, :], dim=-1)[clean_ans].item()
#             recovery = (patched_prob - corr_prob) / (clean_prob - corr_prob + 1e-8)
#             results[tgt_layer] += recovery
#
#             del patched_logits
#             if torch.backends.mps.is_available(): torch.mps.empty_cache()
#
#         del clean_group_vec, corr_group_vec
#         if torch.backends.mps.is_available(): torch.mps.empty_cache()
#
#     # ── Print Results ──
#     num_pairs = min(20, len(prompt_pairs))
#     print("\n  Group Arithmetic Edges (Information Flow):")
#     print(f"  {'Source Group'} -> {'Target MLP':<12} | {'Recovery %':>10}")
#     print("  " + "-"*45)
#
#     for tgt in candidate_mlps:
#         if tgt <= max_src_layer:
#             print(f"  [ALL HEADS] -> MLP {tgt:02d}       | SKIPPED (Causality)")
#             continue
#
#         avg_rec = (results[tgt] / num_pairs) * 100
#         bar = "█" * min(int(max(0, avg_rec) / 2), 20)
#         print(f"  [ALL HEADS] -> MLP {tgt:02d}       | {avg_rec:6.1f}%  {bar}")
#
#     return results

# print("\n\n" + "#"*65)
# print("PHASE 2: GROUP PATH PATCHING (EDGE TESTING)")
# print("#"*65)
#
# device = "mps" if torch.backends.mps.is_available() else "cpu"
# model = HookedTransformer.from_pretrained(
#     "google/gemma-2b", device=device, dtype=torch.float16
# )
# model.eval()
#
# patching_pairs = generate_clean_corrupted_pairs(model, n_pairs=20)
#
# # We group all the early and mid-layer heads that showed up in Tools 1 & 3
# early_router_team = [
#     (0, 0), (0, 2), (0, 3), (0, 4), (0, 6), # L0 positional/embedding readers
#     (6, 5), (6, 7),                         # L6 active readers
#     (9, 0),                                 # L9 active reader
#     (10, 3)                                 # L10 active reader
# ]
#
# # The heavy-lifting MLPs (and the Eraser)
# top_math_mlps = [11, 12, 13, 14, 15, 16]
#
# run_group_edge_patching(
#     model=model,
#     prompt_pairs=patching_pairs,
#     head_group=early_router_team,
#     candidate_mlps=top_math_mlps
# )
#

# -----------------------------------------------------------------------
# 1. PROMPT GENERATION (Using the "Tens-Digit Spoonfeed" Method)
# -----------------------------------------------------------------------
def get_ones_token_safe(model, ans: int):
    ones_digit = str(ans)[-1]
    try:
        tok_ids = model.to_tokens(ones_digit, prepend_bos=False)[0]
        return tok_ids[-1].item(), True
    except Exception:
        return None, False

def build_circuit_prompts(model, n_pairs=50, seed=42):
    random.seed(seed)
    pairs = []
    attempts = 0

    while len(pairs) < n_pairs and attempts < n_pairs * 10:
        attempts += 1
        a = random.randint(10, 49)
        b = random.randint(10, 49)
        ans = a + b

        tok, valid = get_ones_token_safe(model, ans)
        if not valid: continue

        tens_digit = str(ans)[0]
        prompt = f"Math:\n10 + 10 = 20\n21 + 13 = 34\n{a} + {b} = {tens_digit}"

        pairs.append({
            'a': a, 'b': b, 'ans': ans,
            'ones_digit': int(str(ans)[-1]),
            'prompt': prompt,
            'ans_tok': tok
        })

    return pairs

# -----------------------------------------------------------------------
# 2. THE UNIFIED CIRCUIT DISCOVERY PIPELINE
# -----------------------------------------------------------------------
def run_circuit_discovery(model_name: str, target_layer: int = 25, n_pairs: int = 50):
    device = ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*70}")
    print(f"UNIFIED CIRCUIT DISCOVERY: {model_name}")
    print(f"{'='*70}")

    model = HookedTransformer.from_pretrained(model_name, device=device, dtype=torch.bfloat16)
    model.eval()

    print(f"  Building {n_pairs} carefully formatted prompts...")
    pairs = build_circuit_prompts(model, n_pairs=n_pairs)
    if not pairs:
        print("  ❌ Failed to build prompts. Exiting.")
        return

    tokens = model.to_tokens([p['prompt'] for p in pairs])
    ans_toks = torch.tensor([p['ans_tok'] for p in pairs], device=device)
    ones_targets = np.array([p['ones_digit'] for p in pairs])

    # -----------------------------------------------------------------------
    # THE MEMORY OPTIMIZATION: Only cache exactly what we need
    # -----------------------------------------------------------------------
    def cache_filter(name):
        return name.endswith("hook_attn_out") or name.endswith("hook_mlp_out") or name == f"blocks.{target_layer}.hook_resid_post"

    print(f"  Running Unified Forward Pass (Caching only critical nodes)...")
    start_time = time.time()
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=cache_filter)
    print(f"  Forward pass complete in {time.time() - start_time:.1f}s. Memory safe.")

    # Get the Unembedding Matrix for Tool 1
    W_U = model.W_U # [d_model, vocab_size]

    # Get the Empirical Answer Direction at Target Layer for Tool 3
    print(f"\n  Calculating Empirical Answer Direction at Layer {target_layer}...")
    target_resid = cache[f"blocks.{target_layer}.hook_resid_post"][:, -1, :].cpu().float().numpy()

    # Linear probing to find the vector that predicts the ones digit
    reg = LinearRegression().fit(target_resid, ones_targets)
    empirical_dir = torch.tensor(reg.coef_, device=device, dtype=torch.float32)
    empirical_dir = empirical_dir / torch.norm(empirical_dir) # Normalize

    print(f"\n{'='*70}")
    print(f"{'Layer':<6} | {'MLP DLA':<10} | {'Attn DLA':<10} | {'MLP Proj':<10} | {'Attn Proj':<10} | {'R^2 (SVD)':<10}")
    print(f"{'-'*70}")

    results = {}

    # Analyze layer by layer
    for layer in range(model.cfg.n_layers):
        mlp_name = f"blocks.{layer}.hook_mlp_out"
        attn_name = f"blocks.{layer}.hook_attn_out"

        if mlp_name not in cache or attn_name not in cache:
            continue

        mlp_out = cache[mlp_name][:, -1, :].float()  # [batch, d_model]
        attn_out = cache[attn_name][:, -1, :].float() # [batch, d_model]

        # -------------------------------------------------------------------
        # TOOL 1: Direct Logit Attribution (DLA)
        # Dot product of the component output with the unembedding vector of the correct answer
        # -------------------------------------------------------------------
        mlp_dla = (mlp_out * W_U[:, ans_toks].T).sum(dim=-1).mean().item()
        attn_dla = (attn_out * W_U[:, ans_toks].T).sum(dim=-1).mean().item()

        # -------------------------------------------------------------------
        # TOOL 3: Empirical Backtracking (Projection)
        # Dot product of the component output with the empirical answer vector from Layer 25
        # -------------------------------------------------------------------
        mlp_proj = (mlp_out @ empirical_dir).mean().item()
        attn_proj = (attn_out @ empirical_dir).mean().item()

        # -------------------------------------------------------------------
        # TOOL 2: Computational Signal (R^2 SVD)
        # How much variance in the answer does this layer's addition explain?
        # -------------------------------------------------------------------
        delta_resid = (mlp_out + attn_out).cpu().numpy()
        layer_reg = LinearRegression().fit(delta_resid, ones_targets)
        r2_score = layer_reg.score(delta_resid, ones_targets)

        results[layer] = {
            'mlp_dla': mlp_dla, 'attn_dla': attn_dla,
            'mlp_proj': mlp_proj, 'attn_proj': attn_proj,
            'r2': r2_score
        }

        # Highlight significant layers
        marker = "🚀" if r2_score > 0.4 or mlp_dla > 1.0 else "  "

        print(f"L{layer:<4} | {mlp_dla:>9.3f} | {attn_dla:>9.3f} | {mlp_proj:>9.3f} | {attn_proj:>9.3f} | {r2_score:>8.3f} {marker}")

    # Flush memory
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return results


# ─────────────────────────────────────────────────────────────
# FIX: Token-agnostic prompt building
# Instead of requiring single-token answers, we:
# 1. For full arithmetic: always use the ones-digit as the
#    measurement target, reached via autoregressive generation
# 2. Accept multi-digit answers by measuring at the correct step
# ─────────────────────────────────────────────────────────────

def get_token_id(model, text: str):
    """Returns token id for a single character/word, or None."""
    try:
        toks = model.to_tokens(text, prepend_bos=False)[0]
        if len(toks) >= 1:
            return toks[-1].item()
    except Exception:
        pass
    return None


def build_prompts_robust(model, n_pairs=50, seed=42):
    """
    Builds prompts that work regardless of tokenizer.

    FULL FORMAT: "Math:...{a} + {b} ="
      Model generates full answer autoregressively.
      We cache at generation step 1 (first digit) AND
      check what step reaches the ones digit.

    ONES FORMAT: "Math:...{a} + {b} = {hundreds+tens}"
      Model predicts the ones digit directly.
      Works for all answer lengths.

    Both formats use the ONES DIGIT as the measurement target
    because that is the hardest digit to predict (carries affect it).
    """
    random.seed(seed)
    full_pairs, ones_pairs = [], []
    attempts = 0

    while ((len(full_pairs) < n_pairs or len(ones_pairs) < n_pairs)
           and attempts < n_pairs * 30):
        attempts += 1
        a   = random.randint(10, 49)
        b   = random.randint(10, 49)
        ans = a + b    # range: 20-98

        ans_str       = str(ans)          # e.g. "57"
        ones_digit    = int(ans_str[-1])  # e.g. 7
        ones_char     = ans_str[-1]       # e.g. "7"
        prefix_digits = ans_str[:-1]      # e.g. "5"

        # Get token id for the ones digit
        ones_tok = get_token_id(model, ones_char)
        if ones_tok is None:
            continue

        # ── Full arithmetic prompt ──
        # We feed: "Math:...{a} + {b} = {prefix_digits}"
        # and measure prediction of ones digit
        # This works whether ans is 2 or 3 digits
        if len(full_pairs) < n_pairs:
            full_prompt = (f"Math:\n10 + 10 = 20\n21 + 13 = 34\n"
                           f"{a} + {b} = {prefix_digits}")
            full_pairs.append({
                'a': a, 'b': b, 'ans': ans,
                'ones_digit': ones_digit,
                'prompt': full_prompt,
                'ans_tok': ones_tok,
                'format': 'full'
            })

        # ── Ones-digit prompt (same structure for comparison) ──
        if len(ones_pairs) < n_pairs:
            ones_prompt = (f"Math:\n10 + 10 = 20\n21 + 13 = 34\n"
                           f"{a} + {b} = {prefix_digits}")
            ones_pairs.append({
                'a': a, 'b': b, 'ans': ans,
                'ones_digit': ones_digit,
                'prompt': ones_prompt,
                'ans_tok': ones_tok,
                'format': 'ones'
            })

    print(f"  Built: {len(full_pairs)} full-arith, "
          f"{len(ones_pairs)} ones-hint pairs")
    return full_pairs, ones_pairs


# ─────────────────────────────────────────────────────────────
# CORE DLA ANALYSIS
# All three professor fixes + R² overfitting fix retained
# ─────────────────────────────────────────────────────────────

def run_dla_analysis(model, pairs, label=""):
    if not pairs:
        print(f"  {label}: no pairs available")
        return {}

    device = next(model.parameters()).device
    W_U    = model.W_U.detach().to(device).float()   # professor fix 1

    tokens   = model.to_tokens([p['prompt'] for p in pairs])
    ans_toks = torch.tensor([p['ans_tok'] for p in pairs], device=device)
    ones_tgt = np.array([p['ones_digit'] for p in pairs])

    def cache_filter(name):
        return (name.endswith("hook_attn_out") or
                name.endswith("hook_mlp_out")  or
                name == f"blocks.{model.cfg.n_layers-1}.hook_resid_post")

    print(f"\n  [{label}] Running forward pass...")
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=cache_filter)

    # Empirical answer direction
    final_key = f"blocks.{model.cfg.n_layers-1}.hook_resid_post"
    emp_dir   = None
    if final_key in cache:
        final_np = cache[final_key][:, -1, :].cpu().float().numpy()
        reg      = Ridge(alpha=1.0).fit(final_np, ones_tgt)
        emp_dir  = F.normalize(                          # professor fix 3
            torch.tensor(reg.coef_, dtype=torch.float32, device=device),
            p=2, dim=0
        )

    # R² pipeline with PCA to avoid overfitting
    n_components = min(20, len(pairs) - 2, 40)
    r2_pipe = Pipeline([
        ('sc',  StandardScaler()),
        ('pca', PCA(n_components=n_components)),
        ('reg', Ridge(alpha=1.0))
    ])

    print(f"\n  [{label}] Layer-by-layer analysis:")
    print(f"  {'L':>4}  {'MLP_DLA':>10}  {'Attn_DLA':>10}  "
          f"{'MLP_Proj':>9}  {'Attn_Proj':>9}  {'R2_CV':>7}")
    print("  " + "-"*68)

    results = {}

    for layer in range(model.cfg.n_layers):
        mlp_key  = f"blocks.{layer}.hook_mlp_out"
        attn_key = f"blocks.{layer}.hook_attn_out"
        if mlp_key not in cache or attn_key not in cache:
            continue

        mlp_out  = cache[mlp_key][:, -1, :].float()
        attn_out = cache[attn_key][:, -1, :].float()

        # DLA
        mlp_dla  = (mlp_out  * W_U[:, ans_toks].T).sum(-1).mean().item()
        attn_dla = (attn_out * W_U[:, ans_toks].T).sum(-1).mean().item()

        # Empirical projection
        mlp_proj  = (mlp_out  @ emp_dir).mean().item() if emp_dir is not None else 0.0
        attn_proj = (attn_out @ emp_dir).mean().item() if emp_dir is not None else 0.0

        # Cross-validated R²
        delta = (mlp_out + attn_out).cpu().float().numpy()
        cv_r2 = 0.0
        if len(ones_tgt) >= 15:
            try:
                cv_r2 = float(max(0.0, cross_val_score(
                    r2_pipe, delta, ones_tgt,
                    cv=min(5, len(ones_tgt)//5), scoring='r2'
                ).mean()))
            except Exception:
                cv_r2 = 0.0

        results[layer] = dict(
            mlp_dla=mlp_dla, attn_dla=attn_dla,
            mlp_proj=mlp_proj, attn_proj=attn_proj,
            r2_cv=cv_r2
        )

        total = abs(mlp_dla) + abs(attn_dla)
        flag  = ("🔥 DOMINANT"   if total > 20   else
                 "✅ Active"      if total > 5    else
                 "📊 Predictive" if cv_r2 > 0.30 else "")

        print(f"  L{layer:<3}  {mlp_dla:>10.3f}  {attn_dla:>10.3f}  "
              f"{mlp_proj:>9.3f}  {attn_proj:>9.3f}  {cv_r2:>7.3f}  {flag}")

    # Professor fix 2: explicit memory cleanup
    del cache
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return results


# ─────────────────────────────────────────────────────────────
# SYNTHESIS: Map the three zones
# ─────────────────────────────────────────────────────────────

def synthesize_results(results: dict, model_name: str):
    """
    Identifies the three functional zones from the results:
      Zone 1 — Representation (R²=0, low DLA): early layers
      Zone 2 — Computation   (rising R², moderate DLA): middle
      Zone 3 — Translation   (high DLA, high R²): final layers
    """
    if not results:
        return

    layers  = sorted(results.keys())
    r2s     = [results[l]['r2_cv'] for l in layers]
    dlas    = [abs(results[l]['mlp_dla']) +
               abs(results[l]['attn_dla']) for l in layers]

    # Find zone boundaries
    computation_start = next(
        (l for l, r2 in zip(layers, r2s) if r2 > 0.15), None
    )
    translation_start = next(
        (l for l, d in zip(layers, dlas) if d > 20), None
    )

    print(f"\n{'='*60}")
    print(f"CIRCUIT MAP: {model_name.split('/')[-1]}")
    print(f"{'='*60}")
    print(f"\n  Zone 1 — REPRESENTATION: Layers 0 – {(computation_start or 0) - 1}")
    print(f"    R²=0, low DLA. Numbers stored geometrically (helix)")
    print(f"    but no linearly decodable arithmetic answer yet.")
    print()

    if computation_start and translation_start:
        print(f"  Zone 2 — COMPUTATION: Layers {computation_start} – "
              f"{translation_start - 1}")
        print(f"    R² rises from 0 to ~0.9. This is where arithmetic")
        print(f"    is actually computed and assembled into the residual stream.")
        print(f"    Key neurons (carry detectors, magnitude trackers) operate here.")
        print()

    if translation_start:
        print(f"  Zone 3 — TRANSLATION: Layers {translation_start}+ ")
        print(f"    DLA spikes. R² ≈ 1. Final MLP layers project the")
        print(f"    internal arithmetic answer to vocabulary token logits.")
        print(f"    This is what your earlier massive DLA values showed.")
        print()

    # Identify the single most important computation layer
    comp_layers = [l for l in layers
                   if (results[l]['r2_cv'] > 0.15 and
                       abs(results[l]['mlp_dla']) +
                       abs(results[l]['attn_dla']) < 20)]

    if comp_layers:
        best_comp = max(comp_layers,
                        key=lambda l: results[l]['r2_cv'])
        print(f"  Key computation layer: L{best_comp}  "
              f"(R²={results[best_comp]['r2_cv']:.3f})")
        print(f"  This is where arithmetic crystallises.")
        print(f"  Run head-level DLA at this layer to find")
        print(f"  the specific attention heads doing the work.")


def compare_formats(model_name: str):
    device = ("mps"  if torch.backends.mps.is_available() else
              "cuda" if torch.cuda.is_available()         else "cpu")

    print(f"\n{'='*70}")
    print(f"CIRCUIT DISCOVERY: {model_name}")
    print(f"{'='*70}")

    model = HookedTransformer.from_pretrained(
        model_name, device=device, dtype=torch.bfloat16
    )
    model.eval()

    full_pairs, _ = build_prompts_robust(model, n_pairs=50)
    results       = run_dla_analysis(model, full_pairs, label="ARITHMETIC")
    synthesize_results(results, model_name)

    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

# -----------------------------------------------------------------------
# PROMPT BUILDER (Brought in-house to make script self-contained)
# -----------------------------------------------------------------------
def get_token_id(model, text: str):
    try:
        toks = model.to_tokens(text, prepend_bos=False)[0]
        if len(toks) >= 1:
            return toks[-1].item()
    except Exception:
        pass
    return None

def build_prompts_robust(model, n_pairs=50, seed=42):
    random.seed(seed)
    full_pairs = []
    attempts = 0
    while len(full_pairs) < n_pairs and attempts < n_pairs * 30:
        attempts += 1
        a = random.randint(10, 49)
        b = random.randint(10, 49)
        ans = a + b
        ans_str = str(ans)
        ones_digit = int(ans_str[-1])
        ones_char = ans_str[-1]
        prefix_digits = ans_str[:-1]

        ones_tok = get_token_id(model, ones_char)
        if ones_tok is None:
            continue

        full_prompt = (f"Math:\n10 + 10 = 20\n21 + 13 = 34\n"
                       f"{a} + {b} = {prefix_digits}")

        full_pairs.append({
            'a': a, 'b': b, 'ans': ans,
            'ones_digit': ones_digit,
            'prompt': full_prompt,
            'ans_tok': ones_tok,
        })
    return full_pairs

# -----------------------------------------------------------------------
# HEAD-LEVEL DLA DRILL DOWN
# -----------------------------------------------------------------------
def head_level_dla(model, pairs, computation_layer: int):
    device = next(model.parameters()).device
    W_U = model.W_U.detach().to(device).float()
    n_heads = model.cfg.n_heads

    tokens = model.to_tokens([p['prompt'] for p in pairs])
    ans_toks = torch.tensor([p['ans_tok'] for p in pairs], device=device)
    ones_tgt = np.array([p['ones_digit'] for p in pairs])

    # 1. THE FIX: Hook 'z' (the pre-weight head values) instead of 'result'
    hook_z_name = f"blocks.{computation_layer}.attn.hook_z"
    hook_mlp = f"blocks.{computation_layer}.hook_mlp_out"

    def cache_filter(name):
        return name in [hook_z_name, hook_mlp]

    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, names_filter=cache_filter)

    mlp_out = cache[hook_mlp][:, -1, :].float()
    mlp_dla = (mlp_out * W_U[:, ans_toks].T).sum(-1).mean().item()

    head_results = {}
    r2_pipe = Pipeline([
        ('sc', StandardScaler()),
        ('pca', PCA(n_components=min(10, len(pairs)-2))),
        ('reg', Ridge(alpha=1.0))
    ])

    print(f"\n  Head-level analysis at Layer {computation_layer}:")
    print(f"  {'Head':>5}  {'DLA':>9}  {'R2_CV':>7}  Role")
    print("  " + "-"*45)

    # Grab the Output Weights for this specific layer
    W_O = model.blocks[computation_layer].attn.W_O.detach().float() # [n_heads, d_head, d_model]

    for head in range(n_heads):
        # 2. THE FIX: Reconstruct the head output manually!
        # z_head shape: [batch, d_head]
        z_head = cache[hook_z_name][:, -1, head, :].float()

        # Multiply z by the Output matrix for this specific head
        # Result shape: [batch, d_model]
        head_out_batch = z_head @ W_O[head]

        # DLA for this head
        head_dla = (head_out_batch * W_U[:, ans_toks].T).sum(-1).mean().item()

        # R² for this head
        head_np = head_out_batch.cpu().numpy()
        cv_r2 = 0.0
        if len(ones_tgt) >= 15:
            try:
                cv_r2 = float(max(0.0, cross_val_score(
                    r2_pipe, head_np, ones_tgt, cv=5, scoring='r2'
                ).mean()))
            except Exception:
                cv_r2 = 0.0

        head_results[head] = {'dla': head_dla, 'r2': cv_r2}

        flag = ("🔥 KEY HEAD" if abs(head_dla) > 2 and cv_r2 > 0.2 else
                "📊 Predictive" if cv_r2 > 0.2 else
                "✅ DLA active" if abs(head_dla) > 1 else "")

        print(f"  H{head:<4}  {head_dla:>9.3f}  {cv_r2:>7.3f}  {flag}")

    print(f"\n  MLP L{computation_layer}: DLA={mlp_dla:.3f}")

    key_heads = [
        (h, r['dla'], r['r2'])
        for h, r in head_results.items()
        if abs(r['dla']) > 1.0 and r['r2'] > 0.15
    ]
    key_heads.sort(key=lambda x: abs(x[1]), reverse=True)

    if key_heads:
        print(f"\n  KEY HEADS (both high DLA and high R²):")
        for h, dla, r2 in key_heads:
            print(f"    Head {h:2d}: DLA={dla:+.3f}, R²={r2:.3f}")
            print(f"    → This head is COMPUTING arithmetic at L{computation_layer}")
    else:
        print(f"\n  No single head dominates. Arithmetic is distributed")
        print(f"  across many heads at this layer — consistent with")
        print(f"  Phi-3's distributed computation finding.")

    del cache
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return head_results

# -----------------------------------------------------------------------
# EXECUTION
# -----------------------------------------------------------------------
def run_head_drilldown(model_name: str, computation_layer: int, n_pairs: int = 50):
    device = ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"HEAD-LEVEL DRILL-DOWN: {model_name}")
    print(f"Target: Layer {computation_layer}")
    print(f"{'='*60}")

    # THE FIX: Added use_attn_result=True so TransformerLens stores individual head outputs
    model = HookedTransformer.from_pretrained(
        model_name,
        device=device,
        dtype=torch.bfloat16
    )
    model.eval()

    # # 2. THE REAL FIX: Turn on individual head tracking safely!
    # model.set_use_attn_result(True)

    full_pairs = build_prompts_robust(model, n_pairs=n_pairs)
    head_level_dla(model, full_pairs, computation_layer)

    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

def get_token_id(model, text: str):
    try:
        toks = model.to_tokens(text, prepend_bos=False)[0]
        if len(toks) >= 1: return toks[-1].item()
    except Exception: pass
    return None

def build_hard_prompts_ones_target(model, n_pairs=50, seed=42):
    random.seed(seed)
    full_pairs = []
    attempts = 0

    # 3-digit arithmetic to strain the context window and routing
    while len(full_pairs) < n_pairs and attempts < n_pairs * 30:
        attempts += 1
        a, b = random.randint(150, 499), random.randint(150, 499)
        ans = a + b
        ans_str = str(ans)

        # We target the ONES digit (the hardest, carry-sensitive digit)
        ones_tok = get_token_id(model, ans_str[-1])
        if ones_tok is None: continue

        full_pairs.append({
            'a': a, 'b': b, 'ans': ans,
            'ones_digit': int(ans_str[-1]),
            # Prompt feeds the hundreds and tens, asks for the ones
            'prompt': f"Math:\n105 + 112 = 217\n231 + 143 = 374\n{a} + {b} = {ans_str[:-1]}",
            'ans_tok': ones_tok,
        })

    print(f"  Built {len(full_pairs)} hard 3-digit test pairs (targeting ones digit).")
    return full_pairs

def make_ablation_hook(heads_to_zero: list):
    """Factory function to capture heads_to_zero by value."""
    def hook_fn(value, hook):
        for h in heads_to_zero:
            value[:, -1, h, :] = 0.0
        return value
    return hook_fn

def ablate_hard_arithmetic(model, pairs, layer: int, key_heads: list):
    device = next(model.parameters()).device
    tokens = model.to_tokens([p['prompt'] for p in pairs])
    ans_toks = torch.tensor([p['ans_tok'] for p in pairs], device=device)
    hook_name = f"blocks.{layer}.attn.hook_z"

    print(f"\n{'='*65}")
    print(f"HARD ABLATION: Layer {layer} Attention Heads")
    print(f"Task: 3-Digit Addition (Predicting Ones Digit)")
    print(f"Key heads under test: {key_heads}")
    print(f"{'='*65}")
    print(f"\n  {'Ablation':>22} | {'P(correct)':>11} | {'Drop':>8} | Verdict")
    print("  " + "-"*58)

    # 1. Baseline
    with torch.no_grad():
        base_logits = model(tokens)

    base_probs = torch.softmax(base_logits[:, -1, :], dim=-1)
    base_ans_prob = base_probs[torch.arange(len(pairs)), ans_toks].mean().item()
    print(f"  {'Baseline (no ablation)':>22} | {base_ans_prob:>10.2%} | {'---':>8} |")

    del base_logits, base_probs
    if torch.backends.mps.is_available(): torch.mps.empty_cache()

    # 2. Individual heads + all together
    ablation_targets = [(h, f"Head {h}") for h in key_heads]
    ablation_targets.append((key_heads, f"All {len(key_heads)} heads"))
    results = {}

    for target, label in ablation_targets:
        heads = [target] if isinstance(target, int) else target

        with torch.no_grad():
            ablated_logits = model.run_with_hooks(
                tokens,
                fwd_hooks=[(hook_name, make_ablation_hook(heads))]
            )

        ablated_probs = torch.softmax(ablated_logits[:, -1, :], dim=-1)
        ablated_ans_prob = ablated_probs[torch.arange(len(pairs)), ans_toks].mean().item()
        prob_drop = base_ans_prob - ablated_ans_prob

        verdict = ("🔥 CATASTROPHIC" if prob_drop > 0.30 else
                   "🚨 NECESSARY" if prob_drop > 0.15 else
                   "✅ Contributes" if prob_drop > 0.05 else
                   "○ Redundant" if prob_drop > -0.02 else
                   "⚠️ Harmful")

        print(f"  {'Ablate ' + label:>22} | {ablated_ans_prob:>10.2%} | {prob_drop:>+8.2%} | {verdict}")
        results[label] = {'ablated_prob': ablated_ans_prob, 'prob_drop': prob_drop}

        del ablated_logits, ablated_probs
        if torch.backends.mps.is_available(): torch.mps.empty_cache()

    # 3. Synthesis
    print(f"\n  SUMMARY:")
    all_result = results.get(f"All {len(key_heads)} heads", {})
    if all_result:
        total_drop = all_result['prob_drop']
        individual_sum = sum(r['prob_drop'] for l, r in results.items() if 'All' not in l)

        print(f"  Total drop (all ablated): {total_drop:+.2%}")
        print(f"  Sum of individual drops:  {individual_sum:+.2%}")

        if total_drop > individual_sum * 1.2:
            print("  → Heads are SYNERGISTIC. The circuit shatters without them.")
        elif total_drop > individual_sum * 0.8:
            print("  → Heads are INDEPENDENT. Each contributes separately to complex math.")
        else:
            print("  → Heads are still REDUNDANT. The model finds another way.")

    return results

# def run_final_experiment():
#     device = ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
#     print(f"\nLoading Gemma-7B...")
#     model = HookedTransformer.from_pretrained("google/gemma-7b", device=device, dtype=torch.bfloat16)
#     model.eval()
#
#     pairs = build_hard_prompts_ones_target(model, n_pairs=50)
#
#     # Testing the dominant heads we found at Layer 25
#     ablate_hard_arithmetic(model, pairs, layer=22, key_heads=[2, 4, 6, 8, 10, 12])
#
#     del model
#     if torch.backends.mps.is_available(): torch.mps.empty_cache()

# def run_final_experiment():
#     device = ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
#     print(f"\nLoading Gemma-7B...")
#     model = HookedTransformer.from_pretrained("google/gemma-7b", device=device, dtype=torch.bfloat16)
#     model.eval()
#
#     pairs = build_prompts_robust(model, n_pairs=50)
#     print(f"Built {len(pairs)} test pairs")
#
#     ablate_key_heads_sequentially(model, pairs, layer=25, key_heads=[2, 4, 8, 12])
#
#     del model
#     if torch.backends.mps.is_available(): torch.mps.empty_cache()


# -----------------------------------------------------------------------
# PROMPT BUILDER: Genuine Hard Task (No Hints)
# -----------------------------------------------------------------------
def build_pure_math_prompts(digits=3, n_pairs=50, seed=42):
    random.seed(seed)
    pairs = []

    min_val = 10**(digits-1)
    max_val = (10**digits) - 1

    for _ in range(n_pairs):
        a = random.randint(min_val, max_val)
        b = random.randint(min_val, max_val)
        ans = str(a + b)

        # Zero hints. Just the equation.
        prompt = f"Math:\n105 + 112 = 217\n231 + 143 = 374\n{a} + {b} ="

        pairs.append({
            'a': a, 'b': b, 'ans': ans,
            'prompt': prompt
        })

    print(f"  Built {len(pairs)} pure {digits}-digit test pairs.")
    return pairs

# -----------------------------------------------------------------------
# AUTOREGRESSIVE EVALUATION
# -----------------------------------------------------------------------
def evaluate_generation(model, pairs, fwd_hooks=None):
    """Generates the answer autoregressively and checks for an exact match."""
    correct = 0

    for p in pairs:
        tokens = model.to_tokens(p['prompt'])

        # Generate enough tokens for the answer plus a buffer space
        max_tokens_needed = len(p['ans']) + 2

        with torch.no_grad():
            # THE FIX: We must use the context manager for generation hooks!
            if fwd_hooks:
                with model.hooks(fwd_hooks=fwd_hooks):
                    output_tokens = model.generate(
                        tokens,
                        max_new_tokens=max_tokens_needed,
                        verbose=False
                    )
            else:
                output_tokens = model.generate(
                    tokens,
                    max_new_tokens=max_tokens_needed,
                    verbose=False
                )

        # Decode only the newly generated tokens
        generated_str = model.tokenizer.decode(output_tokens[0][tokens.shape[1]:])

        # Strip spaces from both strings to ensure exact mathematical match
        if generated_str.strip().startswith(p['ans'].strip()):
            correct += 1

    return correct / len(pairs)

# -----------------------------------------------------------------------
# CAUSAL ABLATION (The Scalpel)
# -----------------------------------------------------------------------
def make_ablation_hook(heads_to_zero: list):
    """Factory function to safely capture heads by value."""
    def hook_fn(value, hook):
        for h in heads_to_zero:
            # FIXED: Only zero out the final token position (routing), not all positions (reading)
            value[:, -1, h, :] = 0.0
        return value
    return hook_fn

def run_edge_of_competence_test(model, digits, layer, key_heads, n_pairs=50):
    print(f"\n{'='*65}")
    print(f"STRESS TEST: {digits}-DIGIT ADDITION (Autoregressive Exact Match)")
    print(f"{'='*65}")

    pairs = build_pure_math_prompts(digits=digits, n_pairs=n_pairs)
    hook_name = f"blocks.{layer}.attn.hook_z"

    # 1. Baseline
    print("  Evaluating Baseline (No Ablation)...")
    # FIXED: Explicitly pass None for clarity
    base_accuracy = evaluate_generation(model, pairs, fwd_hooks=None)
    print(f"  Baseline Accuracy: {base_accuracy:>10.2%}\n")

    if base_accuracy < 0.20:
        print("  ⚠️ Baseline is too low. The model fundamentally cannot do this task.")
        print("  Ablation results will be uninformative.\n")
        return

    # 2. Ablate All Key Heads
    print(f"  Ablating Key Heads {key_heads} at Layer {layer}...")
    ablated_accuracy = evaluate_generation(
        model,
        pairs,
        fwd_hooks=[(hook_name, make_ablation_hook(key_heads))]
    )

    drop = base_accuracy - ablated_accuracy

    marker = "🔥 NECESSARY (Circuit Confirmed)" if drop > 0.15 else "○ Still Redundant"
    print(f"  Ablated Accuracy:  {ablated_accuracy:>10.2%}")
    print(f"  Accuracy Drop:     {drop:>+10.2%}  {marker}")

    if torch.backends.mps.is_available(): torch.mps.empty_cache()

# -----------------------------------------------------------------------
# EXECUTION
# -----------------------------------------------------------------------
def run_final_experiment_with_3digits():
    device = ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nLoading Gemma-7B...")
    model = HookedTransformer.from_pretrained("google/gemma-7b", device=device, dtype=torch.bfloat16)
    model.eval()

    # The 4 key heads we found previously
    key_heads = [2, 4, 8, 12]
    layer = 25

    # Test: 3-Digit Addition (The True Edge of Competence)
    run_edge_of_competence_test(model, digits=3, layer=layer, key_heads=key_heads, n_pairs=50)

    del model
    if torch.backends.mps.is_available(): torch.mps.empty_cache()

if __name__ == "__main__":
    #google/gemma-2-2b
    #microsoft/Phi-3-mini-4k-instruct
    #EleutherAI/gpt-j-6B
    #google/gemma-7b
    #EleutherAI/pythia-6.9b
    #meta-llama/Llama-3.2-3B

    #scan_model_on_the_fly("EleutherAI/pythia-6.9b")
    #Layer 1, Head 17 (Dims 2,6) | CV: 0.312, Lin: 0.934

    #You can run it at the bottom of your script like this:
    test_output_plane_computation("google/gemma-7b", layer=14, head=2, dim1=2, dim2=5)
    #test_output_plane_computation("microsoft/Phi-3-mini-4k-instruct", layer=24, head=28, dim1=3, dim2=7)

    # Run it for Gemma 7B:
    #test_mlp_subspace_alignment("google/gemma-7b", layer=14, head=2, dim1=2, dim2=5)

    # Or run it for Phi-3 (You'll need to update layer, head, dim1, dim2 based on your previous Phi-3 logs!)
    # test_mlp_subspace_alignment("microsoft/Phi-3-mini-4k-instruct", layer=24, head=28, dim1=3, dim2=7)

    #avoid svd calculation on mlp due to its massive size, read fn to understand more
    #test_safe_mlp_alignment("google/gemma-7b", layer=14, head=2, dim1=2, dim2=5)
    #test_safe_mlp_alignment("microsoft/Phi-3-mini-4k-instruct", layer=24, head=28, dim1=3, dim2=7)

    # Run it for Gemma 7B!
    # test_causal_phase_shift("google/gemma-7b", layer=14, head=2, dim1=2, dim2=5)
    # test_causal_phase_shift("microsoft/Phi-3-mini-4k-instruct", layer=24, head=28, dim1=3, dim2=7)

    # Logit lens
    #run_rigorous_logit_lens("microsoft/Phi-3-mini-4k-instruct")
    #run_universal_logit_lens("google/gemma-7b")

    # casual graph
    #run_causal_brain_transplant("google/gemma-7b", layer=14, head=2)
    #run_causal_brain_transplant("microsoft/Phi-3-mini-4k-instruct", layer=24, head=28)

    #run_sledgehammer_transplant("microsoft/Phi-3-mini-4k-instruct", layer=24)
    #we change layer from 14 to 25 for causal proof
    #run_sledgehammer_transplant("google/gemma-7b", layer=10)

    # Run it!
    #plot_arithmetic_clock_face("google/gemma-7b", layer=14, head=2, dim1=2, dim2=5)

    # Run it!
    #sweep_svd_planes("google/gemma-7b", layer=14, head=2)
    #sweep_svd_planes("microsoft/Phi-3-mini-4k-instruct", layer=24, head=28)
    # Run it!
    #plot_dynamic_clock_face("google/gemma-7b", layer=14, head=2, dim1=2, dim2=5)
    #plot_arithmetic_clock_face("microsoft/Phi-3-mini-4k-instruct", layer=24, head=28, dim1=3, dim2=7)

    # run it at Layer 14 or Layer 21 (right before math ends)
    # plot_arithmetic_distance_matrices("google/gemma-7b", layer=21)
    # plot_arithmetic_distance_matrices("microsoft/Phi-3-mini-4k-instruct", layer=24)
    # plot_arithmetic_distance_matrices("microsoft/Phi-3-mini-4k-instruct", layer=21)

    # 1. Provide the mathematical score for the Professor's 2D theory
    #run_professors_automated_circle_scanner("google/gemma-7b", layer=14, head=2)
    #run_professors_automated_circle_scanner("microsoft/Phi-3-mini-4k-instruct", layer=24, head=28)
    # 2. Test the High-Dimensional Vector Translation theory
    #test_vector_translation_math("google/gemma-7b", layer=21)
    #test_vector_translation_math("microsoft/Phi-3-mini-4k-instruct", layer=24)

    # 3. See how many dimensions the math actually spans
    #plot_pca_variance_scree("google/gemma-7b", layer=21)
    #plot_pca_variance_scree("microsoft/Phi-3-mini-4k-instruct", layer=24)

    #run_unified_helix_scanner("google/gemma-7b", layer=14, head=2, max_n=100)
    #run_unified_helix_scanner("microsoft/Phi-3-mini-4k-instruct", layer=24, head=28, max_n=100)

    # Execute
    #run_corrected_pipeline("google/gemma-7b", layer=14, head=2)
    #run_corrected_pipeline("microsoft/Phi-3-mini-4k-instruct", layer=24, head=28)

    # run_phase_corrected_pipeline("google/gemma-7b", layer=14, head=2)
    # run_phase_corrected_pipeline("microsoft/Phi-3-mini-4k-instruct", layer=24, head=28)
    # run_phase_corrected_pipeline("meta-llama/Llama-3.2-3B", layer=1, head=17)


    # run_final_pipeline("google/gemma-7b", layer=14, head=2, period=10.0, best_layer=21)
    # run_final_pipeline("microsoft/Phi-3-mini-4k-instruct", layer=24, head=28, period=11.74, best_layer=25)


    # Test on the best computational layers
    # Use Layer 25 — your causally proven layer, not Layer 21
    # run_vector_translation_proof("google/gemma-7b", causal_layer=25)
    # run_vector_translation_proof("microsoft/Phi-3-mini-4k-instruct", causal_layer=25)

    # run_analysis("google/gemma-7b", layer_range=range(19, 25), n_pairs=50)
    # run_analysis("microsoft/Phi-3-mini-4k-instruct", layer_range=range(20, 27), n_pairs=50)
    # run_analysis("google/gemma-2b", layer_range=range(8, 10), n_pairs=100)


    # run_final_characterization("google/gemma-7b", key_layer=22, key_neuron=429)
    # run_final_characterization("google/gemma-7b", key_layer=24, key_neuron=20892)
    # run_final_characterization("google/gemma-2b", key_layer=10, key_neuron=3481)
    # run_final_characterization("google/gemma-2b", key_layer=10, key_neuron=9767)
    # run_final_characterization("google/gemma-2b", key_layer=10, key_neuron=9377)
    # run_final_characterization("google/gemma-2b", key_layer=10, key_neuron=14119)
    # run_final_characterization("google/gemma-2b", key_layer=10, key_neuron=11906)


    # Run 2B first to ensure your environment is stable before hitting the 7B
    # run_laser_focus_analysis("google/gemma-2b", causal_layer=17)
    # run_laser_focus_analysis("google/gemma-7b", causal_layer=25)

    # Gemma-7B (Using Layer 25 as the causally proven target)
    # run_circuit_discovery("google/gemma-2b", target_layer=17, n_pairs=50)

    # If you run Gemma-7B later (which has 28 layers):
    # run_circuit_discovery("google/gemma-7b", target_layer=25, n_pairs=50)

    # If you run Phi-3 (which has 32 layers):
    # run_circuit_discovery("microsoft/Phi-3-mini-4k-instruct", target_layer=25, n_pairs=50)

    #compare_formats("google/gemma-2b")
    #compare_formats("microsoft/Phi-3-mini-4k-instruct")
    #compare_formats("google/gemma-7b")
    #compare_formats("microsoft/Phi-3-mini-4k-instruct")
    #compare_formats("google/gemma-2b")
    #compare_formats("google/gemma-7b")  # run separately if memory allows

    # # Phi-3: Distributed logic
    # run_head_drilldown("microsoft/Phi-3-mini-4k-instruct", computation_layer=25)
    #
    # # Gemma-2B: Glass cannon logic
    # run_head_drilldown("google/gemma-2b", computation_layer=15)

    # Gemma-7B: Carry-router logic
    #run_head_drilldown("google/gemma-7b", computation_layer=25)


    #run_final_experiment()
    #run_final_experiment_with_3digits()



