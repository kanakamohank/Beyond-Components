import torch
import numpy as np
import itertools
from transformer_lens import HookedTransformer
import logging

logger = logging.getLogger(__name__)

class GeometricArithmeticPipeline:
    """
    Implements the 6-Phase Geometric SVD Framework for Arithmetic Circuit Discovery
    as defined in RESEARCH_ROADMAP_UPDATED.md.
    """
    def __init__(self, model: HookedTransformer):
        self.model = model
        self.cfg = model.cfg
        self.device = next(model.parameters()).device

    # ==========================================
    # UTILITY: Robust Tokenization (Fixes GPT-Neo)
    # ==========================================
    def get_operand_token_index(self, prompt: str, target_number: int) -> int:
        """Robustly finds the LAST token index of a fragmented number."""
        tokens = self.model.to_tokens(prompt)[0]
        str_tokens = self.model.to_str_tokens(prompt)

        # Search backwards to find the last piece of the fragmented number
        target_str = str(target_number)
        for i in range(len(str_tokens) - 1, -1, -1):
            # Strip whitespace and special characters used by various tokenizers (GPT-2, Neo, Llama)
            clean_tok = str_tokens[i].strip(' Ġ_')
            if clean_tok in target_str and len(clean_tok) > 0:
                return i

        logger.warning(f"Could not reliably find {target_number} in {str_tokens}. Defaulting to -2.")
        return len(tokens) - 2 # Fallback to typical operand position

    # ==========================================
    # PHASE 2: OV Circuit Extraction & SVD
    # ==========================================
    def extract_ov_svd(self, layer: int, head: int):
        """Extracts W_OV = W_V @ W_O and performs SVD (Fixes the Activation Fallacy)."""
        with torch.no_grad():
            # Detach to prevent gradient tracking
            W_V = self.model.W_V[layer, head].detach()
            W_O = self.model.W_O[layer, head].detach()

            W_OV = W_V @ W_O

            # SVD on MPS triggers fallbacks. Compute on CPU, then return to device.
            U, S, Vh = torch.linalg.svd(W_OV.cpu(), full_matrices=False)

            return U.to(self.device), S.to(self.device), Vh.to(self.device)

    # ==========================================
    # PHASE 3: Input Plane Geometric Testing
    # ==========================================
    def test_input_plane(self, layer: int, head: int, numbers: list, prompt_template="{n} + 5 ="):
        with torch.no_grad():
            U, S, Vh = self.extract_ov_svd(layer, head)
            activations = []
            valid_nums = []

            for n in numbers:
                prompt = prompt_template.format(n=n)

                # ---> THE FIX: Use the LAST token (-1) just like your old code! <---
                idx = -1

                _, cache = self.model.run_with_cache(self.model.to_tokens(prompt))
                resid_pre = cache[f"blocks.{layer}.hook_resid_pre"][0, idx, :]
                activations.append(resid_pre)
                valid_nums.append(n)

            acts_tensor = torch.stack(activations)
            best_plane = None
            best_score = float('inf')

            # Use your original 2D Helix math (no mean centering)
            for k1, k2 in itertools.combinations(range(10), 2):
                v1, v2 = Vh[k1], Vh[k2]
                raw_coords = torch.stack([acts_tensor @ v1, acts_tensor @ v2], dim=1)
                # ---> YOUR INTUITION RESTORED: MEAN CENTERING <---
                # Find the center of the geometry and shift the coordinates
                center = raw_coords.mean(dim=0)
                coords = raw_coords - center

                # Now calculate polar coordinates from the TRUE center of the circle
                radii = coords.norm(dim=1)
                angles = torch.atan2(coords[:, 1], coords[:, 0]).detach().cpu().numpy()

                radii_mean = radii.mean().item()
                radius_cv = (radii.std() / radii_mean).item() if radii_mean > 1e-6 else float('inf')

                unwrapped = np.unwrap(angles)
                if np.std(unwrapped) > 1e-6:
                    angle_lin = abs(np.corrcoef(valid_nums, unwrapped)[0, 1])
                    if np.isnan(angle_lin): angle_lin = 0.0
                else:
                    angle_lin = 0.0

                if radius_cv < 0.2 and angle_lin > 0.9:
                    score = radius_cv - angle_lin
                    if score < best_score:
                        best_score = score
                        best_plane = {
                            'k1': k1, 'k2': k2, 'cv': radius_cv,
                            'lin': angle_lin, 'v1': v1, 'v2': v2
                        }

            return best_plane



    # ==========================================
    # PHASE 4: Output Plane Computation Testing
    # ==========================================
    def test_output_plane(self, layer: int, head: int, operand_pairs: list, u_k1: int, u_k2: int):
        with torch.no_grad():
            U, _, _ = self.extract_ov_svd(layer, head)
            u1, u2 = U[:, u_k1], U[:, u_k2]

            outputs = []
            target_sums = []

            for a, b in operand_pairs:
                prompt = f"{a} + {b} ="

                # ---> THE FIX: Extract the output at the LAST token (-1) <---
                # Because the computation is written to the residual stream at the '=' sign!
                idx = -1

                target_sums.append(a + b)

                _, cache = self.model.run_with_cache(self.model.to_tokens(prompt))
                head_out = cache[f"blocks.{layer}.attn.hook_z"][0, idx, head, :]

                W_O = self.model.W_O[layer, head]
                output_vec = head_out @ W_O
                outputs.append(output_vec)

            outputs_tensor = torch.stack(outputs)

            coords = torch.stack([outputs_tensor @ u1, outputs_tensor @ u2], dim=1)
            radii = coords.norm(dim=1)
            angles = torch.atan2(coords[:, 1], coords[:, 0]).detach().cpu().numpy()

            radius_cv = (radii.std() / radii.mean()).item()
            angle_lin = abs(np.corrcoef(target_sums, np.unwrap(angles))[0, 1])

            return {
                'is_output_helix': radius_cv < 0.2 and angle_lin > 0.9,
                'cv': radius_cv,
                'linearity': angle_lin
            }

    # ==========================================
    # PHASE 5: MLP Subspace Alignment (Architecture-Agnostic)
    # ==========================================
    def measure_mlp_alignment(self, layer: int, head: int, u_k1: int, u_k2: int):
        with torch.no_grad():
            U_ov, _, _ = self.extract_ov_svd(layer, head)
            attn_plane = torch.stack([U_ov[:, u_k1], U_ov[:, u_k2]]) # [2, d_model]

            mlp_module = self.model.blocks[layer].mlp

            # Detach the weights when extracting them
            if hasattr(mlp_module, 'W_gate') and hasattr(mlp_module, 'W_in'):
                W_read = torch.cat([mlp_module.W_gate.detach(), mlp_module.W_in.detach()], dim=1)
            elif hasattr(mlp_module, 'W_in'):
                W_read = mlp_module.W_in.detach()
            else:
                raise AttributeError(f"Unrecognized MLP architecture in model: {self.cfg.model_name}")

            # Compute SVD on CPU to prevent MPS warnings
            U_mlp, S_mlp, Vh_mlp = torch.linalg.svd(W_read.cpu(), full_matrices=False)
            U_mlp = U_mlp.to(self.device)

            mlp_reading_subspace = U_mlp[:, :10].T # [10, d_model]

            attn_plane_norm = torch.nn.functional.normalize(attn_plane, dim=1)
            mlp_subspace_norm = torch.nn.functional.normalize(mlp_reading_subspace, dim=1)

            alignment_matrix = torch.matmul(attn_plane_norm, mlp_subspace_norm.T) # [2, 10]
            max_alignment = alignment_matrix.abs().max().item()

            return {
                'max_alignment': max_alignment,
                'alignment_matrix': alignment_matrix.cpu().numpy(),
                'is_coupled': max_alignment > 0.7
            }
    # ==========================================
    # PHASE 6: Causal Verification (Phase-Shift)
    # ==========================================
    def causal_phase_shift(self, layer: int, head: int, v1: torch.Tensor, v2: torch.Tensor,
                           a: int, b: int, shift_delta: int, period: float = 10.0):
        """
        Directly rotates the representation vector inside the discovered SVD plane.
        """
        prompt = f"{a} + {b} ="
        tokens = self.model.to_tokens(prompt)
        idx = self.get_operand_token_index(prompt, a)

        expected_ans = a + b + shift_delta
        theta = 2 * np.pi * shift_delta / period

        # Get clean cache
        _, cache = self.model.run_with_cache(tokens)
        clean_resid = cache[f"blocks.{layer}.hook_resid_pre"][0, idx, :]

        # Compute current 2D coordinates in the helix plane
        c1 = (clean_resid @ v1).item()
        c2 = (clean_resid @ v2).item()

        # Apply 2D Rotation Matrix math
        new_c1 = c1 * np.cos(theta) - c2 * np.sin(theta)
        new_c2 = c1 * np.sin(theta) + c2 * np.cos(theta)

        # Reconstruct vector: Remove old projection, add rotated projection
        rotated_resid = clean_resid - (c1 * v1 + c2 * v2) + (new_c1 * v1 + new_c2 * v2)

        # Patching Hook
        def rotation_hook(resid, hook):
            resid[0, idx, :] = rotated_resid
            return resid

        # Run intervened model
        patched_logits = self.model.run_with_hooks(
            tokens,
            fwd_hooks=[(f"blocks.{layer}.hook_resid_pre", rotation_hook)]
        )

        predicted_token = patched_logits[0, -1, :].argmax().item()
        predicted_str = self.model.to_string([predicted_token])

        return {
            'original_sum': a + b,
            'expected_shifted_sum': expected_ans,
            'model_output': predicted_str
        }