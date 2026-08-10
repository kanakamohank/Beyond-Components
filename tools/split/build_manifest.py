#!/usr/bin/env python3
"""Classify every tracked file in Beyond-Components into per-paper buckets.

Buckets: compass | arithmetic | both | drop
Emits SPLIT_MANIFEST.tsv (path, bucket, reason).
"""
import re
import subprocess
from pathlib import Path

ROOT = Path("/home/user/Beyond-Components")

files = subprocess.run(
    ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
).stdout.splitlines()

# ---------------------------------------------------------------- cookbook refs
compass_cb = (ROOT / "COMPASS_COOKBOOK.md").read_text(errors="replace")
arith_cb = (ROOT / "ARITHMETIC_CIRCUIT_PLAN.md").read_text(errors="replace")

REF = re.compile(r"experiments/[A-Za-z0-9_/.-]+\.(?:py|sh)")
compass_scripts = set(REF.findall(compass_cb))
arith_scripts = set(REF.findall(arith_cb))

# ------------------------------------------------- explicit experiments/ rulings
# The 27 scripts neither cookbook names, classified by module docstring.
EXP_EXTRA = {
    # --- compass ---
    "experiments/baseline_comparison.py": ("compass", "docstring: baselines for the semantic compass"),
    "experiments/flip_ratio_by_domain.py": ("compass", "docstring: per-domain flip ratio over spotcheck CSVs"),
    "experiments/storage_steering_scatter.py": ("compass", "docstring: storage-vs-steering for the 5 compass heads"),
    "experiments/stereoset_ensemble_eval_gemma.py": ("compass", "per-model variant of stereoset_ensemble_eval.py"),
    "experiments/stereoset_ensemble_eval_llama.py": ("compass", "per-model variant of stereoset_ensemble_eval.py"),
    "experiments/stereoset_ensemble_eval_phi3.py": ("compass", "per-model variant of stereoset_ensemble_eval.py"),
    # --- arithmetic ---
    "experiments/circuit_synthesis.md": ("arithmetic", "multi-model arithmetic circuit synthesis notes"),
    "experiments/analyze_sum_encoding.py": ("arithmetic", "docstring: Phase 4c SVD sum-encoding (grokking theory)"),
    "experiments/analyze_svd_directions.py": ("arithmetic", "docstring: Phase 3 Fourier analysis of SVD directions"),
    "experiments/arithmetic_bus_validation.py": ("arithmetic", "dense-patch validation of the arithmetic bus"),
    "experiments/arithmetic_circuit_scan.py": ("arithmetic", "v2 of the arithmetic layer scan"),
    "experiments/causal_digit_probing.py": ("arithmetic", "docstring: causal probing of ones-digit encoding"),
    "experiments/causal_validation.py": ("arithmetic", "docstring: Phase 5 direction-level scalar swapping"),
    "experiments/diagnose_rotation_sign.py": ("arithmetic", "docstring: Fourier phase-rotation sign diagnosis"),
    "experiments/diagnose_unembed_direct.py": ("arithmetic", "docstring: direct-answer unembed patching diagnosis"),
    "experiments/mathematical_toolkit.py": ("arithmetic", "docstring: toolkit for arithmetic circuit discovery"),
    "experiments/phase_rotation_statistics.py": ("arithmetic", "docstring: stats for Fourier phase rotation"),
    "experiments/phi3_arithmetic_scan_gemma_worked.py": ("arithmetic", "arithmetic layer scan, gemma run"),
    "experiments/phi3_arithmetic_scan_phi3_worked.py": ("arithmetic", "arithmetic layer scan, phi3 run"),
    "experiments/plot_eigenvector_dft.py": ("arithmetic", "plots for eigenvector_dft.py (Step 2)"),
    "experiments/run_fourier_discovery.py": ("arithmetic", "docstring: Phase 1 Fourier discovery on arithmetic prompts"),
    "experiments/run_multi_model_suite.py": ("arithmetic", "docstring: multi-model mathematical-toolkit runner"),
    "experiments/steering_residual_vector.py": ("arithmetic", "residual steering on the arithmetic bus"),
    "experiments/validate_unembed_fix.py": ("arithmetic", "validates compute_unembed_basis_direct_answer"),
    "experiments/visualize_toolkit_results.py": ("arithmetic", "plots for mathematical_toolkit.py"),
    # --- Beyond Components only ---
    "experiments/analyze_checkpoint.py": ("drop", "Beyond Components mask-checkpoint inspector"),
    "experiments/ablation/comprehensive_sigma_test.py": ("drop", "Beyond Components sigma amplification"),
    "experiments/evaluation/generate_sigma_table.py": ("drop", "Beyond Components sigma results table"),
}

# ------------------------------------------------------------------ src/ ruling
SRC = {
    "src/__init__.py": ("both", "package root"),
    "src/analysis/__init__.py": ("arithmetic", "package root for Fourier analysis"),
    "src/analysis/experiment_history.py": ("arithmetic", "logs arithmetic experiment history"),
    "src/analysis/fourier_discovery.py": ("arithmetic", "ARITHMETIC_CIRCUIT_PLAN references"),
    "src/analysis/fourier_plots.py": ("arithmetic", "plots for fourier_discovery"),
    "src/data/__init__.py": ("both", "package root"),
    "src/data/arithmetic_dataset.py": ("arithmetic", "ARITHMETIC_CIRCUIT_PLAN references"),
    "src/data/data_loader.py": ("both", "GP/IOI/GT loaders; compass Stage-1 + arithmetic training"),
    "src/models/__init__.py": ("both", "package root"),
    "src/models/evaluate_model.py": ("arithmetic", "arithmetic model evaluation"),
    "src/models/helix_circuit_discovery.py": ("arithmetic", "helix circuit discovery"),
    "src/models/masked_transformer_circuit.py": ("both", "Stage-1 mask learning; cited by both cookbooks"),
    "src/models/offline_svd_scanner.py": ("arithmetic", "OV-SVD helix scanner"),
    "src/models/online_svd_scanner.py": ("arithmetic", "ARITHMETIC_CIRCUIT_PLAN references"),
    "src/run_helix_analysis.py": ("arithmetic", "helix analysis driver"),
    "src/utils/__init__.py": ("both", "package root"),
    "src/utils/constants.py": ("both", "project-wide constants"),
    "src/utils/model_registry.py": ("arithmetic", "used by tests/test_model_registry.py"),
    "src/utils/utils.py": ("both", "model loading, column helpers, seeding"),
    "src/utils/visualization.py": ("both", "mask heatmaps / training curves"),
}

# ------------------------------------------------------------------ root ruling
ROOT_FILES = {
    ".gitignore": ("both", "shared ignore rules"),
    "CLAUDE.md": ("both", "repo coding guidelines"),
    "requirements.txt": ("both", "shared dependency set"),
    "setup.py": ("both", "package install script"),
    "README.md": ("drop", "Beyond Components README; rewritten per repo"),
    "COMPASS_COOKBOOK.md": ("compass", "the compass cookbook"),
    "ARITHMETIC_CIRCUIT_PLAN.md": ("arithmetic", "the arithmetic cookbook"),
    "MATHEMATICAL_TOOLKIT_PROPOSAL.md": ("arithmetic", "toolkit proposal for arithmetic circuits"),
    "FOURIER_PHASE_ROTATION_FINDINGS.md": ("arithmetic", "Fourier phase-rotation findings"),
    "Diagnosing Fisher Patching.md": ("arithmetic", "Fisher patching debug log"),
    "helix_cross_model_analysis.md": ("arithmetic", "cross-model helix analysis"),
    "svd_stats_ov_helix_circuit.md": ("arithmetic", "OV helix SVD stats"),
    "analysis_log_extracted.md": ("arithmetic", "model analysis log"),
    "Raw_log_from_IDE_console.txt": ("arithmetic", "raw console log of arithmetic runs"),
    "experiment_history.jsonl": ("arithmetic", "written by src/analysis/experiment_history.py"),
    "experiment_layernorm_depth.py": ("arithmetic", "layernorm-depth arithmetic experiment"),
    "run_geometric_pipeline.py": ("arithmetic", "geometric arithmetic pipeline driver"),
    "investigate_helix_usage_validated.py": ("both", "compass 9-test battery + cross-task helix investigation"),
    "run_train.py": ("both", "wrapper for experiments/train.py"),
    "run_ablation.py": ("arithmetic", "wrapper for experiments/ablation/intervention.py, which ships to arithmetic"),
}

CONFIGS = {
    "configs/gp_config.yaml": ("both", "Gender Pronoun; compass Stage-1 + shared training"),
    "configs/ioi_config.yaml": ("drop", "Beyond Components IOI task"),
    "configs/gt_config.yaml": ("drop", "Beyond Components Greater-Than task"),
}

KNOWLEDGE = {
    "knowledge/A_Mechanistic_Interpretability_Analysis_of_Grokking.ipynb": "arithmetic",
    "knowledge/COMPREHENSIVE_PAPER_ANALYSIS.md": "both",
    "knowledge/beyond_components_deep_analysis.md": "both",
    "knowledge/comparison_modular_digit_vs_repo.md": "arithmetic",
    "knowledge/index.md": "both",
    "knowledge/relavant_papers.txt": "both",
    "knowledge/summary_arithmetic_circuits.md": "arithmetic",
    "knowledge/summary_arithmetic_reasoning.md": "arithmetic",
    "knowledge/summary_beyond_components.md": "both",
    "knowledge/summary_clock_pizza.md": "arithmetic",
    "knowledge/summary_fourier_features_addition.md": "arithmetic",
    "knowledge/summary_grokking_mechanistic.md": "arithmetic",
    "knowledge/summary_modular_digit_arithmetic.md": "arithmetic",
    "knowledge/summary_modular_polynomials.md": "arithmetic",
    "knowledge/summary_singular_vectors_features.md": "both",
    "knowledge/summary_toy_superposition.md": "both",
    "knowledge/summary_trigonometry_addition.md": "arithmetic",
}

# helix_usage_validated/: outputs of the shared helix scanner go to both repos;
# everything else in there is a compass artifact.
HELIX_SHARED = re.compile(r"(_sweep_output|_trace_output)|^(gptj6b|gptneo125m|gemma7b|gpt2medium)_")


def classify(p: str):
    if p in ROOT_FILES:
        return ROOT_FILES[p]
    if p in CONFIGS:
        return CONFIGS[p]
    if p in SRC:
        return SRC[p]
    if p in EXP_EXTRA:
        return EXP_EXTRA[p]
    if p in KNOWLEDGE:
        return (KNOWLEDGE[p], "research note, routed by topic")

    if p.startswith("configs/arithmetic_"):
        return ("arithmetic", "arithmetic training config")

    if p.startswith("experiments/"):
        if p == "experiments/train.py":
            return ("both", "shared Stage-1 trainer; named in ARITHMETIC_CIRCUIT_PLAN.md, used by compass Stage-1")
        if p in compass_scripts:
            return ("compass", "named in COMPASS_COOKBOOK.md")
        if p in arith_scripts:
            return ("arithmetic", "named in ARITHMETIC_CIRCUIT_PLAN.md")
        if p == "experiments/ablation/intervention.py":
            return ("arithmetic", "BC-lineage but referenced by ARITHMETIC_CIRCUIT_PLAN.md")
        if p == "experiments/evaluation/comprehensive_metrics_table.py":
            return ("compass", "BC-lineage but referenced by COMPASS_COOKBOOK.md")
        if p.endswith("__init__.py"):
            return ("both", "package root")
        return ("UNCLASSIFIED", "experiments/ script with no rule")

    if p.startswith(("paper_compass/", "paper_compass_acl/")):
        return ("compass", "compass paper source")
    if p.startswith("acl-style-files-master/"):
        return ("compass", "ACL style files for the compass ACL submission")
    if p.startswith("paper/"):
        return ("arithmetic", "arithmetic paper source")
    if p.startswith("svd_logs/"):
        return ("drop", "Beyond Components IOI circuit-discovery run")
    if p.startswith("fourier_results/"):
        return ("arithmetic", "Fourier discovery outputs")
    if p.startswith("tests/"):
        return ("arithmetic", "all tests target arithmetic/Fourier code")
    if p.startswith("memory/"):
        return ("both", "repo working notes")
    if p.startswith("claude_skills/"):
        return ("both", "repo tooling")

    if p.startswith("images/"):
        if p.endswith("intervention.png"):
            return ("drop", "Beyond Components teaser figure")
        return ("arithmetic", "helix / Fourier / geometry figure")

    if p.startswith("helix_usage_validated/"):
        stem = p.split("/", 1)[1]
        if HELIX_SHARED.search(stem):
            return ("both", "cross-model helix scan output (shared scanner)")
        return ("compass", "compass experiment artifact")

    if p.startswith("data/"):
        if "winogender" in p:
            return ("compass", "Winogender occupation stats")
        return ("both", "shared dataset loader / docs")

    return ("UNCLASSIFIED", "no rule matched")


rows = [(p, *classify(p)) for p in files]

out = ROOT / "SPLIT_MANIFEST.tsv"
with out.open("w") as fh:
    fh.write("path\tbucket\treason\n")
    for p, b, r in rows:
        fh.write(f"{p}\t{b}\t{r}\n")

from collections import Counter

counts = Counter(b for _, b, _ in rows)
print("total tracked files:", len(rows))
for k in ("compass", "arithmetic", "both", "drop", "UNCLASSIFIED"):
    print(f"  {k:<14} {counts.get(k, 0)}")

unc = [p for p, b, _ in rows if b == "UNCLASSIFIED"]
if unc:
    print("\nUNCLASSIFIED:")
    for p in unc:
        print("   ", p)
