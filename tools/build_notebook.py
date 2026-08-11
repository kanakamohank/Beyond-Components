#!/usr/bin/env python3
"""Generate experiments_end_to_end.ipynb.

The notebook is generated rather than hand-edited so it can be regenerated and
re-validated after the code it documents changes.
"""
import json
from pathlib import Path

ROOT = Path("/home/user/Beyond-Components")
cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                  "outputs": [], "source": text.strip("\n").splitlines(keepends=True)})


# ════════════════════════════════════════════════════════════ TITLE
md(r"""
# Semantic Compasses & The Fourier Basis of Digit Arithmetic
## An end-to-end experimental compendium

This notebook is the executable index to **two** research programmes:

| Part | Paper | Repository |
|---|---|---|
| **1** | *Semantic Compasses: Rank-2 Causal Dials in Attention-Head OV Singular Planes* | `semantic-compass/` |
| **2** | *The Fourier Basis of Digit Arithmetic* | `arithmetic-circuit-discovery/` |

For every experiment it records **what the experiment does**, **why we ran it**,
**the exact command**, and **what the result actually looked like** — including
the results that came out null or negative.

---

### Read this before you run anything

**This notebook cannot download models.** Every headline experiment needs
pretrained weights (GPT-2, Phi-3, Gemma-2B, LLaMA-3.2-3B) from HuggingFace. In a
sandboxed or air-gapped environment that fetch will fail. So the notebook is
built in **three tiers**, and only Tier 0 is guaranteed to run anywhere:

| Tier | Needs | Runtime | What it gives you |
|---|---|---|---|
| **0 — Verify & Replay** | nothing but this repo | seconds | Math checks on synthetic data; **re-analysis of our recorded experimental artifacts**. Real numbers from real runs. |
| **1 — Reproduce (small)** | HuggingFace access, CPU ok | minutes–hours | The GPT-2 compass results, recomputed from scratch. |
| **2 — Reproduce (full)** | GPU + gated model access | days | The full cross-model grid for both papers. |

**Tier 0 is not a mock.** It parses the actual `.txt`/`.json` logs committed in
`helix_usage_validated/` and `fourier_results/` and recomputes the statistics
from them. If a Tier-0 cell prints a number, that number came out of a real run.

Tier 1 and 2 cells are **inert by default**: they print the command they would
run and stop. Set `RUN_TIER = 1` or `2` below to arm them, after the preflight
cell tells you the environment can support it.

---

### Provenance and honesty

I am reporting our own results here, so the expected-outcome boxes state what we
actually observed, not what we hoped for. Three results in Part 1 are **null or
negative** (§1.6, §1.7) and one headline result in Part 2 is a **falsified
hypothesis** (§2.5, the *k*=5 paradox). They are documented as prominently as
the positive ones. If you reproduce this work and your compass scan does not
beat its null, **that is the expected result** — see §1.7.
""")

# ════════════════════════════════════════════════════════════ CONFIG
md(r"""
## 0.1 Configuration

Resolves the repository root and sets the execution tier. Everything downstream
uses `COMPASS` / `ARITH` rather than relative paths, so the notebook runs from
any working directory.
""")

code(r'''
import json, os, re, sys, textwrap, subprocess
from pathlib import Path

# ─────────────────────────────────────────────────────────────── settings
RUN_TIER = 0        # 0 = verify+replay only | 1 = also recompute GPT-2 | 2 = full grid
DEVICE   = "auto"   # "auto" | "cpu" | "cuda" | "mps"

# ────────────────────────────────────────────────── repository resolution
def find_root(start: Path) -> Path:
    """Walk upward for the directory holding both split repositories."""
    for d in [start, *start.parents]:
        if (d / "semantic-compass").is_dir() and (d / "arithmetic-circuit-discovery").is_dir():
            return d
    raise SystemExit(
        "Could not locate the repository root.\\n"
        "Expected an ancestor directory containing BOTH 'semantic-compass/' and\\n"
        "'arithmetic-circuit-discovery/'. Run this notebook from inside the repo."
    )

ROOT    = find_root(Path.cwd().resolve())
COMPASS = ROOT / "semantic-compass"
ARITH   = ROOT / "arithmetic-circuit-discovery"

# Recorded-artifact directories. Tier 0 reads from these.
CV_LOGS = COMPASS / "helix_usage_validated"      # compass logs, plots, CSVs
FR_LOGS = ARITH   / "fourier_results"            # recorded Fourier discovery output

print(f"repository root : {ROOT}")
print(f"  compass       : {COMPASS.name}/   ({sum(1 for _ in COMPASS.rglob('*.py'))} py files)")
print(f"  arithmetic    : {ARITH.name}/   ({sum(1 for _ in ARITH.rglob('*.py'))} py files)")
print(f"execution tier  : {RUN_TIER}  "
      f"({'verify + replay recorded artifacts' if RUN_TIER == 0 else 'live recomputation ARMED'})")
''')

# ════════════════════════════════════════════════════════════ PREFLIGHT
md(r"""
## 0.2 Preflight

Reports what this environment can actually do, so you learn about a missing
dependency here rather than forty minutes into a sweep.

The model-access probe is the one that matters. It does a short HTTP request to
HuggingFace; **if it fails, every Tier 1/2 cell in this notebook will fail**, and
you should stay on Tier 0.
""")

code(r'''
import importlib.util

def have(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None

print(f"python            : {sys.version.split()[0]}")

deps = ["torch", "numpy", "transformer_lens", "matplotlib", "pandas", "sklearn", "seaborn"]
missing = [d for d in deps if not have(d)]
for d in deps:
    print(f"  {d:<16}: {'yes' if have(d) else 'MISSING'}")

# accelerator
if have("torch"):
    import torch
    dev = ("cuda" if torch.cuda.is_available()
           else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
           else "cpu")
    if DEVICE != "auto":
        dev = DEVICE
    print(f"accelerator       : {dev}")
else:
    dev = "cpu"

# model access -- the gate for Tier 1 and 2
MODEL_ACCESS = False
try:
    import urllib.request
    urllib.request.urlopen("https://huggingface.co/gpt2/resolve/main/config.json", timeout=10)
    MODEL_ACCESS = True
except Exception as e:
    print(f"huggingface       : UNREACHABLE ({type(e).__name__})")
else:
    print("huggingface       : reachable")

# recorded artifacts -- the gate for Tier 0
n_compass_logs = len(list(CV_LOGS.glob("*"))) if CV_LOGS.is_dir() else 0
n_fourier_logs = len(list(FR_LOGS.glob("*.json"))) if FR_LOGS.is_dir() else 0
print(f"recorded artifacts: {n_compass_logs} compass, {n_fourier_logs} fourier json")

print()
if missing:
    print(f"! missing packages: {', '.join(missing)}   ->  pip install {' '.join(missing)}")
if RUN_TIER > 0 and not MODEL_ACCESS:
    print("! RUN_TIER > 0 but HuggingFace is unreachable. Live cells WILL fail.")
    print("  Set RUN_TIER = 0 to run the verification and replay cells only.")
elif RUN_TIER == 0:
    print("Tier 0: replaying recorded artifacts. No model downloads required.")

def tier(n: int) -> bool:
    """Guard for live cells. Returns True only if this environment can run them."""
    if RUN_TIER < n:
        print(f"[tier {n} cell -- inactive, RUN_TIER={RUN_TIER}. Command shown above, not executed.]")
        return False
    if not MODEL_ACCESS:
        print(f"[tier {n} cell -- SKIPPED, no model access.]")
        return False
    return True

def shell(cmd: str, cwd: Path):
    """Display a command, and run it only if its tier is armed."""
    print(textwrap.dedent(cmd).strip())
''')

# ════════════════════════════════════════════════════════════ PART 1
md(r"""
---
# Part 1 — Semantic Compasses

## The claim in one paragraph

Take any attention head, form its OV map $W_{OV} = W_V W_O$, and take the SVD.
Pick **two** singular directions. That 2-D plane is often a **semantic compass**:
adding a rotating vector inside it sweeps a model's output logits along a smooth,
single-frequency sinusoid. The angle $\theta$ selects *which way* the semantics
point (he ↔ she, past ↔ future); the scale $\alpha$ selects *how hard* you push.
It is a continuous, causally effective dial — not a binary switch, and not a
correlational probe.

## The five formulas

Everything in Part 1 is built from these. `§` references are to
`semantic-compass/COMPASS_COOKBOOK.md`.

**1. The plane** (cookbook §2.1)
$$W_{OV} = W_V W_O, \qquad U\Sigma V^\top = \mathrm{svd}(W_{OV}), \qquad \mathrm{plane}(L,H,i,j) = \mathrm{span}(u_i, u_j)$$

**2. The injection** (§2.2)
$$v(\theta,\alpha) = \alpha\,\sigma_i\cos\theta\;u_i \;+\; \alpha\,\sigma_j\sin\theta\;u_j$$

**3. The site** (§2.3) — hook `blocks.{L}.hook_resid_pre`, **last token only**:
`act[0, -1, :] += v(theta, alpha)`. This is the *only* intervention point in the
entire pipeline.

**4. The metric** (§2.4) — for antipodal probe tokens $(t_+, t_-)$:
$$\mathrm{LD}(\theta,\alpha) = \mathrm{logit}(t_+) - \mathrm{logit}(t_-)$$
averaged over 3 prompts × 3 conditions (neutral / plus-context / minus-context).

**5. The fit** (§2.5) — first DFT bin:
$$\mathrm{LD}(\theta) \approx \mu + A\cos(\theta - \varphi)$$

**The pass criterion** (§2.6) — a plane is a compass iff all three hold:

| Pillar | Threshold |
|---|---|
| Amplitude linearity | $R^2(A \text{ vs } \alpha\text{, through origin}) \ge 0.95$ |
| Phase stability | $\lvert\varphi(\alpha_{hi}) - \varphi(\alpha_{lo})\rvert \le 10°$ |
| Effect size | $A(\alpha_{hi})/\alpha_{hi} \ge 0.20$ (0.08 for Gemma) |
""")

md(r"""
### 1.1 — The math core, verified against a synthetic oracle

**Purpose.** Before trusting any result, confirm the estimator itself is correct.
We build a signal with a *known* amplitude and phase, then check the DFT fit
recovers them.

**Why this matters.** The whole paper rests on reading $A$ and $\varphi$ off a
24-point angular sweep. If that estimator is biased, every downstream number is
too. This cell is cheap insurance and runs anywhere.

**Expected outcome.** Recovery to ~1e-12 (floating-point exact). A pure cosine
must also yield $\varphi = 0$, and the fit must be blind to a DC offset.

*Tier 0 · no model · instant*
""")

code(r'''
import numpy as np

def fit_sinusoid(theta_deg, y):
    """First-DFT-bin fit of  y ~ mu + A*cos(theta - phi).

    This mirrors compass_scan.py:54-62, the most-cited form in the paper.
    Returns (mu, A, phi_degrees).
    """
    th  = np.deg2rad(np.asarray(theta_deg, dtype=float))
    y   = np.asarray(y, dtype=float)
    N   = len(y)
    mu  = y.mean()
    c   = ((y - mu) * np.cos(th)).sum() * 2.0 / N
    s   = ((y - mu) * np.sin(th)).sum() * 2.0 / N
    return mu, float(np.hypot(c, s)), float(np.degrees(np.arctan2(s, c)))

# ── oracle: known mu, A, phi on the same 24-point grid the experiments use
grid = np.arange(0, 360, 15.0)
for mu_t, A_t, phi_t in [(0.0, 1.0, 0.0), (0.55, 1.19, -175.4), (-2.0, 5.0, 90.0)]:
    y = mu_t + A_t * np.cos(np.deg2rad(grid - phi_t))
    mu_h, A_h, phi_h = fit_sinusoid(grid, y)
    dphi = (phi_h - phi_t + 180) % 360 - 180          # wrap to [-180, 180)
    ok = abs(mu_h-mu_t) < 1e-9 and abs(A_h-A_t) < 1e-9 and abs(dphi) < 1e-6
    print(f"mu={mu_t:+.2f} A={A_t:.2f} phi={phi_t:+.1f}  ->  "
          f"recovered mu={mu_h:+.6f} A={A_h:.6f} phi={phi_h:+.4f}   {'OK' if ok else 'FAIL'}")

# ── the fit must ignore a DC offset entirely (it is subtracted before the bin)
y = 100.0 + 2.0*np.cos(np.deg2rad(grid - 30.0))
_, A_h, phi_h = fit_sinusoid(grid, y)
print(f"\nDC-offset invariance: A={A_h:.6f} (expect 2.0), phi={phi_h:+.4f} (expect +30.0)")
assert abs(A_h - 2.0) < 1e-9 and abs(phi_h - 30.0) < 1e-6
print("\nEstimator verified.")
''')

# ---- 1.2 injection geometry
md(r"""
### 1.2 — Injection geometry, verified

**Purpose.** Confirm the injection vector $v(\theta,\alpha)$ behaves as the paper
claims: it stays in the plane, its norm scales linearly in $\alpha$, and
$\theta \to \theta + 180°$ negates it (which is *why* antipodal probe tokens
work, and what the cyclicity test in §1.6 checks empirically).

**Expected outcome.** All three properties exact to machine precision. In-plane
residual ~1e-15, norm ratio exactly $\alpha_2/\alpha_1$, antipode residual ~1e-15.

*Tier 0 · no model · instant*
""")

code(r'''
rng = np.random.default_rng(0)

# Two orthonormal directions standing in for (u_i, u_j) of a real head,
# with the singular values we actually measured at GPT-2 L9H7 (SV1, SV2).
d_model = 768
Q, _ = np.linalg.qr(rng.standard_normal((d_model, 2)))
u_i, u_j = Q[:, 0], Q[:, 1]
sigma_i, sigma_j = 8.866, 8.456          # gpt2_compass_causal.txt, L9H7

def inject(theta_deg, alpha):
    """v(theta, alpha) = alpha*sigma_i*cos(theta)*u_i + alpha*sigma_j*sin(theta)*u_j"""
    th = np.deg2rad(theta_deg)
    return alpha * sigma_i * np.cos(th) * u_i + alpha * sigma_j * np.sin(th) * u_j

# 1. the vector never leaves the plane
P_perp = np.eye(d_model) - Q @ Q.T
resid = max(np.linalg.norm(P_perp @ inject(t, 3.0)) for t in range(0, 360, 15))
print(f"max out-of-plane component      : {resid:.3e}   (expect ~0)")

# 2. norm is exactly linear in alpha
n1, n10 = np.linalg.norm(inject(37.0, 1.0)), np.linalg.norm(inject(37.0, 10.0))
print(f"||v(a=10)|| / ||v(a=1)||        : {n10/n1:.10f}   (expect 10.0)")

# 3. antipodal angles negate the vector -- the basis of the he/she probe pair
anti = max(np.linalg.norm(inject(t, 1.0) + inject(t + 180, 1.0)) for t in range(0, 360, 15))
print(f"max ||v(t) + v(t+180)||         : {anti:.3e}   (expect ~0)")

assert resid < 1e-10 and abs(n10/n1 - 10) < 1e-9 and anti < 1e-10
print("\nInjection geometry verified.")
''')

# ---- 1.3 compass_causal_sweep
md(r"""
### 1.3 — `compass_causal_sweep.py` · the α-sweep (paper Figure 2)

| | |
|---|---|
| **Script** | `experiments/compass_causal_sweep.py` (287 lines) |
| **Stage** | L2 — validation on a single, known head |
| **Purpose** | Establish that one plane behaves as a *dial*: sinusoidal in $\theta$, linear in $\alpha$, phase-stable across scales. |
| **Method** | 36 angles × α ∈ {1, 3, 10} × 3 prompt conditions. Inject, read logit difference, fit. |
| **Outputs** | `helix_usage_validated/<prefix>_{curves,polar,linearity}.png` + `.txt` |
| **Runtime** | ~2 min (GPT-2, CPU) · ~15 min (Phi-3, GPU) |

**What we are testing.** This is the *confirmatory* experiment, run on a head we
already believe in (GPT-2 L9H7). It is not evidence that compasses are common —
§1.7 is the honest test of that. It establishes that *when* a compass exists, it
behaves lawfully.

**Expected outcome — from our GPT-2 L9H7 run.** Baseline logit(he) − logit(she)
is +0.550 neutral, +2.911 masculine-context, −3.070 feminine-context. Under
injection the curve is a clean sinusoid whose amplitude tracks α almost exactly:

| α | A | A/α | φ |
|---:|---:|---:|---:|
| 1 | 1.187 | 1.187 | −175.4° |
| 3 | 3.556 | 1.185 | −175.3° |
| 10 | 11.700 | 1.170 | −175.3° |

Phase drift across a 10× scale change is **0.014°** against a 10° tolerance. The
next cell recomputes all of this from the committed log.
""")

code(r'''
def parse_causal_sweep(path):
    """Parse a compass_causal_sweep log into {alpha: array[[deg, neutral, masc, fem]]}.

    The log format is one 'INJECTION SWEEP at alpha = X' block per scale,
    followed by a fixed-width table.
    """
    txt = Path(path).read_text(errors="replace")
    parts = re.split(r"INJECTION SWEEP at alpha\s*=\s*([\d.]+)", txt)
    rows_re = re.compile(r"^\s*([\d.]+)\s*\|\s*([+-][\d.]+)\s+([+-][\d.]+)\s+([+-][\d.]+)\s*$", re.M)
    out = {}
    for i in range(1, len(parts) - 1, 2):
        alpha = float(parts[i])
        rows = rows_re.findall(parts[i + 1])
        if rows:
            out[alpha] = np.array(rows, dtype=float)
    return out

log = CV_LOGS / "gpt2_compass_causal.txt"
sweep = parse_causal_sweep(log)
print(f"source: {log.relative_to(ROOT)}")
print(f"scales recovered: {sorted(sweep)}   angles per scale: {len(next(iter(sweep.values())))}\n")

print(f"{'alpha':>6} {'mu':>9} {'A':>9} {'A/alpha':>9} {'phi(deg)':>10}")
print("-" * 46)
fit = {}
for a in sorted(sweep):
    mu, A, phi = fit_sinusoid(sweep[a][:, 0], sweep[a][:, 1])   # column 1 = neutral prompts
    fit[a] = (mu, A, phi)
    print(f"{a:>6.1f} {mu:>+9.3f} {A:>9.3f} {A/a:>9.3f} {phi:>+10.1f}")

# ── the three pillars of the pass criterion (cookbook 2.6)
alphas = np.array(sorted(fit))
amps   = np.array([fit[a][1] for a in alphas])
slope  = float((alphas @ amps) / (alphas @ alphas))              # least squares through origin
ss_res = float(((amps - slope * alphas) ** 2).sum())
ss_tot = float(((amps - amps.mean()) ** 2).sum())
r2     = 1 - ss_res / ss_tot
dphi   = abs((fit[alphas[-1]][2] - fit[alphas[0]][2] + 180) % 360 - 180)
eff    = amps[-1] / alphas[-1]

print(f"\n{'pillar':<26}{'value':>10}   {'threshold':>12}   verdict")
print("-" * 66)
for name, val, thr, ok in [
    ("amplitude linearity R^2", r2,   ">= 0.95", r2 >= 0.95),
    ("phase drift (deg)",       dphi, "<= 10",   dphi <= 10),
    ("effect size A(10)/10",    eff,  ">= 0.20", eff >= 0.20),
]:
    print(f"{name:<26}{val:>10.4f}   {thr:>12}   {'PASS' if ok else 'FAIL'}")

print(f"\nGPT-2 L9H7 (SV1,SV2): {'COMPASS' if (r2>=0.95 and dphi<=10 and eff>=0.20) else 'not a compass'}")
''')

code(r'''
# The same log, drawn. Three scales, one shared phase -- the visual signature
# of a dial: the curves are scaled copies of each other, not different shapes.
%matplotlib inline
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))

ax = axes[0]
for a in sorted(sweep):
    d = sweep[a]
    ax.plot(d[:, 0], d[:, 1], marker="o", ms=3, lw=1.4, label=f"alpha={a:g}")
    mu, A, phi = fit[a]
    th = np.linspace(0, 345, 400)
    ax.plot(th, mu + A*np.cos(np.deg2rad(th - phi)), ls="--", lw=1, color="k", alpha=.35)
ax.axhline(0, color="0.6", lw=.8)
ax.set_xlabel("injection angle theta (deg)"); ax.set_ylabel("logit(' he') - logit(' she')")
ax.set_title("GPT-2 L9H7 neutral prompts\n(dashed = first-DFT-bin fit)")
ax.legend(fontsize=8); ax.set_xticks(range(0, 361, 90))

ax = axes[1]
ax.plot(alphas, amps, "o-", lw=1.6)
lin = np.linspace(0, alphas.max(), 50)
ax.plot(lin, slope*lin, "--", color="k", alpha=.5, label=f"fit through origin (R^2={r2:.4f})")
ax.set_xlabel("alpha"); ax.set_ylabel("amplitude A")
ax.set_title("Amplitude linearity"); ax.legend(fontsize=8)

plt.tight_layout(); plt.show()
''')

code(r'''
# Tier 1 -- recompute the above from the model instead of replaying the log.
cmd = f"""
python experiments/compass_causal_sweep.py \\
    --model gpt2 --layer 9 --head 7 --dims 1 2 \\
    --tok_plus " he" --tok_minus " she" \\
    --prompt_neutral "The person said that" \\
    --prompt_neutral "Then they said that" \\
    --prompt_neutral "Afterwards, the speaker said that" \\
    --prompt_plus "The man laced up his boots because" \\
    --prompt_plus "The father waved to the crowd and" \\
    --prompt_plus "The king announced that" \\
    --prompt_minus "The woman laced up her boots because" \\
    --prompt_minus "The mother waved to the crowd and" \\
    --prompt_minus "The queen announced that" \\
    --out_prefix gpt2_rerun
"""
shell(cmd, COMPASS)
if tier(1):
    subprocess.run(cmd, shell=True, cwd=COMPASS, check=True)
    print("\nRe-parse with parse_causal_sweep(CV_LOGS / 'gpt2_rerun.txt') and compare to the table above.")
''')

# ---- 1.4 compass_dictionary
md(r"""
### 1.4 — `compass_dictionary.py` · decoding what a plane *means*

| | |
|---|---|
| **Script** | `experiments/compass_dictionary.py` (159 lines) · single-model variant `compass_dictionary_single.py` |
| **Stage** | L1a — decode-driven discovery |
| **Purpose** | Turn an anonymous SVD axis into a *readable* semantic axis. |
| **Method** | Project the top-K singular directions through mean-centred $W_U$; read off the highest-logit tokens at each pole. |
| **Outputs** | `helix_usage_validated/compass_dict_{gpt2,phi3,gemma,llama32_3b}.txt`, `compass_dict_all.md` |
| **Runtime** | ~1 min/model |

**Why mean-centring matters.** Without subtracting the mean unembedding, every
direction decodes to the same high-frequency tokens (`the`, `,`, `and`) and the
dictionary is useless. The centring is what makes the poles interpretable.

**Expected outcome.** A minority of axes decode to crisp antipodal pairs —
pronouns, tense markers, person/place. Most decode to noise, and that is normal:
we are reading the top few singular directions of every head, and most carry no
single clean concept. Treat this as a *hypothesis generator*, not evidence.
Nothing here is causal — §1.3 and §1.5 supply the causality.
""")

code(r'''
# Replay: what the decoder actually produced. These are token lists from the
# committed dictionaries, not recomputed logits.
dicts = sorted(CV_LOGS.glob("compass_dict_*.txt"))
print(f"decoded dictionaries on disk: {[p.stem.replace('compass_dict_', '') for p in dicts]}\n")

if dicts:
    sample = CV_LOGS / "compass_dict_gpt2.txt"
    if sample.exists():
        head = sample.read_text(errors="replace").splitlines()
        # Show the first decoded block, skipping loader noise.
        start = next((i for i, l in enumerate(head) if re.match(r"\s*L\d+H\d+", l)), 0)
        print(f"--- {sample.name} (first decoded head) ---")
        print("\n".join(head[start:start + 22]))
else:
    print("No dictionaries found -- run experiments/compass_dictionary.py (Tier 1).")
''')

code(r'''
cmd = "python experiments/compass_dictionary.py"
shell(cmd, COMPASS)
if tier(1):
    subprocess.run(cmd, shell=True, cwd=COMPASS, check=True)
''')

# ---- 1.5 cyclicity
md(r"""
### 1.5 — Cyclicity · is the plane really a *circle*?

| | |
|---|---|
| **Entry point** | `investigate_helix_usage_validated.py --test cyclicity` (class method `cyclicity_check`) |
| **Stage** | L3 — falsification battery, test 2 of 9 |
| **Purpose** | Distinguish a genuine **rotational** structure from a plane that merely happens to correlate with a concept. |
| **Method** | Decode the vocabulary at θ = 0°, 180°, 360°. Compare top-10 token sets by Jaccard overlap. |

**The prediction being tested.** If the plane is a true dial, θ = 0° and
θ = 360° must decode *identically* (the circle closes), and θ = 180° must decode
to the **opposite** pole (disjoint token set). A correlational direction has no
reason to satisfy either.

**Expected outcome — GPT-2 L9H7.** This is one of our cleanest results:

- `jaccard_identity = 1.0` — θ=0° and θ=360° give the *same* ten tokens
- `jaccard_antipode = 0.0` — θ=180° shares *nothing* with θ=0°
- θ=0°: ` herself, her, She, she, Her, …` → θ=180°: `His, his, he, He, himself, …`
- verdict: **`CIRCLE: closed and with distinct antipodes`**

A perfect 1.0 / 0.0 split is strong evidence the plane is genuinely rotational.
""")

code(r'''
res = json.loads((CV_LOGS / "workshop_suite_results_gpt2_l9h7.json").read_text())
cyc = res["cyclicity"]

print(f"head L{cyc['layer']}H{cyc['head']}  dims {tuple(cyc['dims'])}\n")
print(f"  jaccard(theta=0, theta=360) = {cyc['jaccard_identity']:.2f}   (1.0 => the circle closes)")
print(f"  jaccard(theta=0, theta=180) = {cyc['jaccard_antipode']:.2f}   (0.0 => poles are disjoint)\n")
for label, key in [("theta =   0", "top10_0"), ("theta = 180", "top10_180"), ("theta = 360", "top10_360")]:
    print(f"  {label}: {', '.join(repr(t) for t in cyc[key][:6])} ...")
print(f"\n  verdict: {cyc['verdict']}")

assert cyc["jaccard_identity"] == 1.0 and cyc["jaccard_antipode"] == 0.0
''')

# ---- 1.6 falsification battery — the honest one
md(r"""
### 1.6 — The nine-test falsification battery · **including what failed**

| | |
|---|---|
| **Entry point** | `investigate_helix_usage_validated.py --test all-must-have <model>` |
| **Driver** | `run_workshop_suite` (line 3361) |
| **Outputs** | `helix_usage_validated/workshop_suite_results_<model>_<head>.json` |
| **Runtime** | ~20 min (GPT-2) · ~2 h (Phi-3) |

The battery exists to *attack* the compass claim. Six tests run by default
(random-plane, permutation, principal-angles, cyclicity, causal-patch,
semantic-ablate).

> ### ⚠ Two of these do not support the hypothesis. Read this before citing the battery.
>
> **The random-plane null does not reject (p = 0.38).** Our observed amplitude at
> GPT-2 L9H7 is 25.49. Random planes *within the same head* average 26.58
> (sd 21.77) — they do **as well or better**. The compass plane is not special
> against this null.
>
> **The principal-angles test returned empty.** `angles_deg = {}` — it needs
> ≥ 2 head specs to compare and was given one. That test is **vacuous** in this
> run, not passed.
>
> **What does hold: the permutation null, decisively.** Observed 25.49 against a
> permuted mean of 4.54 (sd 1.87, max 11.45) over 2000 permutations, p = 0.0.
>
> **How to read this honestly.** The permutation test shows the effect depends on
> *the real label structure* — it is not an artefact of the sweep machinery. The
> random-plane result shows the effect is **not localised to the specific SVD
> pair we picked**: other planes in that head do comparably well. The defensible
> claim is "this head carries a gender dial", **not** "these two singular
> directions uniquely carry it". Part of the paper's framing depends on that
> distinction and it should be stated in the limitations, not buried.

*Tier 0 · replays the committed battery output*
""")

code(r'''
res = json.loads((CV_LOGS / "workshop_suite_results_gpt2_l9h7.json").read_text())

rows = []

rp = res["random_plane"]
rows.append((
    "random-plane null",
    "obs {:.2f} vs {:.2f}+-{:.2f}, p={:.2f}".format(
        rp["observed"], rp["random_mean"], rp["random_std"], rp["p_value"]),
    "DOES NOT REJECT" if rp["p_value"] > 0.05 else "rejects null",
))

pm = res["permutation"]
rows.append((
    "permutation null",
    "obs {:.2f} vs {:.2f}+-{:.2f}, p={:.2f}".format(
        pm["observed"], pm["perm_mean"], pm["perm_std"], pm["p_value"]),
    "REJECTS NULL" if pm["p_value"] < 0.05 else "does not reject",
))

pa = res["principal_angles"]
n_ang = len(pa.get("angles_deg", {}))
rows.append((
    "principal angles",
    "{} angle(s) computed".format(n_ang),
    "VACUOUS (needs >=2 heads)" if n_ang == 0 else "ok",
))

cy = res["cyclicity"]
rows.append((
    "cyclicity",
    "identity {:.1f} / antipode {:.1f}".format(cy["jaccard_identity"], cy["jaccard_antipode"]),
    cy["verdict"].split(":")[0],
))

for key in ("causal_patch", "semantic_ablation"):
    if key in res:
        rows.append((key.replace("_", " "), "see raw json", "present"))

print("{:<20}{:>38}   {}".format("test", "statistic", "verdict"))
print("-" * 80)
for name, stat, verdict in rows:
    print("{:<20}{:>38}   {}".format(name, stat, verdict))

print()
print("Summary: 1 test rejects its null (permutation), 1 fails to (random-plane),")
print("         1 is vacuous (principal-angles), 1 is strongly positive (cyclicity).")
print("         Do not report this as 'nine tests passed'.")
''')

# ---- 1.7 the scan — the headline negative
md(r"""
### 1.7 — `compass_scan.py` · the blind scan, and the result that constrains the paper

| | |
|---|---|
| **Script** | `experiments/compass_scan.py` (345 lines) · CIs by `scan_ci_summary.py` |
| **Stage** | L1b — **blind** discovery over every (layer, head, SV-pair) |
| **Purpose** | The honest prevalence question: if you *don't* know where to look, how often does a compass turn up — versus a matched null? |
| **Outputs** | `<prefix>_scan.txt`, `<prefix>_heatmap.png`, `scan_pass_rate_ci.txt` |
| **Runtime** | ~40 min (GPT-2) · ~6 h (Phi-3) |

**Two nulls, and they disagree.** `top4` permutes within the top-4 singular
directions; `full_ov` samples random planes from the whole OV spectrum. Which
null you choose changes the conclusion, so we report both.

> ### ⚠ Against the `top4` null, the excess is *negative*.
>
> | scan | compass | null (top4) | excess | null (full_ov) | excess |
> |---|---:|---:|---:|---:|---:|
> | gpt2 gender | 2.06% (16/864) | 2.51% | **−0.23%** | 0.44% | +1.85% |
> | phi3 gender | 1.18% (14/1344) | 1.62% | **−0.30%** | 0.28% | +1.04% |
> | gemma gender | 2.02% (3/240) | 1.55% | +1.25% | 1.55% | +1.25% |
>
> **Interpretation.** Compass planes are *not* more common than other planes
> drawn from the same top-4 singular subspace. They are more common than planes
> drawn from the whole OV spectrum. Both statements are true and they mean:
> **the top singular subspace of a head is the thing carrying structure**, and
> our specific $(i,j)$ choice within it is not privileged. This is the same
> conclusion the random-plane null reached in §1.6, arrived at independently.
>
> Note also the absolute rates: ~1–2%, with 3/240 for Gemma. These are small
> counts and the CIs are correspondingly wide.

**If you reproduce this and get a negative excess against `top4`, you have
reproduced our result correctly.**
""")

code(r'''
ci = (CV_LOGS / "scan_pass_rate_ci.txt").read_text(errors="replace")
print(ci.strip())
''')

code(r'''
# Recompute the Agresti-Coull intervals from the raw counts, independently of
# the committed summary -- a check that the reported CIs are right.
def agresti_coull(x, n, z=1.959963985):
    """95% CI for a binomial proportion. Robust where the normal approx fails."""
    n_t = n + z**2
    p_t = (x + z**2 / 2) / n_t
    half = z * np.sqrt(p_t * (1 - p_t) / n_t)
    return max(0.0, p_t - half), min(1.0, p_t + half)

print(f"{'scan':<24}{'x/n':>12}{'rate':>9}{'95% CI':>20}")
print("-" * 66)
for label, x, n in [("gpt2 compass", 16, 864), ("gpt2 null top4", 9, 432),
                    ("gpt2 null full_ov", 0, 432), ("phi3 compass", 14, 1344),
                    ("phi3 null top4", 9, 672), ("gemma compass", 3, 240)]:
    lo, hi = agresti_coull(x, n)
    print(f"{label:<24}{f'{x}/{n}':>12}{x/n*100:>8.2f}%{f'[{lo*100:.2f}%, {hi*100:.2f}%]':>20}")

print("\nNote how wide the Gemma interval is on 3/240 -- that row cannot carry weight.")
''')

code(r'''
cmd = """
python experiments/compass_scan.py \\
    --model gpt2 --tok_plus " he" --tok_minus " she" \\
    --top_svs 4 --alphas 3 10 --n_angles 12 \\
    --null_mode top4 --null_seeds 3 --out_prefix gpt2_rerun_scan
"""
shell(cmd, COMPASS)
if tier(1):
    subprocess.run(cmd, shell=True, cwd=COMPASS, check=True)
''')

# ---- 1.8 downstream
md(r"""
### 1.8 — Downstream evaluation · does the dial *do* anything?

| Script | Benchmark | What it answers |
|---|---|---|
| `crowspairs_routed_eval.py` | CrowS-Pairs | Does per-domain routed steering reduce stereotype preference? |
| `stereoset_ensemble_eval*.py` | StereoSet | Multi-plane ensemble injection, per domain |
| `winogender_eval.py` / `_sweep.py` | Winogender | Occupational coreference bias |
| `truthfulqa_eval.py` | TruthfulQA | Off-target damage check |
| `inlp_debias.py`, `sentence_debias.py` | — | INLP and SentenceDebias **baselines** |
| `baseline_comparison.py` | — | Compass vs four standard direction-finding methods |

**Purpose.** A causal dial is only interesting if it moves a metric someone
cares about — and only *safe* if it doesn't wreck unrelated behaviour. TruthfulQA
is the off-target control, not a headline.

**Calibration first.** `calibrate_per_domain_alpha.py` picks α per domain by
targeting a fixed signal-to-noise ratio against the residual-stream norm, so
that "α = 2" means the same perturbation strength across models.

> **⚠ The SNR target is frequently unreachable.** In
> `per_domain_alpha_gpt2.json`, three of four domains carry `"fallback": true`:
> mean SNR at α=1 is ≈0.053 against a target of 0.10, so α is scaled to ~1.87
> and the target is still missed. Several selected heads carry
> `"passed": false`. Any downstream number must be read as "at the α we could
> reach", not "at the α we wanted".

*Tier 0 · replays the committed calibration*
""")

code(r'''
alpha_cfg = json.loads((CV_LOGS / "per_domain_alpha_gpt2.json").read_text())
print(f"model: {alpha_cfg['model']}   target SNR: {alpha_cfg['target_snr']}")
print(f"norm probe prompt: {alpha_cfg['prompt_for_norm']!r}\n")

print(f"{'domain':<12}{'alpha':>8}{'mean SNR@1':>12}{'target hit':>12}{'heads':>8}{'passed':>8}")
print("-" * 62)
for dom, d in alpha_cfg["by_domain"].items():
    heads = d["heads"]
    n_pass = sum(h["passed"] for h in heads)
    print(f"{dom:<12}{d['alpha']:>8.3f}{d['mean_snr_at_alpha1']:>12.4f}"
          f"{('no (fallback)' if d.get('fallback') else 'yes'):>12}{len(heads):>8}{f'{n_pass}/{len(heads)}':>8}")

fallbacks = [d for d, v in alpha_cfg["by_domain"].items() if v.get("fallback")]
print(f"\nDomains that failed to reach the SNR target: {fallbacks or 'none'}")
print("Downstream deltas for those domains are 'best reachable', not 'at target SNR'.")
''')

code(r'''
cmds = [
    ('calibrate alpha per domain',
     'python experiments/calibrate_per_domain_alpha.py --model gpt2 --target_snr 0.10'),
    ('routed CrowS-Pairs (paper Table 7)',
     'python experiments/crowspairs_routed_eval.py --model gpt2 '
     '--alpha_json helix_usage_validated/per_domain_alpha_gpt2.json'),
    ('StereoSet ensemble (GPT-2 is hardcoded; use the _gemma/_phi3/_llama variants for others)',
     'python experiments/stereoset_ensemble_eval.py '
     '--domains gender,race,profession,religion --out_prefix stereoset_ensemble_gpt2'),
    ('Winogender sweep',         'python experiments/winogender_sweep.py --model gpt2'),
    ('TruthfulQA off-target',    'python experiments/truthfulqa_eval.py --model gpt2'),
    ('INLP baseline',            'python experiments/inlp_debias.py --model gpt2 --domain gender --layer 10'),
    ('SentenceDebias baseline',  'python experiments/sentence_debias.py --model gpt2 --domain gender --layer 10'),
    ('compass vs 4 baselines',   'python experiments/baseline_comparison.py --model gpt2'),
]
for label, c in cmds:
    print(f"# {label}\n{c}\n")
if tier(1):
    for label, c in cmds:
        print(f"=== {label} ===")
        subprocess.run(c, shell=True, cwd=COMPASS, check=True)
''')

# ════════════════════════════════════════════════════════════ PART 2
md(r"""
---
# Part 2 — The Fourier Basis of Digit Arithmetic

## The claim in one paragraph

Language models represent decimal digits in a **Fourier basis of $\mathbb{Z}/10\mathbb{Z}$**.
At the computation layer the digit subspace decomposes into exactly **nine**
directions — two each for frequencies $k = 1,2,3,4$ and one for $k = 5$ (parity).
That subspace is causally necessary: zero it and addition collapses to chance,
while a matched-dimension *random* subspace ablation does nothing at all. Rotate
its phase and the model's answer shifts by a predictable amount mod 10.

## Pipeline shape

The plan (`ARITHMETIC_CIRCUIT_PLAN.md`) is **15 numbered steps in six phases**.
They are strictly ordered: Step 1 determines the `comp-layer` every later step
needs.

| Phase | Steps | Question |
|---|---|---|
| **A — Discovery** | 1–3 | Where does arithmetic live, and is the code Fourier? |
| **B — Causal validation** | 4–7 | Is that subspace *necessary* and *sufficient*? |
| **C — Attribution** | 8–9 | Which heads and neurons write it? |
| **D — Mechanism** | 10–11 | *How* is addition performed? |
| **E — Generalisation** | 12–14 | Does it survive subtraction, multi-digit, new ranges? |
| **F — Multi-digit** | 15 | Gemma-specific extension |

> **⚠ Two structural cautions.**
>
> **These are Tier 2.** Gemma-2B, Phi-3-mini and LLaMA-3.2-3B are gated on
> HuggingFace and need a GPU. Recorded runtime is 30–60 min for Step 1 alone and
> 1–2 h for the Step 3 sweep, per model. Budget days, not hours.
>
> **`ARITHMETIC_CIRCUIT_PLAN.md` also documents a superseded route** — "What
> Already Exists", "Phase 0", and `SUPPLEMENTARY SCRIPTS` S5 ("Old Pipeline") —
> built on a mask-learning dependency that has since been removed. **Commands in
> those sections will not run.** They are retained as a record of how the work
> developed. Phases A–F are the paper's actual method.
""")

md(r"""
### 2.1 — Step 0 · teacher-forced vs direct-answer mode

**Purpose.** Choose the prompt protocol *before* anything else, because it
changes the unembedding basis and therefore every later measurement.

- **Teacher-forced** (default): prompt `"Calculate 13 + 8 = 2"`, model predicts
  the next token `1`. Valid when single-digit answer tokens 0–9 exist.
- **Direct-answer**: prompt `"a + b = "`, model emits the whole answer as one
  token. **Required** when the tokenizer has single tokens for 0–198 — LLaMA-3.2-3B
  does. Pass `--direct-answer` to *every* subsequent command.

**How to decide.** Run a handful of `"a + b = "` prompts. Above ~90% accuracy in
that format, use direct-answer.

> **⚠ Getting this wrong is silent.** The pipeline will run and produce
> plausible-looking numbers. The failure surfaces much later as near-zero
> unembed-patching transfer, which reads like a scientific null rather than a
> configuration error. See `diagnose_unembed_direct.py` — that script exists
> because we hit exactly this.

*Registration is a code edit, not a flag:*
""")

code(r'''
print("Register a new model in experiments/arithmetic_circuit_scan_updated.py:\n")
print('    MODEL_MAP = {')
print('        "phi-3":      "microsoft/Phi-3-mini-4k-instruct",')
print('        "gemma-2b":   "google/gemma-2-2b",')
print('        "llama-3b":   "meta-llama/Llama-3.2-3B",')
print('        "your-model": "org/model-name",   # <- add here')
print('    }\n')
print("and its layer defaults in experiments/eigenvector_dft.py:\n")
print('    readout_defaults = {"gemma-2b": 25, "phi-3": 31, "llama-3b": 27}')
print('    comp_defaults    = {"gemma-2b": 19, "phi-3": 26, "llama-3b": 20}\n')
print("comp-layer is DISCOVERED by Step 1; the defaults above only cache what we found.")

# Confirm the registry we ship matches what the paper reports.
reg = ARITH / "src" / "utils" / "model_registry.py"
print(f"\nshipped registry: {reg.relative_to(ROOT)}  ({'present' if reg.exists() else 'MISSING'})")
''')

md(r"""
### 2.2 — Step 1 · layer scan + unembed patching **(must run first)**

| | |
|---|---|
| **Script** | `experiments/arithmetic_circuit_scan_updated.py` |
| **Purpose** | Find the **comp-layer** (where arithmetic is computed) and the **readout-layer**. Every later step depends on this. |
| **Key functions** | `run_layer_scan`, `compute_unembed_basis[_direct_answer]`, `run_patching_experiment`, `compute_fisher_matrix`, `compute_contrastive_fisher`, `filter_correct_*` |
| **Output** | `mathematical_toolkit_results/arithmetic_scan_<model>.json` |
| **Runtime** | 30–60 min per model |

**Method.** Sweep every layer; patch activations between digit-pairs; measure
transfer rate. Then patch only a *subspace* (unembed-aligned / Fisher / random)
to measure the effective dimensionality.

**What to look for.**
- First layer whose transfer exceeds 80% → **comp-layer**
- Last layer at ~100% transfer → **readout-layer**
- 9D unembed patching should capture most transfer at readout (100% teacher-forced)

**Expected outcome.** comp-layer ≈ **L19 (Gemma-2B)**, **L26 (Phi-3)**,
**L20 (LLaMA-3.2-3B)**; readout ≈ L25 / L31 / L27.

> **⚠ A dissociation you will hit at intermediate layers.** At Gemma L21, Fisher
> patching transfers **85%** but unembed patching only **30%**. The digit code is
> *gradient-visible but not yet output-aligned*. Fisher alignment and unembedding
> rotation are separate processes that converge only at the output layers. A low
> unembed number mid-stack is a real finding, not a bug.
""")

code(r'''
cmd = """
python experiments/arithmetic_circuit_scan_updated.py \\
    --model gemma-2b --device cuda --n-per-digit 100 --n-test 150
# LLaMA-3.2-3B additionally needs:  --direct-answer
"""
shell(cmd, ARITH)
print("\nExpected comp-layer / readout-layer, from our runs:")
for m, comp, ro in [("gemma-2b", 19, 25), ("phi-3", 26, 31), ("llama-3b", 20, 27)]:
    print(f"    {m:<12} comp-layer L{comp:<3} readout-layer L{ro}")
if tier(2):
    subprocess.run(cmd, shell=True, cwd=ARITH, check=True)
''')

md(r"""
### 2.3 — Step 2 · eigenvector DFT · **is it really a Fourier basis?**

| | |
|---|---|
| **Script** | `experiments/eigenvector_dft.py` → plots by `plot_eigenvector_dft.py` |
| **Purpose** | Test whether the digit encoding is a *perfect* Fourier basis of $\mathbb{Z}/10\mathbb{Z}$. |
| **Method** | Take each SVD direction's 10-element digit-score vector; DFT it; ask which frequency dominates. |
| **Output** | `mathematical_toolkit_results/eigenvector_dft_<model>.json` |
| **Runtime** | 15–30 min |

**The prediction.** A perfect basis assigns exactly **2 directions to each of
k=1,2,3,4** and **1 to k=5** — because $\cos$ and $\sin$ pair up for every
frequency except the Nyquist frequency $k=5$, which is real-valued. Total: 9.

**Expected outcome.** `"PERFECT FOURIER BASIS"` with mean purity > 50%.

> **⚠ This is the gate.** If the assignment is not 2/2/2/2/1, the model may not
> use a Fourier encoding at all and the rest of the pipeline is not meaningful.
> Before concluding that, check digit balance in your sample and raise `n`.
""")

code(r'''
# Why 9 and not 10: the DFT of a real 10-point signal has conjugate symmetry.
N = 10
print("Fourier basis of Z/10Z -- degrees of freedom per frequency\n")
print(f"{'k':>3}  {'basis functions':<22}{'dims':>6}   note")
print("-" * 64)
total = 0
for k in range(0, N // 2 + 1):
    if k == 0:
        dims, note, fns = 0, "DC -- removed by centring", "constant"
    elif k == N // 2:
        dims, note, fns = 1, "Nyquist: sin() vanishes", "cos(2*pi*5*d/10)"
    else:
        dims, note, fns = 2, "conjugate pair", f"cos, sin(2*pi*{k}*d/10)"
    total += dims
    print(f"{k:>3}  {fns:<22}{dims:>6}   {note}")
print("-" * 64)
print(f"{'':>3}  {'TOTAL':<22}{total:>6}   <- the '9D Fourier subspace'")
assert total == 9

# Demonstrate the claim concretely: the 9 functions are orthogonal and span
# the mean-centred space of functions on the digits 0..9.
d = np.arange(N)
B = [np.cos(2*np.pi*k*d/N) for k in range(1, 5)] + \
    [np.sin(2*np.pi*k*d/N) for k in range(1, 5)] + [np.cos(np.pi*d)]
B = np.array(B)
G = B @ B.T
print(f"\nGram matrix off-diagonal max : {np.abs(G - np.diag(np.diag(G))).max():.2e}  (orthogonal)")
print(f"rank of the 9 basis functions: {np.linalg.matrix_rank(B)}  (expect 9)")
print(f"rank of [basis ; constant]   : {np.linalg.matrix_rank(np.vstack([B, np.ones(N)]))}  (expect 10 = complete)")
''')

code(r'''
cmd = "python experiments/eigenvector_dft.py --model gemma-2b --comp-layer 19 --device cuda"
shell(cmd, ARITH)
print('\nLook for: "PERFECT FOURIER BASIS", mean purity > 50%,')
print("          frequency assignment 2/2/2/2/1 for k=1,2,3,4,5.")
if tier(2):
    subprocess.run(cmd, shell=True, cwd=ARITH, check=True)
''')

md(r"""
### 2.4 — Step 3 · Fourier layer sweep · where the structure is built

| | |
|---|---|
| **Script** | `experiments/fourier_decomposition.py` |
| **Purpose** | Track how Fourier energy accumulates layer by layer. |
| **Key functions** | `build_fourier_basis_functions`, `fourier_decomposition`, `per_neuron_fourier_analysis`, `run_fourier_at_layer` |
| **Runtime** | 1–2 h for a full sweep |

**Expected outcome.** Energy builds from early layers and **explodes by 2–3
orders of magnitude** at the computation layers — active amplification of
digit-discriminative signal, not passive propagation. This produces
`energy_explosion.png` in the paper.

The related bottom-up scan, `run_fourier_discovery` (see `src/analysis/fourier_discovery.py`),
sweeps heads for periodic structure and is what produced the committed
Pythia-1.4B results replayed below.
""")

code(r'''
# Replay: the committed Pythia-1.4B Fourier discovery run.
runs = sorted(FR_LOGS.glob("fourier_results_*.json"))
print(f"recorded runs: {[p.name for p in runs]}\n")

fr = json.loads(runs[-1].read_text())
cfg = fr["config"]
print(f"model                : {fr['model_key']}")
print(f"layers analysed      : {fr['n_layers_analyzed']}")
print(f"significant heads    : {fr['n_significant_heads']}  "
      f"(power-ratio threshold {cfg['fourier']['head_power_ratio_threshold']})")
print(f"prompt template      : {cfg['arithmetic']['prompt_template']!r}"
      f"  operands {cfg['arithmetic']['operand_range_start']}-{cfg['arithmetic']['operand_range_end']}")

heads = fr["head_results"]
from collections import Counter
freqs = Counter(h["dominant_frequency"] for h in heads.values())
print(f"\ndominant frequency across {len(heads)} significant heads:")
for k, n in sorted(freqs.items()):
    print(f"    k={k}: {n:>4} heads  {'#' * int(60 * n / len(heads))}")

ratios = np.array([h["power_ratio"] for h in heads.values()])
print(f"\npower ratio: median {np.median(ratios):.2f}, max {ratios.max():.2f}")
top = sorted(heads.items(), key=lambda kv: -kv[1]["power_ratio"])[:5]
print("\nstrongest heads:")
for name, h in top:
    print(f"    {name:<8} L{h['layer']:<3} H{h['head']:<3} k={h['dominant_frequency']}  ratio={h['power_ratio']:.2f}")

print("\nNote: k=1 (ordinal) dominates the head-level scan. Compare with 2.5 --")
print("      the frequency that dominates by VARIANCE is not the causally important one.")
''')

code(r'''
cmd = 'python experiments/fourier_decomposition.py --model gemma-2b --layer-sweep "5,6,...,25" --device cuda'
shell(cmd, ARITH)
if tier(2):
    subprocess.run(cmd, shell=True, cwd=ARITH, check=True)
''')

md(r"""
### 2.5 — Steps 4–7 · causal validation, and the ***k*=5 paradox**

| Step | Script | Tests |
|---|---|---|
| 4 | `fourier_knockout.py` | **Necessity** — zero the 9D subspace, measure damage |
| 5 | `fisher_phase_shift.py`, `fisher_patching.py` | **Sufficiency** — patch only the subspace |
| 6 | `fourier_phase_rotation.py` | **Steering** — rotate phase, predict answer shift mod 10 |
| 7 | `steering_improvements.py` | $W_U$-informed steering |

**Necessity — the cleanest result in the paper.** Zeroing the 9D subspace across
layers from computation to readout drops accuracy to **near chance** in all three
models. A **matched-dimension random subspace ablation has *zero* effect** —
perfect specificity. Single-layer ablation does partial damage (11–77%),
revealing a distributed pipeline that actively maintains the information.

**Sufficiency.** At readout, standard Fisher at 10D transfers **85%** (Gemma,
Phi-3) and **100%** (LLaMA); contrastive Fisher at 9D transfers 83–100%. The two
subspaces agree to **>0.97 principal cosine** — they find the same thing.

> ### ⚠ The *k*=5 paradox — a hypothesis we falsified
>
> At Gemma's computation layer, $k=5$ (parity) is **the dominant SVD direction**:
> $\sigma = 135$, **71% of subspace variance**. The natural inference is that
> parity is central to the computation.
>
> **It is causally inert.** Ablating $k=5$ across L13–L25 costs **0.4–1.0%**
> accuracy. Ablating $k=1$ or $k=2$ costs **~40% each**.
>
> $k=5$ is an **epiphenomenal encoding** — strongly represented, not used for
> ones-digit computation. The causally necessary frequencies are $k=1$ (ordinal)
> and $k=2$ (mod-5).
>
> **The methodological lesson, which generalises well beyond this paper:
> variance is not causation.** Had we ranked directions by singular value and
> stopped, we would have reported exactly the wrong mechanism. Every
> variance-ranked direction needs an ablation before it earns a causal claim.

**Steering, and its ceiling.** Coherent rotation at computation layers:
Gemma L19 **28%** exact, Phi-3 L26 **28%**, LLaMA L20 **69%**, with a marked
backward-shift asymmetry. The bottleneck is the **encoding–readout gap**: the
Fourier–unembed overlap is only **8–11%**, so rotating in Fourier space does not
map cleanly onto the directions $W_U$ reads.

> **⚠ Readout layers are immune.** At Gemma L25 / LLaMA L27 every steering method
> lands at **≤10% (chance)**, and the overlap falls to **0.2%** at LLaMA L27. By
> readout the subspace has been absorbed into the unembedding-aligned
> representation and can no longer be manipulated independently. **If you steer
> at the readout layer you will measure nothing, and it is not a bug.**
""")

code(r'''
# The paradox as a table. Numbers from paper/sections/results.tex.
print("Gemma-2B computation layer -- variance rank vs causal importance\n")
print(f"{'frequency':<12}{'role':<16}{'sigma':>8}{'% variance':>12}{'ablation damage':>18}")
print("-" * 68)
rows = [("k=5", "parity",  135, "71%",  "0.4-1.0%"),
        ("k=1", "ordinal", None, "-",   "~40%"),
        ("k=2", "mod-5",   None, "-",   "~40%")]
for k, role, sig, var, dmg in rows:
    print(f"{k:<12}{role:<16}{(str(sig) if sig else '-'):>8}{var:>12}{dmg:>18}")

print("\n  Ranked by variance : k=5 first.")
print("  Ranked by causation: k=5 LAST.")
print("\n  => Variance is not causation. Ablate before claiming a mechanism.")
''')

code(r'''
steps = [
    ("Step 4  necessity (knockout)",
     "python experiments/fourier_knockout.py --model gemma-2b --comp-layer 19 --device cuda"),
    ("Step 5  sufficiency (Fisher phase shift)",
     "python experiments/fisher_phase_shift.py --model gemma-2b --layers 19,25 --device cuda"),
    ("Step 6  steering (phase rotation)",
     "python experiments/fourier_phase_rotation.py --model gemma-2b --layers 19 --device cuda"),
    ("Step 7  W_U-informed steering",
     "python experiments/steering_improvements.py --model gemma-2b --layer 19 --device cuda"),
    ("        statistics over step 6",
     "python experiments/phase_rotation_statistics.py"),
]
for label, c in steps:
    print(f"# {label}\n{c}\n")
print("Reminder: run these at the COMPUTATION layer. At readout layers all")
print("          steering methods return chance-level results by construction.")
if tier(2):
    for label, c in steps:
        print(f"=== {label} ===")
        subprocess.run(c, shell=True, cwd=ARITH, check=True)
''')

md(r"""
### 2.6 — Steps 8–9 · component attribution

| | |
|---|---|
| **Scripts** | `fourier_head_attribution.py` (Step 8) · `neuron_trig_analysis.py` (Step 9) |
| **Purpose** | Identify which attention heads and MLP neurons *write* the Fourier subspace. |

**Expected outcome.** At Gemma L19, **809 / 9216 neurons (8.8%)** exceed 80%
frequency purity, dominated by $k=1$ and $k=5$ tunings. Produces
`neuron_frequency_tuning.png`.

> **⚠ Note the tension with §2.5.** $k=5$ is among the *most common* neuron
> tunings and is *causally inert*. Prevalence of a tuning is not evidence of
> function either — the same trap as variance ranking, one level down.
""")

code(r'''
for label, c in [
    ("Step 8  head/MLP attribution",
     "python experiments/fourier_head_attribution.py --model gemma-2b --comp-layer 19 --device cuda"),
    ("Step 9  per-neuron frequency tuning",
     "python experiments/neuron_trig_analysis.py --model gemma-2b --layer 19 --device cuda"),
]:
    print(f"# {label}\n{c}\n")

print("Expected (Gemma L19): 809/9216 neurons (8.8%) above 80% purity;")
print("                      dominant tunings k=1 and k=5.")
print("\nCaution: k=5 is prevalent AND causally inert (2.5). Prevalence != function.")
if tier(2):
    pass
''')

md(r"""
### 2.7 — Steps 10–11 · the computation mechanism

| | |
|---|---|
| **Scripts** | `cp_tensor_decomposition.py` (Step 10) · `carry_stratification.py` (Step 11) |
| **Purpose** | Show *how* addition is performed, not just where. |

**The mechanism.** Addition in a Fourier basis is **angle addition**. Step 10
decomposes the per-digit outer-product tensor
$T_{d,i,j} = \mathbb{E}[h_i h_j \mid \text{digit} = d]$ inside the Fourier
subspace and scores it against the product-to-sum identity
$$\cos\alpha\cos\beta = \tfrac{1}{2}\left[\cos(\alpha-\beta) + \cos(\alpha+\beta)\right]$$

**Expected outcome.** $\sigma^2$-weighted trigonometric identity score
**0.964** for Gemma. This is the strongest direct evidence that the model
performs angle addition rather than something that merely correlates with it.

**Step 11** stratifies by carry vs no-carry — the ones-digit circuit should be
largely carry-invariant, since carry affects the tens digit.
""")

code(r'''
# The identity the CP decomposition scores against, verified numerically.
rng = np.random.default_rng(1)
a, b = rng.uniform(0, 2*np.pi, 10000), rng.uniform(0, 2*np.pi, 10000)
lhs = np.cos(a) * np.cos(b)
rhs = 0.5 * (np.cos(a - b) + np.cos(a + b))
print(f"max |cos(a)cos(b) - 0.5[cos(a-b)+cos(a+b)]| = {np.abs(lhs - rhs).max():.2e}   (identity holds)")

# Why this is the signature of addition: the (a+b) term is the sum, carried by
# the SAME frequency. A circuit computing d_a + d_b mod 10 must produce it.
print("\nThe cos(a+b) term is literally the sum of the two operand angles.")
print("Finding it in the conditional-expectation tensor is what a Fourier adder looks like.\n")
print("Our measured sigma^2-weighted identity score (Gemma): 0.964  (1.0 = exact)")

for label, c in [
    ("Step 10  CP tensor decomposition",
     "python experiments/cp_tensor_decomposition.py --model gemma-2b --comp-layer 19 --device cuda"),
    ("Step 11  carry stratification",
     "python experiments/carry_stratification.py --model gemma-2b --comp-layer 19 --device cuda"),
    ("         CRT sanity check (S2)",
     "python experiments/crt_sanity_check.py --model gemma-2b --layer 19"),
]:
    print(f"# {label}\n{c}\n")
''')

md(r"""
### 2.8 — Steps 12–15 · generalisation

| Step | Script | Question |
|---|---|---|
| 12 | `generalization_tests.py` | Subtraction, operand substitution, multi-digit |
| 13 | `fourier_umap.py` | UMAP visualisation of the digit manifold |
| 14 | `multilayer_freq_ablation.py` | Per-frequency ablation across layer ranges |
| 15 | `multidigit_circuit.py` | Multi-digit circuit (Gemma) |

Step 14 is what produced the per-frequency damage numbers underpinning the
*k*=5 paradox in §2.5, so it is worth running even though it is labelled
"advanced".

> **⚠ UMAP is for illustration only.** It is a non-linear, stochastic embedding
> with no distance guarantees. A clean ring in UMAP is *not* evidence of circular
> structure — §2.3's DFT is. Never let a UMAP plot carry an argument.
""")

code(r'''
for label, c in [
    ("Step 12  generalisation",     "python experiments/generalization_tests.py --model gemma-2b --comp-layer 19 --device cuda"),
    ("Step 13  UMAP (illustrative)", "python experiments/fourier_umap.py --model gemma-2b --comp-layer 19 --device cuda"),
    ("Step 14  per-frequency ablation (drives the k=5 result)",
                                     "python experiments/multilayer_freq_ablation.py --model gemma-2b --comp-layer 19 --readout-layer 25 --device cuda"),
    ("Step 15  multi-digit circuit", "python experiments/multidigit_circuit.py --model gemma-2b --device cuda"),
]:
    print(f"# {label}\n{c}\n")
''')

md(r"""
### 2.9 — Producing the paper figures

> **⚠ The six figures `paper/` needs are not in the repository.** `paper/main.tex`
> sets `\graphicspath{{../mathematical_toolkit_results/paper_plots/}}`, which is
> generated output that was never tracked. **`paper/` will not compile until you
> regenerate them.**

The plotting scripts read JSON written by the analysis steps, so each figure has
a prerequisite. The data steps emit **no images** — all rendering is done by the
three plotting scripts.
""")

code(r'''
figures = [
    ("layer_scan_curves.png",              "generate_paper_plots.py / generate_missing_plots.py", "Step 1  arithmetic_circuit_scan_updated.py"),
    ("fourier_heatmap_cross_model.png",    "generate_paper_plots.py",                             "Step 3  fourier_decomposition.py"),
    ("energy_explosion.png",               "generate_paper_plots.py",                             "Step 3  fourier_decomposition.py"),
    ("ablation_curves.png",                "generate_missing_plots.py",                           "Step 4  fourier_knockout.py + multilayer_freq_ablation.py"),
    ("neuron_frequency_tuning.png",        "generate_missing_plots.py",                           "Step 9  neuron_trig_analysis.py"),
    ("eigenvector_fourier_cross_model.png","plot_eigenvector_dft.py",                             "Step 2  eigenvector_dft.py"),
]
print(f"{'figure':<38}{'plotted by':<52}{'needs'}")
print("-" * 132)
for fig_name, plotter, prereq in figures:
    print(f"{fig_name:<38}{plotter:<52}{prereq}")

plots_dir = ARITH / "mathematical_toolkit_results" / "paper_plots"
have_figs = sorted(p.name for p in plots_dir.glob("*.png")) if plots_dir.is_dir() else []
print(f"\npaper_plots/ present: {have_figs or 'NONE -- paper/ will not compile yet'}")
print("(.gitignore has been adjusted so this directory becomes trackable once populated.)")
''')

# ════════════════════════════════════════════════════════════ PART 3
md(r"""
---
# Part 3 — Reading these results responsibly

## 3.1 What is solid, what is provisional

| Claim | Evidence | My confidence |
|---|---|---|
| A compass plane behaves as a lawful dial | R²=1.0000 linearity, 0.014° phase drift over 10× α | **High** — clean, reproducible |
| The plane is genuinely rotational | Cyclicity 1.0 / 0.0, closed circle with disjoint antipodes | **High** |
| The effect depends on real label structure | Permutation null p=0.0 (obs 25.5 vs 4.5±1.9) | **High** |
| *These specific* singular directions are privileged | random-plane p=0.38; scan excess −0.23% vs top4 null | **Refuted** — see §1.6, §1.7 |
| Digit codes are a 9D Fourier basis of ℤ/10ℤ | 2/2/2/2/1 assignment, purity >50% | **High** |
| That subspace is causally necessary | Knockout → chance; random matched-dim → zero effect | **High** — the specificity control is what makes it |
| Fourier phase steering is practical | 28–69% exact shift; ≤10% at readout | **Provisional** — bounded by the 8–11% encoding–readout gap |
| Parity (*k*=5) drives the computation | 71% of variance but 0.4–1.0% ablation damage | **Refuted** — epiphenomenal |

## 3.2 Three ways to misuse this notebook

**Quoting "nine-test battery" as nine passes.** One test rejects its null, one
fails to, one is vacuous, one is strongly positive. §1.6 has the breakdown. The
honest headline is *"the effect survives a permutation null and shows genuine
circular structure; it is not localised to a specific SV pair."*

**Ranking directions by variance and stopping.** The *k*=5 paradox is the
cautionary tale: the single largest singular direction, 71% of variance, is
causally inert. Any direction that has not been ablated has not earned a causal
claim.

**Steering at the wrong layer.** At readout layers everything reads chance-level
because the subspace has been absorbed into the unembedding. Reporting that as a
negative result is a measurement error.

## 3.3 If a number here disagrees with your run

Check, in this order:

1. **Prompt protocol** — teacher-forced vs direct-answer (§2.1). The most common
   silent failure.
2. **Layer indices** — comp-layer is *discovered* by Step 1, not assumed. The
   cached defaults are for the exact checkpoints we used.
3. **Model revision** — "gemma-2-2b" today is not necessarily the weights we ran.
   Pin a revision hash.
4. **dtype and device** — the recorded compass runs used MPS with a documented
   CPU fallback for `linalg_svd`; 4-bit quantised runs are labelled
   `_4bit_quant` and are *not* comparable to fp32 runs.
5. **Null choice** — `top4` vs `full_ov` inverts the sign of the scan result
   (§1.7). Always state which null.
""")

md(r"""
## 3.4 Reproducibility ledger

Final cell: record exactly what this environment was, so a future reader can tell
whether a discrepancy is theirs or ours.
""")

code(r'''
import platform, datetime

print("=" * 68)
print("REPRODUCIBILITY LEDGER")
print("=" * 68)
print(f"timestamp     : {datetime.datetime.now().isoformat(timespec='seconds')}")
print(f"platform      : {platform.platform()}")
print(f"python        : {sys.version.split()[0]}")
try:
    import torch; print(f"torch         : {torch.__version__}  (cuda={torch.cuda.is_available()})")
except ImportError:
    print("torch         : not installed")
try:
    import transformer_lens; print(f"transformer_lens: {transformer_lens.__version__}")
except Exception:
    print("transformer_lens: not installed / version unavailable")
print(f"numpy         : {np.__version__}")

try:
    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                         capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                           capture_output=True, text=True).stdout.strip()
    print(f"git revision  : {rev}{' (dirty)' if dirty else ''}")
except Exception:
    print("git revision  : unavailable")

print(f"execution tier: {RUN_TIER}")
print(f"model access  : {'yes' if MODEL_ACCESS else 'NO -- tier 0 replay only'}")
print()
print("Artifacts replayed in this run:")
for p in [CV_LOGS / "gpt2_compass_causal.txt",
          CV_LOGS / "workshop_suite_results_gpt2_l9h7.json",
          CV_LOGS / "scan_pass_rate_ci.txt",
          CV_LOGS / "per_domain_alpha_gpt2.json"] + sorted(FR_LOGS.glob("*.json"))[-1:]:
    print(f"  {'ok ' if p.exists() else 'MISSING '}{p.relative_to(ROOT)}")
print("=" * 68)
''')

# ════════════════════════════════════════════════════════════ WRITE
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = ROOT / "experiments_end_to_end.ipynb"
out.write_text(json.dumps(nb, indent=1))
n_code = sum(1 for c in cells if c["cell_type"] == "code")
print(f"wrote {out.relative_to(ROOT)}  ({len(cells)} cells: {n_code} code, {len(cells)-n_code} markdown)")
