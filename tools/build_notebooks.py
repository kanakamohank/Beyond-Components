#!/usr/bin/env python3
"""Generate one self-contained experiment notebook per paper.

    semantic-compass/notebooks/semantic_compass_experiments.ipynb
    arithmetic-circuit-discovery/notebooks/arithmetic_circuit_experiments.ipynb

Each notebook resolves its OWN repository root by looking for a marker unique to
that repo, so it works both inside this combined tree and after the repo is
pushed out standalone.

Generated rather than hand-edited so it can be regenerated and re-validated as
the code it documents moves. Run order matters: materialize.py wipes the target
directories, so this must run after it (postprocess.py invokes it).
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
COMPASS = ROOT / "semantic-compass"
ARITH = ROOT / "arithmetic-circuit-discovery"


class Notebook:
    def __init__(self, path: Path):
        self.path = path
        self.cells = []

    def md(self, text):
        self.cells.append({"cell_type": "markdown", "metadata": {},
                           "source": text.strip("\n").splitlines(keepends=True)})

    def code(self, text):
        self.cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                           "outputs": [], "source": text.strip("\n").splitlines(keepends=True)})

    def write(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "cells": self.cells,
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python", "version": "3.11"},
            },
            "nbformat": 4, "nbformat_minor": 5,
        }, indent=1))
        n_code = sum(1 for c in self.cells if c["cell_type"] == "code")
        print(f"  {self.path.relative_to(ROOT)}  "
              f"({len(self.cells)} cells: {n_code} code, {len(self.cells) - n_code} md)")


# ─────────────────────────────────────────────── shared cell builders

def tier_table(what):
    return f"""
### Execution tiers — read before running anything

**This notebook cannot download models.** {what} In a sandboxed or air-gapped
environment that fetch fails, so the notebook is built in three tiers and only
Tier 0 is guaranteed to run anywhere:

| Tier | Needs | Runtime | What it gives you |
|---|---|---|---|
| **0 — Verify & Replay** | nothing but this repo | seconds | Math checks against synthetic oracles; **re-analysis of the recorded experimental artifacts committed here**. Real numbers from real runs. |
| **1 — Reproduce (small)** | HuggingFace access, CPU ok | minutes–hours | Recompute the GPT-2 results from scratch. |
| **2 — Reproduce (full)** | GPU + gated model access | days | The full cross-model grid. |

**Tier 0 is not a mock.** It parses the actual logs committed in this
repository and recomputes the statistics from them. If a Tier-0 cell prints a
number, that number came out of a real run.

Tier 1/2 cells are **inert by default**: they print the command they would run
and stop. Set `RUN_TIER` below to arm them, after the preflight cell confirms
the environment can support it.
"""


def config_cell(marker_file, marker_dir, artifacts_var, artifacts_dir, repo_name):
    return f'''
import json, os, re, sys, textwrap, subprocess
from pathlib import Path

# ─────────────────────────────────────────────────────────────── settings
RUN_TIER = 0        # 0 = verify + replay | 1 = also recompute small model | 2 = full grid
DEVICE   = "auto"   # "auto" | "cpu" | "cuda" | "mps"

# ────────────────────────────────────────────────── repository resolution
# Looks for markers unique to this repo, so the notebook works whether it sits
# inside the combined tree or in the standalone {repo_name} repository.
def find_root(start: Path) -> Path:
    for d in [start, *start.parents]:
        if (d / "{marker_file}").is_file() and (d / "{marker_dir}").is_dir():
            return d
    raise SystemExit(
        "Could not locate the repository root.\\n"
        "Expected an ancestor directory containing '{marker_file}' and '{marker_dir}/'."
    )

REPO = find_root(Path.cwd().resolve())
{artifacts_var} = REPO / "{artifacts_dir}"     # recorded artifacts; Tier 0 reads these

print(f"repository   : {{REPO}}")
print(f"python files : {{sum(1 for _ in REPO.rglob('*.py'))}}")
print(f"artifacts    : {artifacts_var} -> {{len(list({artifacts_var}.glob('*'))) if {artifacts_var}.is_dir() else 0}} files")
print(f"tier         : {{RUN_TIER}}  "
      f"({{'verify + replay recorded artifacts' if RUN_TIER == 0 else 'live recomputation ARMED'}})")
'''


PREFLIGHT = '''
import importlib.util

def have(mod: str) -> bool:
    return importlib.util.find_spec(mod) is not None

print(f"python            : {sys.version.split()[0]}")
deps = ["torch", "numpy", "transformer_lens", "matplotlib", "pandas", "sklearn", "seaborn"]
missing = [d for d in deps if not have(d)]
for d in deps:
    print(f"  {d:<16}: {'yes' if have(d) else 'MISSING'}")

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

print()
if missing:
    print(f"! missing packages: {', '.join(missing)}   ->  pip install {' '.join(missing)}")
if RUN_TIER > 0 and not MODEL_ACCESS:
    print("! RUN_TIER > 0 but HuggingFace is unreachable. Live cells WILL fail.")
    print("  Set RUN_TIER = 0 for verification and replay only.")
elif RUN_TIER == 0:
    print("Tier 0: replaying recorded artifacts. No model downloads required.")

def tier(n: int) -> bool:
    """Guard for live cells. True only if this environment can actually run them."""
    if RUN_TIER < n:
        print(f"[tier {n} cell -- inactive, RUN_TIER={RUN_TIER}. Command shown above, not executed.]")
        return False
    if not MODEL_ACCESS:
        print(f"[tier {n} cell -- SKIPPED, no model access.]")
        return False
    return True

def shell(cmd: str):
    print(textwrap.dedent(cmd).strip())
'''


LEDGER = '''
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
    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO,
                         capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=REPO,
                           capture_output=True, text=True).stdout.strip()
    print(f"git revision  : {rev}{' (dirty)' if dirty else ''}")
except Exception:
    print("git revision  : unavailable")
print(f"execution tier: {RUN_TIER}")
print(f"model access  : {'yes' if MODEL_ACCESS else 'NO -- tier 0 replay only'}")
print()
print("Artifacts replayed in this run:")
for p in REPLAYED:
    print(f"  {'ok ' if p.exists() else 'MISSING '}{p.relative_to(REPO)}")
print("=" * 68)
'''


# ══════════════════════════════════════════════════════════════════════════
#                          COMPASS NOTEBOOK
# ══════════════════════════════════════════════════════════════════════════
nb = Notebook(COMPASS / "notebooks" / "semantic_compass_experiments.ipynb")

nb.md(r"""
# Semantic Compasses — experiment compendium

Executable index to every experiment behind:

> **Semantic Compasses: Rank-2 Causal Dials in Attention-Head OV Singular Planes**

For each experiment: **what it does**, **why we ran it**, **the exact command**,
and **what the result actually looked like** — including the results that came
out null or negative.

> The companion notebook for the arithmetic-circuit work lives in the separate
> `arithmetic-circuit-discovery` repository. The two share no code.

---

## The claim in one paragraph

Take any attention head, form its OV map $W_{OV} = W_V W_O$, take the SVD, and
pick **two** singular directions. That 2-D plane is often a **semantic compass**:
adding a rotating vector inside it sweeps the model's output logits along a
smooth, single-frequency sinusoid. The angle $\theta$ selects *which way* the
semantics point (he ↔ she, past ↔ future); the scale $\alpha$ selects *how hard*
you push. A continuous, causally effective dial — not a binary switch, and not a
correlational probe.
""")

nb.md(tier_table(
    "Every headline experiment needs pretrained weights (GPT-2, Phi-3, Gemma-2B, "
    "LLaMA-3.2-3B) from HuggingFace."
) + r"""

### Honesty note

I am reporting our own results, so the expected-outcome boxes state what we
actually observed. **Two results here are null or negative** (§5, §6). They are
documented as prominently as the positive ones. If you reproduce this work and
your compass scan does not beat its null, **that is the expected result** — §6.
""")

nb.md("## 0 · Configuration")
nb.code(config_cell("COMPASS_COOKBOOK.md", "helix_usage_validated",
                    "LOGS", "helix_usage_validated", "semantic-compass"))

nb.md(r"""
## 0.1 · Preflight

Reports what this environment can do, so you learn about a missing dependency
here rather than forty minutes into a sweep. The model-access probe is the one
that matters: **if it fails, every Tier 1/2 cell will fail.**
""")
nb.code(PREFLIGHT)

nb.md(r"""
## 1 · The five formulas

Everything below is built from these. `§` references are to
[`COMPASS_COOKBOOK.md`](../COMPASS_COOKBOOK.md).

**1. The plane** (§2.1)
$$W_{OV} = W_V W_O, \qquad U\Sigma V^\top = \mathrm{svd}(W_{OV}), \qquad \mathrm{plane}(L,H,i,j) = \mathrm{span}(u_i, u_j)$$

**2. The injection** (§2.2)
$$v(\theta,\alpha) = \alpha\,\sigma_i\cos\theta\;u_i \;+\; \alpha\,\sigma_j\sin\theta\;u_j$$

**3. The site** (§2.3) — hook `blocks.{L}.hook_resid_pre`, **last token only**:
`act[0, -1, :] += v(theta, alpha)`. The *only* intervention point in the pipeline.

**4. The metric** (§2.4) — for antipodal probe tokens $(t_+, t_-)$:
$$\mathrm{LD}(\theta,\alpha) = \mathrm{logit}(t_+) - \mathrm{logit}(t_-)$$
averaged over 3 prompts × 3 conditions (neutral / plus-context / minus-context).

**5. The fit** (§2.5) — first DFT bin: $\mathrm{LD}(\theta) \approx \mu + A\cos(\theta - \varphi)$

**Pass criterion** (§2.6) — a plane is a compass iff all three hold:

| Pillar | Threshold |
|---|---|
| Amplitude linearity | $R^2(A \text{ vs } \alpha$, through origin$) \ge 0.95$ |
| Phase stability | $\lvert\varphi(\alpha_{hi}) - \varphi(\alpha_{lo})\rvert \le 10°$ |
| Effect size | $A(\alpha_{hi})/\alpha_{hi} \ge 0.20$ (0.08 for Gemma) |

### 1.1 — The estimator, verified against a synthetic oracle

**Purpose.** Before trusting any result, confirm the estimator is correct. Build
a signal with *known* amplitude and phase; check the DFT fit recovers them.

**Why it matters.** The whole paper rests on reading $A$ and $\varphi$ off a
24-point angular sweep. A biased estimator biases every downstream number.

**Expected outcome.** Recovery to ~1e-12. A pure cosine must give $\varphi = 0$,
and the fit must be blind to a DC offset.

*Tier 0 · no model · instant*
""")

nb.code(r'''
import numpy as np

def fit_sinusoid(theta_deg, y):
    """First-DFT-bin fit of  y ~ mu + A*cos(theta - phi).

    Mirrors compass_scan.py:54-62, the most-cited form in the paper.
    Returns (mu, A, phi_degrees).
    """
    th  = np.deg2rad(np.asarray(theta_deg, dtype=float))
    y   = np.asarray(y, dtype=float)
    N   = len(y)
    mu  = y.mean()
    c   = ((y - mu) * np.cos(th)).sum() * 2.0 / N
    s   = ((y - mu) * np.sin(th)).sum() * 2.0 / N
    return mu, float(np.hypot(c, s)), float(np.degrees(np.arctan2(s, c)))

grid = np.arange(0, 360, 15.0)
for mu_t, A_t, phi_t in [(0.0, 1.0, 0.0), (0.55, 1.19, -175.4), (-2.0, 5.0, 90.0)]:
    y = mu_t + A_t * np.cos(np.deg2rad(grid - phi_t))
    mu_h, A_h, phi_h = fit_sinusoid(grid, y)
    dphi = (phi_h - phi_t + 180) % 360 - 180
    ok = abs(mu_h - mu_t) < 1e-9 and abs(A_h - A_t) < 1e-9 and abs(dphi) < 1e-6
    print(f"mu={mu_t:+.2f} A={A_t:.2f} phi={phi_t:+.1f}  ->  "
          f"recovered mu={mu_h:+.6f} A={A_h:.6f} phi={phi_h:+.4f}   {'OK' if ok else 'FAIL'}")

y = 100.0 + 2.0 * np.cos(np.deg2rad(grid - 30.0))
_, A_h, phi_h = fit_sinusoid(grid, y)
print(f"\nDC-offset invariance: A={A_h:.6f} (expect 2.0), phi={phi_h:+.4f} (expect +30.0)")
assert abs(A_h - 2.0) < 1e-9 and abs(phi_h - 30.0) < 1e-6
print("\nEstimator verified.")
''')

nb.md(r"""
### 1.2 — Injection geometry, verified

**Purpose.** Confirm $v(\theta,\alpha)$ behaves as claimed: it stays in the
plane, its norm scales linearly in $\alpha$, and $\theta \to \theta + 180°$
negates it — which is *why* antipodal probe tokens work, and what the cyclicity
test in §5 checks empirically.

**Expected outcome.** All three exact to machine precision.

*Tier 0 · no model · instant*
""")

nb.code(r'''
rng = np.random.default_rng(0)

# Two orthonormal directions standing in for (u_i, u_j) of a real head, with the
# singular values we measured at GPT-2 L9H7 (SV1, SV2).
d_model = 768
Q, _ = np.linalg.qr(rng.standard_normal((d_model, 2)))
u_i, u_j = Q[:, 0], Q[:, 1]
sigma_i, sigma_j = 8.866, 8.456          # gpt2_compass_causal.txt

def inject(theta_deg, alpha):
    th = np.deg2rad(theta_deg)
    return alpha * sigma_i * np.cos(th) * u_i + alpha * sigma_j * np.sin(th) * u_j

P_perp = np.eye(d_model) - Q @ Q.T
resid = max(np.linalg.norm(P_perp @ inject(t, 3.0)) for t in range(0, 360, 15))
print(f"max out-of-plane component : {resid:.3e}   (expect ~0)")

n1, n10 = np.linalg.norm(inject(37.0, 1.0)), np.linalg.norm(inject(37.0, 10.0))
print(f"||v(a=10)|| / ||v(a=1)||   : {n10/n1:.10f}   (expect 10.0)")

anti = max(np.linalg.norm(inject(t, 1.0) + inject(t + 180, 1.0)) for t in range(0, 360, 15))
print(f"max ||v(t) + v(t+180)||    : {anti:.3e}   (expect ~0)")

assert resid < 1e-10 and abs(n10 / n1 - 10) < 1e-9 and anti < 1e-10
print("\nInjection geometry verified.")
''')

nb.md(r"""
## 2 · `compass_causal_sweep.py` — the α-sweep (paper Figure 2)

| | |
|---|---|
| **Script** | `experiments/compass_causal_sweep.py` (287 lines) |
| **Stage** | L2 — validation on a single, known head |
| **Purpose** | Establish that one plane behaves as a *dial*: sinusoidal in $\theta$, linear in $\alpha$, phase-stable across scales. |
| **Method** | 36 angles × α ∈ {1, 3, 10} × 3 prompt conditions. Inject, read logit difference, fit. |
| **Outputs** | `helix_usage_validated/<prefix>_{curves,polar,linearity}.png` + `.txt` |
| **Runtime** | ~2 min (GPT-2, CPU) · ~15 min (Phi-3, GPU) |

**What this is and is not.** The *confirmatory* experiment, run on a head we
already believe in (GPT-2 L9H7). It is **not** evidence that compasses are
common — §6 is the honest test of that. It establishes that *when* a compass
exists, it behaves lawfully.

**Expected outcome — our GPT-2 L9H7 run.** Baseline logit(` he`) − logit(` she`)
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

nb.code(r'''
def parse_causal_sweep(path):
    """Parse a compass_causal_sweep log -> {alpha: array[[deg, neutral, masc, fem]]}."""
    txt = Path(path).read_text(errors="replace")
    parts = re.split(r"INJECTION SWEEP at alpha\s*=\s*([\d.]+)", txt)
    rows_re = re.compile(r"^\s*([\d.]+)\s*\|\s*([+-][\d.]+)\s+([+-][\d.]+)\s+([+-][\d.]+)\s*$", re.M)
    out = {}
    for i in range(1, len(parts) - 1, 2):
        rows = rows_re.findall(parts[i + 1])
        if rows:
            out[float(parts[i])] = np.array(rows, dtype=float)
    return out

log = LOGS / "gpt2_compass_causal.txt"
sweep = parse_causal_sweep(log)
print(f"source: {log.relative_to(REPO)}")
print(f"scales: {sorted(sweep)}   angles per scale: {len(next(iter(sweep.values())))}\n")

print(f"{'alpha':>6} {'mu':>9} {'A':>9} {'A/alpha':>9} {'phi(deg)':>10}")
print("-" * 46)
fit = {}
for a in sorted(sweep):
    mu, A, phi = fit_sinusoid(sweep[a][:, 0], sweep[a][:, 1])   # col 1 = neutral prompts
    fit[a] = (mu, A, phi)
    print(f"{a:>6.1f} {mu:>+9.3f} {A:>9.3f} {A/a:>9.3f} {phi:>+10.1f}")

alphas = np.array(sorted(fit))
amps   = np.array([fit[a][1] for a in alphas])
slope  = float((alphas @ amps) / (alphas @ alphas))       # least squares through origin
r2     = 1 - float(((amps - slope*alphas)**2).sum()) / float(((amps - amps.mean())**2).sum())
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

nb.code(r'''
# The same log, drawn. Three scales sharing one phase is the visual signature of
# a dial: the curves are scaled copies of each other, not different shapes.
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
ax.plot(lin, slope*lin, "--", color="k", alpha=.5, label=f"through origin (R^2={r2:.4f})")
ax.set_xlabel("alpha"); ax.set_ylabel("amplitude A")
ax.set_title("Amplitude linearity"); ax.legend(fontsize=8)

plt.tight_layout(); plt.show()
''')

nb.code(r'''
cmd = """
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
shell(cmd)
if tier(1):
    subprocess.run(cmd, shell=True, cwd=REPO, check=True)
    print("\nRe-parse with parse_causal_sweep(LOGS / 'gpt2_rerun.txt') and compare above.")
''')

nb.md(r"""
## 3 · `compass_dictionary.py` — decoding what a plane *means*

| | |
|---|---|
| **Script** | `experiments/compass_dictionary.py` (159 lines) · single-model variant `compass_dictionary_single.py` |
| **Stage** | L1a — decode-driven discovery |
| **Purpose** | Turn an anonymous SVD axis into a *readable* semantic axis. |
| **Method** | Project the top-K singular directions through mean-centred $W_U$; read the highest-logit tokens at each pole. |
| **Outputs** | `helix_usage_validated/compass_dict_{gpt2,phi3,gemma,llama32_3b}.txt`, `compass_dict_all.md` |
| **Runtime** | ~1 min/model |

**Why mean-centring matters.** Without subtracting the mean unembedding, every
direction decodes to the same high-frequency tokens (`the`, `,`, `and`) and the
dictionary is useless.

**Expected outcome.** A minority of axes decode to crisp antipodal pairs;
**most decode to noise, and that is normal** — we read the top singular
directions of every head and most carry no single clean concept. This is a
*hypothesis generator*, not evidence. Nothing here is causal; §2 and §5 supply
the causality.
""")

nb.code(r'''
dicts = sorted(LOGS.glob("compass_dict_*.txt"))
print(f"decoded dictionaries on disk: {[p.stem.replace('compass_dict_', '') for p in dicts]}\n")

sample = LOGS / "compass_dict_gpt2.txt"
if sample.exists():
    lines = sample.read_text(errors="replace").splitlines()
    start = next((i for i, l in enumerate(lines) if re.match(r"\s*L\d+H\d+", l)), 0)
    print(f"--- {sample.name} (first decoded head) ---")
    print("\n".join(lines[start:start + 22]))
else:
    print("No dictionary found -- run experiments/compass_dictionary.py (Tier 1).")
''')

nb.code(r'''
cmd = "python experiments/compass_dictionary.py"
shell(cmd)
if tier(1):
    subprocess.run(cmd, shell=True, cwd=REPO, check=True)
''')

nb.md(r"""
## 4 · Cyclicity — is the plane really a *circle*?

| | |
|---|---|
| **Entry point** | `investigate_helix_usage_validated.py --test cyclicity` (method `cyclicity_check`) |
| **Stage** | L3 — falsification battery, test 2 of 9 |
| **Purpose** | Distinguish genuine **rotational** structure from a plane that merely correlates with a concept. |
| **Method** | Decode the vocabulary at θ = 0°, 180°, 360°; compare top-10 token sets by Jaccard overlap. |

**The prediction.** A true dial must decode *identically* at θ = 0° and θ = 360°
(the circle closes) and to the **opposite** pole at θ = 180° (disjoint set). A
correlational direction has no reason to satisfy either.

**Expected outcome — GPT-2 L9H7.** One of our cleanest results:
`jaccard_identity = 1.0`, `jaccard_antipode = 0.0`, verdict
**`CIRCLE: closed and with distinct antipodes`**.
""")

nb.code(r'''
res = json.loads((LOGS / "workshop_suite_results_gpt2_l9h7.json").read_text())
cyc = res["cyclicity"]

print(f"head L{cyc['layer']}H{cyc['head']}  dims {tuple(cyc['dims'])}\n")
print(f"  jaccard(theta=0, theta=360) = {cyc['jaccard_identity']:.2f}   (1.0 => circle closes)")
print(f"  jaccard(theta=0, theta=180) = {cyc['jaccard_antipode']:.2f}   (0.0 => poles disjoint)\n")
for label, key in [("theta =   0", "top10_0"), ("theta = 180", "top10_180"), ("theta = 360", "top10_360")]:
    print(f"  {label}: {', '.join(repr(t) for t in cyc[key][:6])} ...")
print(f"\n  verdict: {cyc['verdict']}")
assert cyc["jaccard_identity"] == 1.0 and cyc["jaccard_antipode"] == 0.0
''')

nb.md(r"""
## 5 · The nine-test falsification battery — **including what failed**

| | |
|---|---|
| **Entry point** | `investigate_helix_usage_validated.py --test all-must-have <model>` |
| **Driver** | `run_workshop_suite` (line 3361) |
| **Outputs** | `helix_usage_validated/workshop_suite_results_<model>_<head>.json` |
| **Runtime** | ~20 min (GPT-2) · ~2 h (Phi-3) |

The battery exists to *attack* the compass claim. Six tests run by default.

> ### ⚠ Two do not support the hypothesis. Read this before citing the battery.
>
> **The random-plane null does not reject (p = 0.38).** Observed amplitude at
> GPT-2 L9H7 is 25.49; random planes *within the same head* average 26.58
> (sd 21.77) — they do as well or better.
>
> **The principal-angles test returned empty.** `angles_deg = {}` — it needs ≥2
> head specs and was given one. **Vacuous, not passed.**
>
> **What does hold: the permutation null, decisively.** Observed 25.49 vs a
> permuted mean of 4.54 (sd 1.87, max 11.45) over 2000 permutations, p = 0.0.
>
> **Reading this honestly.** The permutation test shows the effect depends on
> *real label structure* — not an artefact of the sweep machinery. The
> random-plane result shows the effect is **not localised to the specific SVD
> pair we picked**. The defensible claim is *"this head carries a gender dial"*,
> **not** *"these two singular directions uniquely carry it."*
""")

nb.code(r'''
res = json.loads((LOGS / "workshop_suite_results_gpt2_l9h7.json").read_text())
rows = []

rp = res["random_plane"]
rows.append(("random-plane null",
             "obs {:.2f} vs {:.2f}+-{:.2f}, p={:.2f}".format(
                 rp["observed"], rp["random_mean"], rp["random_std"], rp["p_value"]),
             "DOES NOT REJECT" if rp["p_value"] > 0.05 else "rejects null"))

pm = res["permutation"]
rows.append(("permutation null",
             "obs {:.2f} vs {:.2f}+-{:.2f}, p={:.2f}".format(
                 pm["observed"], pm["perm_mean"], pm["perm_std"], pm["p_value"]),
             "REJECTS NULL" if pm["p_value"] < 0.05 else "does not reject"))

n_ang = len(res["principal_angles"].get("angles_deg", {}))
rows.append(("principal angles", "{} angle(s) computed".format(n_ang),
             "VACUOUS (needs >=2 heads)" if n_ang == 0 else "ok"))

cy = res["cyclicity"]
rows.append(("cyclicity",
             "identity {:.1f} / antipode {:.1f}".format(cy["jaccard_identity"], cy["jaccard_antipode"]),
             cy["verdict"].split(":")[0]))

for key in ("causal_patch", "semantic_ablation"):
    if key in res:
        rows.append((key.replace("_", " "), "see raw json", "present"))

print("{:<20}{:>38}   {}".format("test", "statistic", "verdict"))
print("-" * 80)
for name, stat, verdict in rows:
    print("{:<20}{:>38}   {}".format(name, stat, verdict))

print("\nSummary: 1 test rejects its null (permutation), 1 fails to (random-plane),")
print("         1 is vacuous (principal-angles), 1 is strongly positive (cyclicity).")
print("         Do not report this as 'nine tests passed'.")
''')

nb.code(r'''
cmd = "python investigate_helix_usage_validated.py --test all-must-have gpt2"
shell(cmd)
if tier(1):
    subprocess.run(cmd, shell=True, cwd=REPO, check=True)
''')

nb.md(r"""
## 6 · `compass_scan.py` — the blind scan, and the result that constrains the paper

| | |
|---|---|
| **Script** | `experiments/compass_scan.py` (345 lines) · CIs by `scan_ci_summary.py` |
| **Stage** | L1b — **blind** discovery over every (layer, head, SV-pair) |
| **Purpose** | The honest prevalence question: not knowing where to look, how often does a compass appear versus a matched null? |
| **Outputs** | `<prefix>_scan.txt`, `<prefix>_heatmap.png`, `scan_pass_rate_ci.txt` |
| **Runtime** | ~40 min (GPT-2) · ~6 h (Phi-3) |

**Two nulls, and they disagree.** `top4` permutes within the top-4 singular
directions; `full_ov` samples random planes from the whole OV spectrum. Which
null you choose changes the conclusion, so we report both.

> ### ⚠ Against the `top4` null the excess is *negative*.
>
> | scan | compass | null (top4) | excess | null (full_ov) | excess |
> |---|---:|---:|---:|---:|---:|
> | gpt2 gender | 2.06% (16/864) | 2.51% | **−0.23%** | 0.44% | +1.85% |
> | phi3 gender | 1.18% (14/1344) | 1.62% | **−0.30%** | 0.28% | +1.04% |
> | gemma gender | 2.02% (3/240) | 1.55% | +1.25% | 1.55% | +1.25% |
>
> Compass planes are **not** more common than other planes from the same top-4
> subspace; they *are* more common than planes from the whole OV spectrum. Both
> are true, and together they mean **the top singular subspace carries the
> structure while our specific $(i,j)$ choice is not privileged** — the same
> conclusion §5's random-plane null reaches independently.
>
> Note the absolute rates: ~1–2%, and 3/240 for Gemma. Small counts, wide CIs.

**If you reproduce this and get a negative excess against `top4`, you have
reproduced our result correctly.**
""")

nb.code(r'''
print((LOGS / "scan_pass_rate_ci.txt").read_text(errors="replace").strip())
''')

nb.code(r'''
# Recompute the Agresti-Coull intervals from raw counts, independently of the
# committed summary -- a check that the reported CIs are right.
def agresti_coull(x, n, z=1.959963985):
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

nb.code(r'''
cmd = """
python experiments/compass_scan.py \\
    --model gpt2 --tok_plus " he" --tok_minus " she" \\
    --top_svs 4 --alphas 3 10 --n_angles 12 \\
    --null_mode top4 --null_seeds 3 --out_prefix gpt2_rerun_scan
"""
shell(cmd)
if tier(1):
    subprocess.run(cmd, shell=True, cwd=REPO, check=True)
''')

nb.md(r"""
## 7 · Downstream evaluation — does the dial *do* anything?

| Script | Benchmark | Question |
|---|---|---|
| `crowspairs_routed_eval.py` | CrowS-Pairs | Does routed per-domain steering reduce stereotype preference? |
| `stereoset_ensemble_eval*.py` | StereoSet | Multi-plane ensemble injection, per domain |
| `winogender_eval.py` / `_sweep.py` | Winogender | Occupational coreference bias |
| `truthfulqa_eval.py` | TruthfulQA | **Off-target damage control** |
| `inlp_debias.py`, `sentence_debias.py` | — | INLP / SentenceDebias baselines |
| `baseline_comparison.py` | — | Compass vs four standard direction-finding methods |

A causal dial is only interesting if it moves a metric someone cares about — and
only *safe* if it leaves unrelated behaviour intact. TruthfulQA is the control,
not a headline.

**Calibration first.** `calibrate_per_domain_alpha.py` picks α per domain by
targeting a fixed SNR against the residual-stream norm, so "α = 2" means the
same perturbation strength across models.

> **⚠ The SNR target is frequently unreachable.** In `per_domain_alpha_gpt2.json`
> three of four domains carry `"fallback": true`: mean SNR at α=1 is ≈0.053
> against a target of 0.10, so α is scaled to ~1.87 and still misses. Several
> selected heads carry `"passed": false`. **Read every downstream number as "at
> the α we could reach", not "at the α we wanted".**
""")

nb.code(r'''
cfg = json.loads((LOGS / "per_domain_alpha_gpt2.json").read_text())
print(f"model: {cfg['model']}   target SNR: {cfg['target_snr']}")
print(f"norm probe prompt: {cfg['prompt_for_norm']!r}\n")

print(f"{'domain':<12}{'alpha':>8}{'mean SNR@1':>12}{'target hit':>15}{'heads':>8}{'passed':>8}")
print("-" * 65)
for dom, d in cfg["by_domain"].items():
    heads = d["heads"]
    n_pass = sum(h["passed"] for h in heads)
    hit = "no (fallback)" if d.get("fallback") else "yes"
    print(f"{dom:<12}{d['alpha']:>8.3f}{d['mean_snr_at_alpha1']:>12.4f}{hit:>15}"
          f"{len(heads):>8}{f'{n_pass}/{len(heads)}':>8}")

fallbacks = [d for d, v in cfg["by_domain"].items() if v.get("fallback")]
print(f"\nDomains that missed the SNR target: {fallbacks or 'none'}")
''')

nb.code(r'''
cmds = [
    ('calibrate alpha per domain',
     'python experiments/calibrate_per_domain_alpha.py --model gpt2 --target_snr 0.10'),
    ('routed CrowS-Pairs (paper Table 7)',
     'python experiments/crowspairs_routed_eval.py --model gpt2 '
     '--alpha_json helix_usage_validated/per_domain_alpha_gpt2.json'),
    ('StereoSet ensemble (GPT-2 hardcoded; _gemma/_phi3/_llama variants for others)',
     'python experiments/stereoset_ensemble_eval.py '
     '--domains gender,race,profession,religion --out_prefix stereoset_ensemble_gpt2'),
    ('Winogender sweep',        'python experiments/winogender_sweep.py --model gpt2'),
    ('TruthfulQA off-target',   'python experiments/truthfulqa_eval.py --model gpt2'),
    ('INLP baseline',           'python experiments/inlp_debias.py --model gpt2 --domain gender --layer 10'),
    ('SentenceDebias baseline', 'python experiments/sentence_debias.py --model gpt2 --domain gender --layer 10'),
    ('compass vs 4 baselines',  'python experiments/baseline_comparison.py --model gpt2'),
]
for label, c in cmds:
    print(f"# {label}\n{c}\n")
if tier(1):
    for label, c in cmds:
        print(f"=== {label} ===")
        subprocess.run(c, shell=True, cwd=REPO, check=True)
''')

nb.md(r"""
## 8 · Reading these results responsibly

### 8.1 What is solid, what is refuted

| Claim | Evidence | Confidence |
|---|---|---|
| A compass plane behaves as a lawful dial | R²=1.0000 linearity, 0.014° phase drift over 10× α | **High** |
| The plane is genuinely rotational | Cyclicity 1.0 / 0.0, closed circle, disjoint antipodes | **High** |
| The effect depends on real label structure | Permutation null p=0.0 (25.5 vs 4.5±1.9) | **High** |
| *These specific* singular directions are privileged | random-plane p=0.38; scan excess −0.23% vs top4 | **Refuted** (§5, §6) |
| Steering moves downstream bias metrics | CrowS-Pairs / StereoSet / Winogender deltas | **Provisional** — α missed target in 3/4 domains |

### 8.2 Three ways to misuse this notebook

**Quoting "nine-test battery" as nine passes.** One test rejects its null, one
fails to, one is vacuous, one is strongly positive. The honest headline: *the
effect survives a permutation null and shows genuine circular structure; it is
not localised to a specific SV pair.*

**Reporting downstream deltas without the α caveat.** Three of four domains never
reached the target SNR.

**Citing the scan without naming the null.** `top4` and `full_ov` give opposite
signs (§6). Always state which.

### 8.3 If a number here disagrees with your run

1. **Null choice** — `top4` vs `full_ov` inverts the scan result.
2. **Model revision** — "gpt2" today may not be the weights we ran. Pin a revision hash.
3. **dtype / device** — recorded runs used MPS with a documented CPU fallback for
   `linalg_svd`; runs labelled `_4bit_quant` are **not** comparable to fp32.
4. **Head and dims** — L9H7 appears with both (SV1,SV2) and (d1=1,d2=3) in our
   logs; they are different planes with different σ.

## 8.4 Reproducibility ledger
""")

nb.code('''
REPLAYED = [LOGS / "gpt2_compass_causal.txt",
            LOGS / "workshop_suite_results_gpt2_l9h7.json",
            LOGS / "scan_pass_rate_ci.txt",
            LOGS / "per_domain_alpha_gpt2.json"]
''' + LEDGER)

nb.write()


# ══════════════════════════════════════════════════════════════════════════
#                        ARITHMETIC NOTEBOOK
# ══════════════════════════════════════════════════════════════════════════
nb = Notebook(ARITH / "notebooks" / "arithmetic_circuit_experiments.ipynb")

nb.md(r"""
# The Fourier Basis of Digit Arithmetic — experiment compendium

Executable index to every experiment behind:

> **The Fourier Basis of Digit Arithmetic: Mechanistic Interpretability of
> Addition Circuits in Language Models**

For each experiment: **what it does**, **why we ran it**, **the exact command**,
and **what the result actually looked like** — including the hypothesis we
falsified.

> The companion notebook for the semantic-compass work lives in the separate
> `semantic-compass` repository. The two share no code.

---

## The claim in one paragraph

Language models represent decimal digits in a **Fourier basis of
$\mathbb{Z}/10\mathbb{Z}$**. At the computation layer the digit subspace
decomposes into exactly **nine** directions — two each for frequencies
$k = 1,2,3,4$ and one for $k = 5$ (parity). That subspace is causally necessary:
zero it and addition collapses to chance, while a matched-dimension *random*
subspace ablation does nothing at all. Rotate its phase and the model's answer
shifts by a predictable amount mod 10.
""")

nb.md(tier_table(
    "Gemma-2B, Phi-3-mini and LLaMA-3.2-3B are gated on HuggingFace and need a GPU."
) + r"""

### Two structural cautions

**Everything here is Tier 2.** Recorded runtime is 30–60 min for Step 1 alone
and 1–2 h for the Step 3 sweep, *per model*. Budget days, not hours. The Tier 0
cells verify the mathematics and replay a committed Pythia-1.4B run.

**`ARITHMETIC_CIRCUIT_PLAN.md` also documents a superseded route** — "What
Already Exists", "Phase 0", and `SUPPLEMENTARY SCRIPTS` S5 ("Old Pipeline") —
built on a mask-learning dependency that has since been removed. **Commands in
those sections will not run.** They are retained as a record of how the work
developed. Phases A–F below are the paper's actual method.
""")

nb.md("## 0 · Configuration")
nb.code(config_cell("ARITHMETIC_CIRCUIT_PLAN.md", "experiments",
                    "FR", "fourier_results", "arithmetic-circuit-discovery"))

nb.md(r"""
## 0.1 · Preflight

The model-access probe is the one that matters: **if it fails, every Tier 2 cell
will fail**, which here is the entire live pipeline.
""")
nb.code(PREFLIGHT)

nb.md(r"""
## 1 · Pipeline shape

**15 numbered steps in six phases**, strictly ordered — Step 1 determines the
`comp-layer` every later step needs.

| Phase | Steps | Question |
|---|---|---|
| **A — Discovery** | 1–3 | Where does arithmetic live, and is the code Fourier? |
| **B — Causal validation** | 4–7 | Is that subspace *necessary* and *sufficient*? |
| **C — Attribution** | 8–9 | Which heads and neurons write it? |
| **D — Mechanism** | 10–11 | *How* is addition performed? |
| **E — Generalisation** | 12–14 | Does it survive subtraction, multi-digit, new ranges? |
| **F — Multi-digit** | 15 | Gemma-specific extension |

### 1.1 — Step 0 · teacher-forced vs direct-answer

**Purpose.** Choose the prompt protocol *before* anything else: it changes the
unembedding basis and therefore every later measurement.

- **Teacher-forced** (default): `"Calculate 13 + 8 = 2"`, model predicts `1`.
  Valid when single-digit answer tokens 0–9 exist.
- **Direct-answer**: `"a + b = "`, model emits the whole answer as one token.
  **Required** when the tokenizer has single tokens for 0–198 — LLaMA-3.2-3B
  does. Pass `--direct-answer` to *every* later command.

**How to decide.** Run a handful of `"a + b = "` prompts; above ~90% accuracy in
that format, use direct-answer.

> **⚠ Getting this wrong is silent.** The pipeline runs and produces
> plausible-looking numbers. It surfaces much later as near-zero unembed-patching
> transfer, which reads like a scientific null rather than a configuration error.
> `diagnose_unembed_direct.py` exists because we hit exactly this.
""")

nb.code(r'''
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
print("comp-layer is DISCOVERED by Step 1; those defaults only cache what we found.")

reg = REPO / "src" / "utils" / "model_registry.py"
print(f"\nshipped registry: {reg.relative_to(REPO)}  ({'present' if reg.exists() else 'MISSING'})")
''')

nb.md(r"""
## 2 · Step 1 — layer scan + unembed patching **(must run first)**

| | |
|---|---|
| **Script** | `experiments/arithmetic_circuit_scan_updated.py` |
| **Purpose** | Find the **comp-layer** and **readout-layer**. Every later step depends on this. |
| **Key functions** | `run_layer_scan`, `compute_unembed_basis[_direct_answer]`, `run_patching_experiment`, `compute_fisher_matrix`, `compute_contrastive_fisher`, `filter_correct_*` |
| **Output** | `mathematical_toolkit_results/arithmetic_scan_<model>.json` |
| **Runtime** | 30–60 min per model |

**Method.** Sweep every layer; patch activations between digit-pairs; measure
transfer. Then patch only a *subspace* (unembed-aligned / Fisher / random) to
measure effective dimensionality.

**What to look for.** First layer above 80% transfer → **comp-layer**; last layer
at ~100% → **readout-layer**; 9D unembed patching should capture most transfer at
readout.

**Expected outcome.** comp-layer ≈ **L19** (Gemma-2B), **L26** (Phi-3),
**L20** (LLaMA-3.2-3B); readout ≈ L25 / L31 / L27.

> **⚠ A dissociation you will hit mid-stack.** At Gemma L21, Fisher patching
> transfers **85%** but unembed patching only **30%**. The digit code is
> *gradient-visible but not yet output-aligned*; Fisher alignment and unembedding
> rotation are separate processes converging only at output layers. **A low
> unembed number mid-stack is a real finding, not a bug.**
""")

nb.code(r'''
cmd = """
python experiments/arithmetic_circuit_scan_updated.py \\
    --model gemma-2b --device cuda --n-per-digit 100 --n-test 150
# LLaMA-3.2-3B additionally needs:  --direct-answer
"""
shell(cmd)
print("\nExpected comp-layer / readout-layer, from our runs:")
for m, comp, ro in [("gemma-2b", 19, 25), ("phi-3", 26, 31), ("llama-3b", 20, 27)]:
    print(f"    {m:<12} comp-layer L{comp:<3} readout-layer L{ro}")
if tier(2):
    subprocess.run(cmd, shell=True, cwd=REPO, check=True)
''')

nb.md(r"""
## 3 · Step 2 — eigenvector DFT · **is it really a Fourier basis?**

| | |
|---|---|
| **Script** | `experiments/eigenvector_dft.py` → plots by `plot_eigenvector_dft.py` |
| **Purpose** | Test whether the digit encoding is a *perfect* Fourier basis of $\mathbb{Z}/10\mathbb{Z}$. |
| **Method** | Take each SVD direction's 10-element digit-score vector, DFT it, ask which frequency dominates. |
| **Output** | `mathematical_toolkit_results/eigenvector_dft_<model>.json` |
| **Runtime** | 15–30 min |

**The prediction.** A perfect basis assigns exactly **2 directions to each of
k=1,2,3,4** and **1 to k=5** — $\cos$ and $\sin$ pair for every frequency except
the Nyquist frequency $k=5$, which is real-valued. Total: **9**.

**Expected outcome.** `"PERFECT FOURIER BASIS"` with mean purity > 50%.

> **⚠ This is the gate.** If the assignment is not 2/2/2/2/1, the model may not
> use a Fourier encoding at all and the rest of the pipeline is not meaningful.
> Before concluding that, check digit balance in your sample and raise `n`.

The cell below derives the number 9 from first principles and verifies the basis
is orthogonal and complete — no model required.
""")

nb.code(r'''
import numpy as np

N = 10
print("Fourier basis of Z/10Z -- degrees of freedom per frequency\n")
print(f"{'k':>3}  {'basis functions':<24}{'dims':>6}   note")
print("-" * 66)
total = 0
for k in range(0, N // 2 + 1):
    if k == 0:
        dims, fns, note = 0, "constant", "DC -- removed by centring"
    elif k == N // 2:
        dims, fns, note = 1, "cos(2*pi*5*d/10)", "Nyquist: sin() vanishes"
    else:
        dims, fns, note = 2, f"cos, sin(2*pi*{k}*d/10)", "conjugate pair"
    total += dims
    print(f"{k:>3}  {fns:<24}{dims:>6}   {note}")
print("-" * 66)
print(f"{'':>3}  {'TOTAL':<24}{total:>6}   <- the '9D Fourier subspace'")
assert total == 9

d = np.arange(N)
B = np.array([np.cos(2*np.pi*k*d/N) for k in range(1, 5)]
             + [np.sin(2*np.pi*k*d/N) for k in range(1, 5)]
             + [np.cos(np.pi*d)])
G = B @ B.T
print(f"\nGram off-diagonal max        : {np.abs(G - np.diag(np.diag(G))).max():.2e}  (orthogonal)")
print(f"rank of the 9 basis functions: {np.linalg.matrix_rank(B)}  (expect 9)")
print(f"rank of [basis ; constant]   : {np.linalg.matrix_rank(np.vstack([B, np.ones(N)]))}  (expect 10 = complete)")
''')

nb.code(r'''
cmd = "python experiments/eigenvector_dft.py --model gemma-2b --comp-layer 19 --device cuda"
shell(cmd)
print('\nLook for: "PERFECT FOURIER BASIS", mean purity > 50%,')
print("          frequency assignment 2/2/2/2/1 for k=1,2,3,4,5.")
if tier(2):
    subprocess.run(cmd, shell=True, cwd=REPO, check=True)
''')

nb.md(r"""
## 4 · Step 3 — Fourier layer sweep · where the structure is built

| | |
|---|---|
| **Script** | `experiments/fourier_decomposition.py` |
| **Purpose** | Track how Fourier energy accumulates layer by layer. |
| **Key functions** | `build_fourier_basis_functions`, `fourier_decomposition`, `per_neuron_fourier_analysis`, `run_fourier_at_layer` |
| **Runtime** | 1–2 h for a full sweep |

**Expected outcome.** Energy builds from early layers and **explodes by 2–3
orders of magnitude** at the computation layers — active amplification, not
passive propagation. This produces `energy_explosion.png`.

The related bottom-up head scan (`src/analysis/fourier_discovery.py`) produced
the committed Pythia-1.4B results replayed below — **real recorded output**.
""")

nb.code(r'''
runs = sorted(FR.glob("fourier_results_*.json"))
print(f"recorded runs: {[p.name for p in runs]}\n")

fr = json.loads(runs[-1].read_text())
cfg = fr["config"]
print(f"model             : {fr['model_key']}")
print(f"layers analysed   : {fr['n_layers_analyzed']}")
print(f"significant heads : {fr['n_significant_heads']}  "
      f"(power-ratio threshold {cfg['fourier']['head_power_ratio_threshold']})")
print(f"prompt template   : {cfg['arithmetic']['prompt_template']!r}"
      f"  operands {cfg['arithmetic']['operand_range_start']}-{cfg['arithmetic']['operand_range_end']}")

from collections import Counter
heads = fr["head_results"]
freqs = Counter(h["dominant_frequency"] for h in heads.values())
print(f"\ndominant frequency across {len(heads)} significant heads:")
for k, n in sorted(freqs.items()):
    print(f"    k={k}: {n:>4} heads  {'#' * int(60 * n / len(heads))}")

ratios = np.array([h["power_ratio"] for h in heads.values()])
print(f"\npower ratio: median {np.median(ratios):.2f}, max {ratios.max():.2f}")
print("\nstrongest heads:")
for name, h in sorted(heads.items(), key=lambda kv: -kv[1]["power_ratio"])[:5]:
    print(f"    {name:<8} L{h['layer']:<3} H{h['head']:<3} k={h['dominant_frequency']}  ratio={h['power_ratio']:.2f}")

print("\nNote: k=1 (ordinal) dominates the head-level scan. Compare with section 5 --")
print("      the frequency that dominates by VARIANCE is not the causally important one.")
''')

nb.code(r'''
cmd = 'python experiments/fourier_decomposition.py --model gemma-2b --layer-sweep "5,6,...,25" --device cuda'
shell(cmd)
if tier(2):
    subprocess.run(cmd, shell=True, cwd=REPO, check=True)
''')

nb.md(r"""
## 5 · Steps 4–7 — causal validation, and the ***k*=5 paradox**

| Step | Script | Tests |
|---|---|---|
| 4 | `fourier_knockout.py` | **Necessity** — zero the 9D subspace, measure damage |
| 5 | `fisher_phase_shift.py`, `fisher_patching.py` | **Sufficiency** — patch only the subspace |
| 6 | `fourier_phase_rotation.py` | **Steering** — rotate phase, predict shift mod 10 |
| 7 | `steering_improvements.py` | $W_U$-informed steering |

**Necessity — the cleanest result in the paper.** Zeroing the 9D subspace from
computation to readout drops accuracy to **near chance** in all three models. A
**matched-dimension random subspace ablation has *zero* effect** — perfect
specificity. Single-layer ablation does partial damage (11–77%), revealing a
distributed pipeline that actively maintains the information.

**Sufficiency.** At readout, standard Fisher at 10D transfers **85%** (Gemma,
Phi-3) and **100%** (LLaMA); contrastive Fisher at 9D transfers 83–100%. The two
subspaces agree to **>0.97 principal cosine**.

> ### ⚠ The *k*=5 paradox — a hypothesis we falsified
>
> At Gemma's computation layer $k=5$ (parity) is **the dominant SVD direction**:
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
> **The lesson, which generalises well beyond this paper: variance is not
> causation.** Had we ranked directions by singular value and stopped, we would
> have reported exactly the wrong mechanism.

**Steering, and its ceiling.** Coherent rotation at computation layers:
Gemma L19 **28%** exact, Phi-3 L26 **28%**, LLaMA L20 **69%**, with marked
backward-shift asymmetry. The bottleneck is the **encoding–readout gap**: the
Fourier–unembed overlap is only **8–11%**.

> **⚠ Readout layers are immune.** At Gemma L25 / LLaMA L27 every steering method
> lands at **≤10% (chance)**, and the overlap falls to **0.2%** at LLaMA L27. By
> readout the subspace has been absorbed into the unembedding-aligned
> representation. **If you steer at the readout layer you will measure nothing,
> and it is not a bug.**
""")

nb.code(r'''
print("Gemma-2B computation layer -- variance rank vs causal importance\n")
print(f"{'frequency':<12}{'role':<16}{'sigma':>8}{'% variance':>12}{'ablation damage':>18}")
print("-" * 68)
for k, role, sig, var, dmg in [("k=5", "parity", "135", "71%", "0.4-1.0%"),
                               ("k=1", "ordinal", "-", "-", "~40%"),
                               ("k=2", "mod-5", "-", "-", "~40%")]:
    print(f"{k:<12}{role:<16}{sig:>8}{var:>12}{dmg:>18}")

print("\n  Ranked by variance : k=5 first.")
print("  Ranked by causation: k=5 LAST.")
print("\n  => Variance is not causation. Ablate before claiming a mechanism.")
''')

nb.code(r'''
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
print("Reminder: run these at the COMPUTATION layer. At readout layers every")
print("          steering method returns chance-level results by construction.")
if tier(2):
    for label, c in steps:
        print(f"=== {label} ===")
        subprocess.run(c, shell=True, cwd=REPO, check=True)
''')

nb.md(r"""
## 6 · Steps 8–9 — component attribution

| | |
|---|---|
| **Scripts** | `fourier_head_attribution.py` (Step 8) · `neuron_trig_analysis.py` (Step 9) |
| **Purpose** | Identify which attention heads and MLP neurons *write* the Fourier subspace. |

**Expected outcome.** At Gemma L19, **809 / 9216 neurons (8.8%)** exceed 80%
frequency purity, dominated by $k=1$ and $k=5$ tunings. Produces
`neuron_frequency_tuning.png`.

> **⚠ Note the tension with §5.** $k=5$ is among the *most common* neuron tunings
> and is *causally inert*. **Prevalence of a tuning is not evidence of function
> either** — the same trap as variance ranking, one level down.
""")

nb.code(r'''
for label, c in [
    ("Step 8  head/MLP attribution",
     "python experiments/fourier_head_attribution.py --model gemma-2b --comp-layer 19 --device cuda"),
    ("Step 9  per-neuron frequency tuning",
     "python experiments/neuron_trig_analysis.py --model gemma-2b --layer 19 --device cuda"),
]:
    print(f"# {label}\n{c}\n")

print("Expected (Gemma L19): 809/9216 neurons (8.8%) above 80% purity;")
print("                      dominant tunings k=1 and k=5.")
print("\nCaution: k=5 is prevalent AND causally inert (section 5). Prevalence != function.")
''')

nb.md(r"""
## 7 · Steps 10–11 — the computation mechanism

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
**0.964** for Gemma — the strongest direct evidence that the model performs
angle addition rather than something merely correlated with it.

**Step 11** stratifies by carry vs no-carry: the ones-digit circuit should be
largely carry-invariant, since carry affects the tens digit.
""")

nb.code(r'''
rng = np.random.default_rng(1)
a, b = rng.uniform(0, 2*np.pi, 10000), rng.uniform(0, 2*np.pi, 10000)
lhs = np.cos(a) * np.cos(b)
rhs = 0.5 * (np.cos(a - b) + np.cos(a + b))
print(f"max |cos(a)cos(b) - 0.5[cos(a-b)+cos(a+b)]| = {np.abs(lhs - rhs).max():.2e}   (identity holds)")

print("\nThe cos(a+b) term is literally the sum of the two operand angles.")
print("Finding it in the conditional-expectation tensor is what a Fourier adder looks like.\n")
print("Our measured sigma^2-weighted identity score (Gemma): 0.964  (1.0 = exact)\n")

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

nb.md(r"""
## 8 · Steps 12–15 — generalisation

| Step | Script | Question |
|---|---|---|
| 12 | `generalization_tests.py` | Subtraction, operand substitution, multi-digit |
| 13 | `fourier_umap.py` | UMAP view of the digit manifold |
| 14 | `multilayer_freq_ablation.py` | Per-frequency ablation across layer ranges |
| 15 | `multidigit_circuit.py` | Multi-digit circuit (Gemma) |

Step 14 produced the per-frequency damage numbers underpinning the *k*=5 paradox
in §5, so it is worth running despite being labelled "advanced".

> **⚠ UMAP is for illustration only.** A non-linear, stochastic embedding with no
> distance guarantees. A clean ring in UMAP is **not** evidence of circular
> structure — §3's DFT is. Never let a UMAP plot carry an argument.
""")

nb.code(r'''
for label, c in [
    ("Step 12  generalisation",
     "python experiments/generalization_tests.py --model gemma-2b --comp-layer 19 --device cuda"),
    ("Step 13  UMAP (illustrative only)",
     "python experiments/fourier_umap.py --model gemma-2b --comp-layer 19 --device cuda"),
    ("Step 14  per-frequency ablation (drives the k=5 result)",
     "python experiments/multilayer_freq_ablation.py --model gemma-2b --comp-layer 19 --readout-layer 25 --device cuda"),
    ("Step 15  multi-digit circuit",
     "python experiments/multidigit_circuit.py --model gemma-2b --device cuda"),
]:
    print(f"# {label}\n{c}\n")
''')

nb.md(r"""
## 9 · Producing the paper figures

> **⚠ The six figures `paper/` needs are not in this repository.** `paper/main.tex`
> sets `\graphicspath{{../mathematical_toolkit_results/paper_plots/}}`, which is
> generated output that was never tracked. **`paper/` will not compile until you
> regenerate them.**

The plotting scripts read JSON written by the analysis steps, so each figure has
a prerequisite. The data steps emit **no images** — all rendering happens in the
three plotting scripts.
""")

nb.code(r'''
figures = [
    ("layer_scan_curves.png",             "generate_paper_plots.py / generate_missing_plots.py", "Step 1  arithmetic_circuit_scan_updated.py"),
    ("fourier_heatmap_cross_model.png",   "generate_paper_plots.py",   "Step 3  fourier_decomposition.py"),
    ("energy_explosion.png",              "generate_paper_plots.py",   "Step 3  fourier_decomposition.py"),
    ("ablation_curves.png",               "generate_missing_plots.py", "Step 4  fourier_knockout.py + multilayer_freq_ablation.py"),
    ("neuron_frequency_tuning.png",       "generate_missing_plots.py", "Step 9  neuron_trig_analysis.py"),
    ("eigenvector_fourier_cross_model.png","plot_eigenvector_dft.py",  "Step 2  eigenvector_dft.py"),
]
print(f"{'figure':<38}{'plotted by':<52}{'needs'}")
print("-" * 132)
for fig_name, plotter, prereq in figures:
    print(f"{fig_name:<38}{plotter:<52}{prereq}")

plots_dir = REPO / "mathematical_toolkit_results" / "paper_plots"
have_figs = sorted(p.name for p in plots_dir.glob("*.png")) if plots_dir.is_dir() else []
print(f"\npaper_plots/ present: {have_figs or 'NONE -- paper/ will not compile yet'}")
print("(.gitignore was adjusted so this directory becomes trackable once populated.)")
''')

nb.md(r"""
## 10 · Reading these results responsibly

### 10.1 What is solid, what is refuted

| Claim | Evidence | Confidence |
|---|---|---|
| Digit codes are a 9D Fourier basis of ℤ/10ℤ | 2/2/2/2/1 assignment, purity >50% | **High** |
| That subspace is causally necessary | Knockout → chance; matched-dim random → zero effect | **High** — the specificity control is what makes it |
| Addition is angle addition | CP trig-identity score 0.964 (Gemma) | **High** |
| Fourier phase steering is practical | 28–69% exact; ≤10% at readout | **Provisional** — bounded by the 8–11% encoding–readout gap |
| Parity (*k*=5) drives the computation | 71% of variance, 0.4–1.0% ablation damage | **Refuted** — epiphenomenal |

### 10.2 Three ways to misuse this notebook

**Ranking directions by variance and stopping.** The *k*=5 paradox is the
cautionary tale: the largest singular direction, 71% of variance, is causally
inert. **Any direction that has not been ablated has not earned a causal claim.**

**Steering at the wrong layer.** At readout layers everything reads chance-level
because the subspace has been absorbed into the unembedding. Reporting that as a
negative result is a measurement error.

**Treating neuron-tuning prevalence as function.** §6: *k*=5 is both prevalent
and inert.

### 10.3 If a number here disagrees with your run

1. **Prompt protocol** — teacher-forced vs direct-answer (§1.1). The most common
   silent failure.
2. **Layer indices** — comp-layer is *discovered* by Step 1, not assumed. The
   cached defaults are for the exact checkpoints we used.
3. **Model revision** — "gemma-2-2b" today may not be the weights we ran. Pin a
   revision hash.
4. **Superseded sections of the plan** — "Phase 0" and S5 reference removed
   scripts and will not run.

## 10.4 Reproducibility ledger
""")

nb.code('''
REPLAYED = sorted(FR.glob("*.json"))
''' + LEDGER)

nb.write()

print("\nBoth notebooks generated.")
