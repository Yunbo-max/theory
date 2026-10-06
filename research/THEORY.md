# What the recursion can and cannot guarantee

Fix a prefix h. Let p be the frozen current policy and μ = Decode(p) its normalized temperature/top-k/top-p distribution. Let p₀ be the initial policy at the **same** prefix. All target distributions are detached inside an optimization phase. The next phase recomputes them from a newly frozen checkpoint.

## Sampling noise and drift are different

For n independent Yⱼ ~ μ and μ̂ = n⁻¹Σ e(Yⱼ), choose a before these draws and set q=(1−a)p+aμ̂. Then

    E[q] = (1−a)p+aμ,
    tr Cov(q) = a²(1−||μ||²)/n.

The cross terms vanish because centered draws are independent. For G(u)=1−||u||², expanding E||q||² gives

    E G(q) = G(E q) − a²G(μ)/n.

This is a fixed-prefix target-space identity. It does not prove lower variance of the full autoregressive parameter gradient: future prefixes depend on earlier draws and cross-covariances remain. Scalar a attenuates mean drift and noise together; its local signal-to-noise ratio need not improve. If a depends on the sampled Y, even the mean changes.

Under ideal exact fitting and **neutral** μ=p at each generation,

    E[G(pᵣ₊₁)|pᵣ] = (1−aᵣ²/nᵣ)G(pᵣ).

For deterministic aᵣ,nᵣ the expected diversity multiplies these factors. A small constant positive a merely delays neutral finite-sample collapse. Real LLM updates are neither exact distribution fitting nor neutral; the identity is a counterexample to an unconditional improvement claim, not an LLM prediction.

## Removing local label noise is insufficient

The full-soft control q=μ is mandatory. It integrates out the next-token label at a fixed prefix. With exact fitting, temperature T and **no truncation**, repeated soft distillation satisfies

    pᴿ(v) ∝ p⁰(v)^(1/Tᴿ).

This follows by substituting pᵣ₊₁(v) ∝ pᵣ(v)^(1/T) inductively and absorbing prefix-independent normalizers. For T>1 it approaches uniformity on its initial support; for T<1 it concentrates on maximizers. There is deterministic distribution drift even without label noise. Top-k/top-p and approximate fitting break this exact power-law formula.

## Proposed main constraint: retain initial probability mass

For c∈[0,1), consider min_q KL(μ||q) subject to Σq=1 and q_v≥c p₀v. KKT stationarity on free coordinates gives q_v=μ_v/λ; complementary slackness on active coordinates gives q_v=c p₀v. Thus

    q_v = max(c p₀v, k μ_v),  Σ_v max(c p₀v,k μ_v)=1.

The left side is continuous and nondecreasing in k; for normalized μ, k∈[0,1] brackets a root. Zero μ coordinates simply retain the floor. A bisection tolerance is a numerical, not statistical, error. This is a standard convex projection used here as a **candidate recursive constraint**, not a claimed novel KKT theorem.

After exact fitting every token retains its initial floor. For a fixed L-token path, a tokenwise floor implies P_new(path)≥cᴸP₀(path), which can be extremely weak for long paths. Finite-step LoRA fitting only approximates q: neither the pointwise floor nor a correctness guarantee automatically transfers to the trained model. Measure realized violations on shared diagnostic prefixes.

An arithmetic anchor q=(1−c)μ+c p₀ is a strong simple control. If the projection does not beat this control or full-soft distillation under matched costs, there is no evidence that its extra structure is valuable.

## Identifiability limit

Two worlds can have the same prompts and policy probabilities but opposite correct answers. Any algorithm observing only those probabilities and its samples produces the same update distribution in both worlds. It cannot guarantee improvement in correctness in both. Self-training can exploit a model's existing structure and sampling bias; probabilities alone cannot certify truth.

## Claim boundaries

- All elementary lemmas are conditional, same-context mathematical analysis; no independent reviewer or theorem prover was used.
- Benchmark hypotheses: useful recursive drift may coexist with support retention; the gain may disappear under arithmetic anchoring, lower learning rate or fixed-data controls.
- Actual GPU memory, runtime, accuracy and new-theorem novelty are unmeasured at handoff.
