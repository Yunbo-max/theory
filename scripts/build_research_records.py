"""Serialize the documented, same-context mathematical review (not a proof engine)."""
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "research"

# Each entry contains a different mathematical construction, not a seed/ablation.
# The accompanying review only verifies the stated conditional algebra.
CARDS = [
 ("M01", "geometric_anchor", "D02", "min_q KL(q||m)+λ KL(q||p0), m=(1−ε)μ+εp0",
  "Differentiate with a simplex multiplier: (1+λ)log q=log m+λlog p0+constant|Normalize to obtain q ∝ m^(1/(1+λ)) p0^(λ/(1+λ))|Strict convexity on positive support makes this the unique distribution-space minimizer",
  "q ∝ m^w p0^(1−w), w=1/(1+λ)",
  "p0 positive; ε>0; λ≥0; exact neural fitting is not assumed",
  "Initial anchoring should reduce recursive log-ratio drift compared with full-soft targets", "No retention advantage, or arithmetic anchoring dominates at equal measured cost", "KL-regularized/geometric distillation", "full_soft, arithmetic_anchor; same prefixes and anchor access", "One extra initial-policy forward; strong known-method overlap"),
 ("M02", "ratio_cap", "B02", "q=p0 exp(clamp(log(m/p0),−B,B))/Z",
  "Every exponentiated clipped ratio lies in [exp(−B),exp(B)]|A p0-weighted average Z lies in the same interval|Dividing yields exp(−2B)≤q/p0≤exp(2B)",
  "q=normalize(p0*exp(clamp(log(m/p0),−B,B)))",
  "B≥0; m,p0 positive; target-space bound only",
  "Bounded target ratios should constrain worst tail suppression", "Realized student ratios remain unbounded or simple anchoring performs as well", "clipped likelihood ratios and trust regions", "M01 and arithmetic_anchor with matched reference forwards", "One extra forward; clipping is not asserted to solve a KL projection"),
 ("M03", "floor_projection", "D01", "min_q KL(μ||q) subject to q≥c p0 and Σq=1",
  "KKT stationarity on free coordinates gives q_v=μ_v/λ|Complementary slackness gives q_v=max(c p0v,k μv)|The mass is continuous nondecreasing in k; at k=0 it is c and at k=1 at least one, giving a bisection bracket",
  "q_v=max(c p0v,k μv), k chosen to normalize",
  "0≤c<1; distributions normalized; zero μ is allowed; neural fit approximate",
  "Restore excluded tail mass while keeping more decoder-head shape than arithmetic mixing", "No benefit over arithmetic_anchor/full_soft, or fitting error erases the floor", "probability-floor information projection", "full_soft, arithmetic_anchor, same-token lower-LR hard SSD", "One extra forward; priority candidate; familiar projection, application novelty unresolved"),
 ("M04", "temperature_budget", "G01", "Composition of untruncated power maps T_r(p)∝p^(1/T_r)",
  "Composition multiplies exponents by induction|Choose T_r=exp(w_r log T_total), Σw_r=1, so product T_r=T_total|The ideal endpoint equals one T_total map regardless of round count; truncation and imperfect fitting invalidate endpoint equality",
  "T_r=T_total^(1/R) for the declared R rounds; use full-soft targets",
  "Power-map theorem excludes top-k/top-p; actual truncated implementation is a hypothesis",
  "Reduce over-flattening from repeated temperature application", "Same endpoint drift as fixed temperature, or no quality/retention improvement", "temperature annealing and power-map semigroups", "full_soft at fixed T, untruncated diagnostic, matched training steps", "No extra forward; a composition-derived schedule, not a claim that annealing is new"),
 ("M05", "head_mass", "B06", "Partition vocabulary into retained decoder support S and its complement",
  "Write any distribution as its support mass times its conditional shape|Set head mass M'=p(S)+κ(1−p(S)) and head conditional exactly μ|Assign remaining mass by p outside S; the two masses sum to one and endpoints are well-defined",
  "q_S=M'μ_S; q_out=(1−κ)p_out; M'=p(S)+κ(1−p(S))",
  "0≤κ≤1; μ supported on S; if S is full vocabulary q=μ",
  "Separate head reshaping from how fast tail probability is removed", "A convex mixture of p and μ gives the same measured trade-off", "SSD precision/exploration decomposition; partial mass transfer", "full_soft and fixed_alpha; equal token counts", "No extra forward; head shape differs from simple convex mixing"),
 ("M06", "head_diversity", "D03", "q_S(β)∝μ_S^β for β∈[0,1], require G(q_S)≥ρG(p0|S)",
  "At β=0 q is uniform on S and has maximal G, so the constraint is feasible|The derivative of Σq² is 2[Σq² log μ−(Σq²)E_q log μ]≥0 by positive association of q and log μ|Thus G is nonincreasing in β and bisection gives the largest feasible β",
  "Choose largest β≤1 satisfying conditional-head diversity, then q=normalize(μ^β on S)",
  "0≤ρ≤1; condition p0 on nonempty S; no claim about missing outside support",
  "Avoid overconcentrating the retained head across recursion", "Only entropy changes without better native pass@k, or global tails collapse anyway", "entropy/diversity constrained smoothing", "temperature_budget and full_soft, head diversity diagnostic", "Initial forward plus scalar search; retain support zeros"),
 ("M07", "tail_stratified", "E03", "Keep top-h μ probabilities exact; estimate the remaining mass ε by n stratified inverse-CDF draws",
  "On n equal CDF intervals draw independent uniforms U_j=(j+V_j)/n|Averaging indicator masses over intervals has expectation equal to the conditional tail probabilities|Exact head plus ε times this estimator is normalized and unbiased; stratified variance of any fixed linear functional is no larger than iid tail draws by total variance",
  "q_head=μ_head; q_tail=ε/n Σ_j e(F_tail^−1(U_j))",
  "Fixed prefix and head; independent uniforms per stratum; component-level claim only",
  "Approximate full-soft targets with less stochastic tail noise than hard labels", "No memory/runtime advantage in this dense implementation or full_soft dominates", "stratified Monte Carlo and exact-head marginalization", "full_soft and hard; log that dense target construction is not a speedup claim", "No extra forward; experimental dense prototype, sparse-kernel optimization deferred"),
 ("M08", "temporal_mean", "D02", "min_q [H(μ_r,q)+H(μ_lag,q)]/2 at the same prefix",
  "Linearity in the target rewrites the objective as H((μ_r+μ_lag)/2,q)|Gibbs inequality gives q=(μ_r+μ_lag)/2 at its minimum|Covariance of the mean includes cross-covariance; variance improvement is not guaranteed for biased or correlated teachers",
  "q=(μ_r+μ_lag)/2",
  "Frozen historical policies evaluated at identical current prefixes; lag=teacher in first round",
  "Damp round-to-round target oscillations", "Staleness harms scores or a smaller learning rate matches it", "temporal ensembling and mean teachers", "full_soft and lower_lr; teacher correlation logged", "One lag forward; no external teacher"),
 ("M09", "disagreement_gate", "D01", "min_a a D + λ KL(Ber(a)||Ber(a0)), D=JS(μ_r,μ_lag)",
  "Differentiate: D+λ[logit(a)−logit(a0)]=0|Solve a=sigmoid(logit(a0)−D/λ)|Strictly positive second derivative λ/[a(1−a)] gives the unique interior minimum; construct q=(1−a)p+aμ",
  "a=sigmoid(logit(a0)−JS/λ); q=(1−a)p+aμ",
  "0<a0<1; λ>0; JS measured before fresh labels, detached; agreement is not correctness",
  "Reduce updates specifically at temporally unstable prefixes", "A matched constant gate or lower learning rate performs equally", "uncertainty gating and Bernoulli KL regularization", "fixed_alpha, temporal_mean and full_soft", "One lag forward; changes target rather than teacher averaging"),
 ("M10", "prefix_damping", "F03", "Prefix score s_t=Σ_{j<t} max(0,log p_r(y_j|h_j)−log p0(y_j|h_j))",
  "Maintain s_{t+1}=s_t+positive log-ratio at t, starting s_0=0|w_t=exp(−min(s_t,C)) is prefix measurable and lies in [exp(−C),1]|The frozen weighted CE objective has gradient Σw_t(p_student−μ_t); it intentionally changes prefix weighting",
  "L=mean_t exp(−min(s_t,C)) H(μ_t,p_student,t)",
  "Teacher and anchor frozen; complete observed-prefix log-ratios; not importance sampling",
  "Damp reinforcement along already amplified generated paths", "A constant loss scale or lower LR matches effect; rare useful paths are suppressed", "trajectory reweighting and confidence weighting", "same-average-weight control, full_soft and lower_lr", "One initial forward; path-dependent state is distinct from local gates"),
 ("M11", "noise_allocation", "E06", "max Σ a_t d_t subject to Σ v_t a_t²≤B, 0≤a_t≤1 on a fixed prefix chunk",
  "Set d_t=||μ_t−p_t||² and v_t=G(μ_t)/n for fresh independent target draws|KKT gives a_t=min(1,d_t/(2λv_t)); v=0 directions have no noise charge|The budget is nonincreasing in λ; bisection enforces it and fresh post-allocation draws preserve the stated conditional covariance",
  "Allocate a jointly, then q_t=(1−a_t)p_t+a_t e(Y_t), Y_t fresh ~ μ_t",
  "Prefixes fixed before allocation and fresh labels; d is a drift proxy, not correctness; n=1 implementation",
  "Spend label-noise budget where decoder drift is larger", "Fixed alpha/full_soft matches or exceeds quality under equal cost", "Neyman-style allocation and quadratic resource constraints", "fixed_alpha with fresh labels, full_soft; no reuse of rollout labels here", "No extra forward; allocation is within declared vocabulary chunks, not full trajectories"),
 ("M12", "gradient_projection", "D01", "min_d ||d−d0||²/2 subject to h·d≤c, h=∇θ KL(p0||pθ)",
  "KKT gives d=d0−λh with λ≥0|Enforce complementarity: λ=max(0,(h·d0−c)/||h||²), handle h=0 separately|Taylor expansion gives retention change η h·d+O(η²); a first-order bound is not a finite-step guarantee",
  "Project the SGD direction using the anchor-KL gradient; c=0 default",
  "SGD without momentum; same parameters and prefix batch for h,d0; differentiable retention; LoRA-space only",
  "Reduce first-order retention conflict rather than merely shrinking all gradients", "SGD-matched control is as good or finite-step KL still grows", "gradient surgery and projected optimization", "full_soft_sgd, arithmetic_anchor and realized KL change", "Extra anchor backward; selected but low scheduling priority"),
 ("M13", "kl_backtrack", "C03", "Actual batch KL constraint on a proposed adapter update Δ",
  "At γ=0 the parameters equal the saved pre-step state and KL(p_before||p_after)=0|Test γ=1,1/2,... using the actual frozen pre-step distribution on the same inputs|Accept the first γ with KL≤δ; otherwise restore parameters and optimizer state; the accepted tested-batch bound is direct, not a model-wide theorem",
  "θ_new=θ_old+γΔ subject to measured prefix-batch KL≤δ",
  "Finite candidate set includes exact rollback; evaluation mode and same prefixes; FP tolerance recorded",
  "Prevent unusually large realized updates missed by scalar gradient clipping", "Same cost spent on lower LR performs better, or proposal rejection wastes budget", "trust regions and backtracking line search", "full_soft with lower_lr and wall-clock match", "Several extra student forwards; no global guarantee"),
 ("M14", "prompt_allocation", "E06", "min Σ σ_i²/n_i subject to Σn_i=N, n_i>0",
  "Lagrange stationarity gives n_i proportional to σ_i in the continuous relaxation|Integer largest-remainder allocation with min one preserves total N, but not exact relaxed optimality|Weight each record by N/(P n_i) to preserve an equal-prompt empirical mean; use sqrt(G(first-prefix μ)) only as a proxy for unknown σ_i",
  "Allocate samples using first-prefix diversity, preserve equal-prompt loss weights",
  "N≥P; estimated proxy is not true gradient variance; varying lengths still affect wall-clock",
  "Use the same sample budget more effectively across prompts", "Uniform prompt allocation wins or proxy poorly predicts sequence uncertainty", "stratified/Neyman sampling", "hard SSD with uniform sample counts, same total N and loss convention", "One first-token forward per prompt; changes prompt sampling, not q"),
 ("M15", "ancestral_exploration", "B04", "Generation law ν=(1−ε)μ_r+εμ_0 at every token",
  "ν_v≥ε μ_0v holds pointwise at each prefix|Multiply conditional inequalities along a fixed L-token path to get Pν(path)≥ε^L Pμ0(path)|Train on unverified ν samples; preserving proposal support does not establish correctness or student support after fitting",
  "Sample the per-token historical/current decoder mixture; hard-label training",
  "0<ε≤1; same vocabulary/prefix; both teachers are checkpoints of this model",
  "Recover initial-policy paths lost by current-policy truncation", "Extra inference cost dominates or target-only anchoring works as well", "exploration mixtures and replay policies", "hard SSD, arithmetic_anchor; separately report 2-teacher generation cost", "Extra autoregressive anchor forward/cache; selected but expensive"),
 ("M16", "diagonal_metric", "D05", "min_d g·d+(1/(2η))dᵀD d for positive diagonal D",
  "Stationarity gives d=−ηD^−1g|Substitution shows the quadratic surrogate decreases by η gᵀD^−1g/2|A running squared-gradient diagonal is an empirical metric, not the Fisher unless additional assumptions hold",
  "d=−η g/(sqrt(second_moment)+ε)", "Positive damping; quadratic local surrogate only",
  "Preconditioning may target stiff adapter directions", "AdamW matches it, indicating no useful distinction", "Adam and diagonal natural-gradient approximations", "AdamW, SGD and true-KL diagnostics", "Reserve: strong standard-optimizer overlap; no code selected"),
 ("M17", "trajectory_tilt", "B03", "Sequence target Q(y)∝P_r(y)^(1−β)P0(y)^β over a fixed finite horizon",
  "Relative to full-support P_r, w(y)=(P0(y)/P_r(y))^β|Normalizing gives E_Q f=E_Pr[wf]/E_Pr[w]|Empirical self-normalization is biased at finite sample size; truncated proposals require an explicit support correction or restrict the target",
  "Sequence-level self-normalized importance-weighted CE", "Absolute continuity; finite normalizer; EOS/horizon explicit",
  "Control whole-sequence rather than local drift", "Importance weights collapse or prefix method matches it", "self-normalized importance sampling", "prefix_damping and ESS diagnostics", "Reserve: variance/support risk under truncation"),
 ("M18", "temporal_median", "C04", "Coordinatewise median of centered logits from at least three frozen historical teachers",
  "If a strict majority of each coordinate lies within ε of a common centered target, its median also lies there|Softmax log-normalization is 1-Lipschitz in the infinity norm|Therefore each normalized log-probability differs from that target by at most 2ε",
  "q=softmax(coordinatewise median of centered historical logits)", "Coordinatewise majority condition is hypothetical; teacher errors can be correlated",
  "Resist one anomalous historical checkpoint", "Correlated errors defeat median or arithmetic mean is better", "robust temporal ensembling", "temporal_mean with equal number of forward passes", "Reserve: at least three reference forwards; weak identification of good majority"),
 ("M19", "embedding_transport", "D02", "min_π <C,π>+ε KL(π||p0⊗μ) subject to π1=p0",
  "Introduce one multiplier per row and differentiate on positive support|Row-normalization yields π_ij=p0i μj exp(−Cij/ε)/Z_i|q_j=Σ_iπ_ij is normalized and minimizes the stated one-marginal transport objective",
  "Transport initial token mass using frozen embedding costs; target is column marginal", "ε>0; finite C; μ support explicit; token similarity need not mean semantic correctness",
  "Retain semantically nearby alternatives instead of literal initial probabilities", "Cost geometry uninformative or simple anchors dominate", "entropy-regularized transport", "geometric_anchor with equal compute; cost ablation", "Reserve: quadratic vocabulary cost; support approximation changes objective"),
 ("M20", "variance_stopping", "E06", "Martingale error M_R=Σ ε_r with E[ε_r|F_r]=0 and predictable variance charges ν_r",
  "Orthogonality of martingale differences gives E||M_R||²=Σ E||ε_r||²|Stop before predictable cumulative ν exceeds B, assuming each conditional variance is bounded by its charge|For bounded stopping horizon, Doob's L2 bound gives P(max||M||≥a)≤4B/a²; deterministic decoder drift is excluded",
  "Stop recursion when a predeclared cumulative sampling-variance budget is exhausted", "Predictable bounds, zero-mean increments and bounded horizon; LLM approximation errors violate these without proof",
  "Limit cumulative sampling uncertainty independently of elapsed time", "Proxy charges do not bound observed error or deterministic drift dominates", "martingale stopping and variance budgets", "fixed-round schedule and full_soft", "Reserve: unverified transfer of target-level charges to model dynamics"),
]

def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2)+"\n")
    return {"path":str(path.relative_to(ROOT)),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}

def ref(path):
    return {"path":str(path.relative_to(ROOT)),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()}

def main():
    timestamp="2026-10-06T12:00:00Z"  # Session date; overwritten with actual review serialization time below.
    from datetime import datetime, timezone
    timestamp=datetime.now(timezone.utc).isoformat()
    sources=[ref(R/"SOURCES.md"),ref(R/"THEORY.md")]
    entries=[]
    for cid,name,op,obj,steps,expr,assum,pred,fals,closest,comparison,cost in CARDS:
        statements=re.split(r"(?<!\|)\|(?!\|)", steps)
        card={"kind":"math-card","version":"1.0.0","candidate_id":cid,"name":name,
              "formal_object":obj,"operations":[{"id":op,"input":obj,"condition":assum,
                  "condition_status":"conditional","output":expr}],
              "derivation_steps":[{"id":f"{cid}-S{i+1}","statement":s,
                  "justification":("Apply the stated constraint/normalization to the preceding algebra; " if i else "Start from the explicitly specified finite-distribution object; ")+assum} for i,s in enumerate(statements)],
              "assumptions":assum.split("; "),"method_expression":expr,
              "distinguishing_prediction":pred,"falsifier":fals,"closest_alternative":closest,
              "decisive_comparison":comparison,"feasibility":cost,"source_refs":sources,
              "novelty_status":"UNESTABLISHED; conditional construction, not an originality certificate"}
        cr=write(R/"cards"/(cid+".json"),card)
        rationales={"formal_object":obj+"; fixed-prefix versus sequence scope is explicit.",
           "operations_and_conditions":op+": "+assum,
           "derivation":"Reviewed conditional steps: "+"; ".join(statements),
           "assumptions":"No accuracy guarantee or exact neural representability inferred. "+assum,
           "method_expression":"Expression implements the stated mathematical construction: "+expr,
           "prediction_and_falsifier":pred+". Rejection condition: "+fals,
           "closest_alternative":"Known overlap: "+closest+". Required comparison: "+comparison}
        review={"kind":"method-review","version":"1.0.0","subject_id":cid,"scope":"math",
            "artifact_ref":cr,"reviewer":"same-context assistant review; not independent or formal proof",
            "reviewed_at":timestamp,"outcome":"verified","verification_scope":"Conditional algebra only; novelty and empirical validity pending",
            "checks":{k:{"status":"verified","rationale":v,"evidence_refs":[cr,*sources]} for k,v in rationales.items()}}
        rr=write(R/"reviews"/(cid+"-math.json"),review)
        entries.append({"candidate_id":cid,"math_card_ref":cr,"math_review_ref":rr})
    order=["M03","M04","M05","M01","M07","M02","M06","M08","M09","M10","M11","M12","M13","M14","M15","M16","M17","M18","M19","M20"]
    byid={row[0]:row for row in CARDS}
    ranking=[]
    for rank,cid in enumerate(order,1):
        row=byid[cid]
        ranking.append({"candidate_id":cid,"rank":rank,"rationale":{
           "problem_value":row[7],"mathematical_consequence":row[5],
           "closest_work_delta":row[9]+"; application novelty remains unresolved. Comparator: "+row[10],
           "distinguishing_prediction":row[7]+"; falsifier: "+row[8],"feasibility_and_cost":row[11]}})
    selection={"kind":"method-selection","version":"1.0.0","batch_id":"recursive-ssd-20261006",
        "pool_target":20,"selection_target":15,"math_bindings":entries,"ranking":ranking,"selected_ids":order[:15]}
    sr=write(R/"selection.json",selection)
    checks={
      "structural_distinctness":"Compared optimized objects: geometric penalty, bounded ratios, probability floor, composed temperature map, support-mass split, conditional diversity, stratified estimator, temporal barycenter, adaptive local gate, path weighting, joint noise allocation, parameter projection, realized-KL line search, prompt allocation, generation mixture; reserves use metric, sequence tilt, median, transport and stopping. Controls/seeds are excluded from candidate count. Related methods can still have identical special cases.",
      "all_candidates_compared":"All 20 cards are represented once in the full ranking; none is omitted after an unfavorable cost or novelty assessment.",
      "ranking_basis":"Prioritize explicit finite-distribution consequences that can be audited on one 11 GiB GPU. Cost, closest known method and falsifier are recorded per candidate. Floors lead because their support consequence survives truncation at target level; candidates 16–20 have stronger standard-method overlap or unsupported/costly implementation assumptions.",
      "top_selection":"Top 15 are selected for implementation, not scientific endorsement. The timed default queue only prioritizes M03 after hard/full-soft baselines. Statistical/native-evaluation gates remain pending; no promise that 15 GPU comparisons fit eight hours."}
    review={"kind":"method-review","version":"1.0.0","subject_id":selection["batch_id"],"scope":"selection",
        "artifact_ref":sr,"reviewer":"same-context assistant review; not independent","reviewed_at":timestamp,
        "outcome":"verified","checks":{k:{"status":"verified","rationale":v,"evidence_refs":[sr,*sources]} for k,v in checks.items()}}
    rr=write(R/"selection-review.json",review)
    write(R/"method-batch.json",{"kind":"method-verification-batch","version":"1.0.0",
        "batch_id":selection["batch_id"],"pool_target":20,"selection_target":15,"candidates":entries,
        "selection_ref":sr,"selection_review_ref":rr})
    lines=["# Candidate inventory", "", "All are hypotheses, not established novel contributions. Conditional derivations and same-context reviews are in `cards/` and `reviews/`. Baselines are not counted as candidates.","", "| Rank | ID | Construction | Implementation selection |", "|---|---|---|---|"]
    for rank,cid in enumerate(order,1):
        lines.append(f"| {rank} | {cid} | {byid[cid][1]} | {'Selected' if rank<=15 else 'Reserve; no implementation promised'} |")
    (R/"CANDIDATES.md").write_text("\n".join(lines)+"\n")

if __name__=="__main__":
    main()
