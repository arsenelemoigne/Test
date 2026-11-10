```python
# Let's check the Phase 0 Go/No-Go rule and the Level definition with 1 answer of slack (p0(q) - 1/8)!
# Wait: let's do the exact probability & binomial math for the level definition:
# In Phase 0 and Phase 1:
# "The original's rate p0(q): share of its 8 natural (uncapped) answers that are right."
# "A question's level (needed budget) b*(q): the lowest rung at which it is right at least p0(q) - 1/8 of the time (as right as the original, one answer of slack)."
# Rungs per type: 0, 5, 10, 20, 30, 40, 50, 65, 80, 90th percentiles, then natural (11 rungs total: 10 cut rungs + natural).
# WAIT A MINUTE!
# Let's calculate what happens when you test a question at 10 rungs (0, 5, 10, 20, 30, 40, 50, 65, 80, 90th percentiles) with 8 samples at each rung, and define b*(q) as the FIRST rung where k_rung >= k_0 - 1 (i.e. p_rung >= p0 - 1/8)!
# Let's simulate this in Python!
# Suppose a question has true accuracy p(B) at budget B, and natural accuracy p_nat.
# Even more simply: suppose at some early rung B, the true accuracy is much lower than p_nat (say p(B) = p_nat - 0.20, a massive 20 percentage point accuracy drop!), OR suppose p(B) is flat or rising slowly.
# Because we take the MINIMUM rung over up to 10 rungs where k_rung >= k_0 - 1:
# 1) First, k_0 itself is a Binomial(8, p_nat) random variable!
# 2) Second, at each rung r = 1..R, k_r is a random variable (in Phase 1, 8 independent samples at each rung; in Phase 0, 8 natural answers cut at each rung and re-answered once each).
# 3) Third, even if at every single rung r, the TRUE accuracy were exactly p(b*(q)) = p_nat - 1/8 = p_nat - 12.5 percentage points(!), the expected accuracy loss from "one answer of slack" (1/8 = 12.5 percentage points!) is HUGE!
# Wait: 1/8 = 12.5 percentage points! The team's accuracy budget is -1.0 percentage point (-0.01), and the Phase 0 oracle bar is -0.5 percentage points (-0.005), yet they gave EVERY QUESTION 1/8 = 12.5 percentage points of slack PLUS multiple-testing stopping bias across 10 rungs!

# Let's run a Monte Carlo simulation of Phase 0 and Phase 1 level assignment to quantify both:
# (a) The multiple-testing / stopping-time selection bias (winner's curse of picking the lowest rung out of 10 with 8 samples where k_r >= k_0 - 1), AND
# (b) The true accuracy loss vs reported "oracle" accuracy in Phase 0!

np.random.seed(42)

# Wait: in Phase 0, how is the oracle curve computed?
# Let's check LADDER_DESIGN_FOR_REVIEW.md Section 6:
# "2. The original answers each 8 times uncapped -> p0 per question and the per-type rungs.
#  3. Each natural answer is cut at each rung and re-answered (adaptive: a question stops climbing at its level).
#  4. Report per type: the distribution of levels; the oracle curve (tokens and accuracy if every question used exactly its level), the ceiling of the method...
#  5. Go / no-go: continue only if the oracle saves >= 60% at accuracy >= original - 0.5 (official weighting) on these questions"

# HOLD ON! Read steps 2, 3, 4, 5 of Phase 0 again!
# If b*(q) is selected on the SAME 8 re-answers as the first rung where p_rung(q) >= p0(q) - 1/8,
# and then the "oracle curve" reports the accuracy of those SAME 8 re-answers at b*(q)...
# Wait: by definition of b*(q), for EVERY question with p0(q) > 0, the measured accuracy on those 8 samples at b*(q) is AT LEAST p0(q) - 1/8, and often HIGHER than p0(q) because it stops at the first rung that hits >= k0 - 1!
# Wait: what is the TRUE out-of-sample accuracy at the selected rung b*(q) vs the in-sample "oracle" accuracy reported on the same 8 samples?!
# Let's simulate this on realistic A(B) curves matching Qwen3-0.6B's actual data from Track 2g (benchlot / lot B / lot A)!

# Let's check the actual empirical A(B) curves from benchlot (L1515) and Track 2g (L1230):
# Rungs: 0, p5, p10, p20, p30, p40, p50, p65, p80, p90, nat (11 rungs).
# Let's model realistic true success curves p_true(q, r) for SVAMP, MBPP+, IFEval, BFCL:
rungs_pct = np.array([0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 0.90, 1.00])

def sim_phase0_oracle(p_curve_fn, n_q=10000):
    # p_curve_fn returns (n_q, 11) array of true accuracy at each rung for each question
    P_true = p_curve_fn(n_q) # shape (n_q, 11)
    # Step 2: 8 uncapped natural answers -> k0 ~ Binomial(8, P_true[:, -1])
    k0 = np.random.binomial(8, P_true[:, -1])
    # Step 3: 8 answers at each rung -> kr ~ Binomial(8, P_true[:, r])
    K_r = np.random.binomial(8, P_true)
    # Note: at the last rung (natural, index 10), it's the natural answer itself, so K_r[:, 10] = k0
    K_r[:, 10] = k0
    
    # Level b*(q): lowest rung r where K_r[:, r] >= k0 - 1 (for k0 > 0; if k0 == 0, what happens? "Never-solved questions (p0=0) get no stop early pairs", say they stay at natural or rung 0)
    chosen_r = np.full(n_q, 10, dtype=int)
    for i in range(n_q):
        if k0[i] > 0:
            for r in range(11):
                if K_r[i, r] >= k0[i] - 1:
                    chosen_r[i] = r
                    break
    
    # In-sample measured accuracy at chosen_r vs k0:
    in_sample_acc_orig = k0.mean() / 8.0 * 100
    in_sample_acc_oracle = K_r[np.arange(n_q), chosen_r].mean() / 8.0 * 100
    
    # TRUE out-of-sample accuracy at chosen_r vs true natural accuracy:
    true_acc_orig = P_true[:, -1].mean() * 100
    true_acc_at_chosen = P_true[np.arange(n_q), chosen_r].mean() * 100
    
    mean_rung_pct = rungs_pct[chosen_r].mean() * 100
    return {
        'in_sample_diff': in_sample_acc_oracle - in_sample_acc_orig,
        'true_diff': true_acc_at_chosen - true_acc_orig,
        'optimism_bias': (in_sample_acc_oracle - in_sample_acc_orig) - (true_acc_at_chosen - true_acc_orig),
        'mean_budget_pct': mean_rung_pct,
        'mean_saved_think_pct': 100 - mean_rung_pct
    }

# Case 1: Suppose true accuracy increases smoothly with budget (like SVAMP: 20.6% at 0 -> 88.7% at nat, settling late as in benchlot)
def svamp_curve(n_q):
    # Mix of easy (40%), medium (45%), hard (15%) questions
    # Easy settles by p20-p30, medium settles by p65-p80, hard needs p90-nat
    u = np.random.uniform(0, 1, size=(n_q, 1))
    # Threshold where question settles:
    half_pt = np.where(u < 0.20, 0.0, np.where(u < 0.50, 0.35, np.where(u < 0.85, 0.65, 0.90)))
    p_max = np.where(u < 0.85, 0.95, np.where(u < 0.93, 0.60, 0.05))
    p_min = np.where(u < 0.20, 0.85, 0.05)
    # Sigmoid transition around half_pt
    P = p_min + (p_max - p_min) / (1.0 + np.exp(-12.0 * (rungs_pct[None, :] - half_pt)))
    P[:, 0] = np.where(u[:, 0] < 0.20, 0.85, 0.05) # ~20.6% overall at B=0
    return P

# Case 2: Suppose true accuracy is moderately flat (like Code / IFEval / Tools where B=0 is 41-71% and nat is 52-82%)
def code_curve(n_q):
    u = np.random.uniform(0, 1, size=(n_q, 1))
    # 35% right even at B=0, 25% need thinking (settle around 0.2-0.7), 40% never solved or rarely solved
    p_min = np.where(u < 0.35, 0.85, np.where(u < 0.65, 0.25, 0.02))
    p_max = np.where(u < 0.35, 0.92, np.where(u < 0.65, 0.75, 0.05))
    half_pt = np.where(u < 0.35, 0.05, 0.45)
    P = p_min + (p_max - p_min) / (1.0 + np.exp(-8.0 * (rungs_pct[None, :] - half_pt)))
    return P

res_svamp = sim_phase0_oracle(svamp_curve)
res_code = sim_phase0_oracle(code_curve)
print("SVAMP simulation:", res_svamp)
print("Code simulation:", res_code)
```