# Mission: Empirical Validation

A staged experimental protocol for testing whether affect propagation on causal
state graphs describes a real structure or redescribes a familiar one.

Target executor: Claude Code. Each mission is self contained, states what to
build, what to expect, and what result kills it.

---

## The rule that governs everything below

Every mission specifies the expected result **before** the data is collected,
and specifies what result would count as failure. If a mission cannot fail, it
is not in this document.

The theory under test makes one core claim: influence composes by
multiplication along a chain, not by addition. Everything else follows.
Multiplication has three fingerprints, and all three are testable:

1. **Geometric decay.** Influence falls by a constant percentage per step, not
   a constant amount.
2. **Annihilation by a single weak link.** One near zero gain destroys the
   whole upstream path regardless of how strong every other link is, and
   regardless of where in the chain it sits.
3. **A critical point.** Behaviour changes qualitatively at a threshold rather
   than smoothly across it.

Additive composition predicts none of these. That is the whole experiment.

---

## Mission 0: Instrumentation

**Objective.** Build the measurement apparatus and prove it is trustworthy
before it is used to test anything. Most failed experiments of this kind fail
here and never find out.

### Build

A chain generator with one non negotiable property: **each node is generated
from its parent only.** The seed must be removed from context after step one.
If the generator can see the seed at every step, there is a direct edge from
seed to every node, the graph is not a chain, and every number downstream is
uninterpretable.

```
generate_chain(seed, depth N):
    x_1 = generate(seed)
    for n in 2..N:
        x_n = generate(x_{n-1})        # x_{n-1} ONLY. Not the seed.
    return [x_1 ... x_N]
```

Node definition for this experiment, fixed by fiat: **one node is one short
narrative beat, two to three sentences, describing a state of affairs.** The
generation instruction is "write the next beat that follows causally from
this one." This is the definition that has been missing from the theory. Here
it is stipulated rather than discovered, which is legitimate for a
computational test and is not legitimate for a human one.

### The measurement, and the trap inside it

Influence of the seed at depth n is measured as embedding similarity between
the seed and node n. The trap is that unrelated texts do not sit at similarity
zero. Two random English paragraphs typically sit somewhere around 0.1 to 0.3
depending on the embedder. If you plot raw similarity against depth you will
see decay flattening toward that floor, and you will read it as "not
geometric, it plateaus." That conclusion would be an artifact of the
instrument, not a fact about the system.

So the floor must be measured and subtracted:

```
floor = mean over all i != j of cos(E(seed_i), E(node_n_j))
I(n)  = (cos(E(seed), E(x_n)) - floor) / (cos(E(seed), E(x_1)) - floor)
```

I(1) equals 1 by construction. Everything after that is a prediction.

### Run

Generate 60 chains from 60 distinct seeds, depth 20. Compute the floor from
all cross pairings. Hold out nothing yet.

### Expected result

The floor should come out clearly non zero. If your embedder puts unrelated
pairs near 0.15, that is normal and it is exactly why this mission exists.

**Instrument validation check.** Take node n from chain A and node n from
chain B and compute similarity. It should sit at the floor within noise and
show no trend with depth. If it trends, your embedder is picking up something
structural about depth itself, such as text getting longer or more abstract as
chains progress, and that artifact will contaminate every later mission.

### Failure mode

If the instrument check fails, stop. Do not proceed to Mission 1. Fix the
embedder, normalise node length, or change the measurement to a counterfactual
design (generate two chains from two seeds with matched machinery and measure
divergence rather than similarity).

---

## Mission 1: The pre-registered decay test

This is the core experiment. It is the Fisher move applied to your theory.

**Objective.** Determine whether influence decays geometrically at a rate that
can be predicted in advance from a single early measurement.

### Design

Split the 60 chains into a **calibration set of 15** and a **test set of 45**.
The split is made before any analysis.

**Step 1. Estimate g from the calibration set only, using depth 1 to 2 only.**

```
g_hat = mean over calibration chains of I(2) / I(1)
```

That single number is the entire model. Nothing else is fitted, ever.

**Step 2. Register the prediction before looking at the test set.**

Write to a file, hash it, commit it to git:

```
predicted I(n) = g_hat ** (n - 1),  for n = 3 .. 20
```

The commit hash is your timestamp. This is what makes the test honest, and it
is the only part of this protocol that cannot be added afterward.

**Step 3. Compute observed I(n) on the test set. Compare.**

### Expected result if the theory holds

Plot log I(n) against n. Multiplicative composition predicts a **straight
line** with slope ln(g_hat), where g_hat came from depths 1 and 2 alone.

Concretely, if g_hat comes out around 0.85, the registered predictions are:

| depth | predicted I(n) |
|---|---|
| 5 | 0.522 |
| 10 | 0.232 |
| 15 | 0.103 |
| 20 | 0.046 |

Observed values falling within the confidence band around these numbers, on
data used to fit nothing, is the result that matters.

### Expected result if the null holds

Three specific alternatives, each of which looks different on the plot:

- **Linear decay.** I(n) falls by a constant amount per step. On a log plot
  this curves downward sharply. Composition is additive and the core claim of
  the theory is wrong.
- **Plateau after early collapse.** I(n) drops for two or three steps then
  flattens well above the floor. This means the chain locks onto a topic and
  stops moving, which is a real property of generators and not a form of
  multiplicative decay.
- **Straight line at the wrong slope.** Decay is geometric but g_hat from
  depth 1 to 2 does not predict it. This is the most interesting failure and
  the most likely one. See below.

### The failure mode you should actually expect

**Gain drift.** There is no guarantee g is constant across depth. Early steps
may move further than late ones, because a chain that has settled into a
subject stops wandering. If g rises with depth, observed decay will be slower
than registered, the log plot will curve upward, and the registered prediction
will overshoot at depth 15 and 20.

This is not a floor artifact and Mission 0 will not have caught it. It is a
substantive finding, and it modifies the theory rather than killing it: g
becomes g(n), a function of depth, and the clean product Γ becomes a product
of unequal terms. The stability criterion survives. The tidy closed form
1/(1 − Γ) does not.

Report it as a result, not a defect. Distinguish it from a floor artifact by
checking whether the curve flattens toward the measured floor (artifact) or
toward some value above it (drift).

### Decision rule

State these before running:

- Registered predictions hold at depths 10, 15 and 20 within the band:
  multiplicative structure confirmed out of sample. This is the strong result.
- Geometric decay holds but at a rate not predicted by depth 1 to 2:
  multiplicativity is supported, constancy of g is refuted, theory is amended.
- Decay is linear or plateaus above the floor: the core claim is refuted for
  this system. Report it and stop.

---

## Mission 2: Manipulation checks

**Objective.** Prove that Mission 1 measured what it claims to. Two controls,
both of which should produce a known answer. If they do not, Mission 1 is void.

### Control A: seed visible

Regenerate the chains with the seed included in context at every step. This is
the confound Mission 0 was built to eliminate, so re-introducing it
deliberately tests whether the elimination worked.

**Expected.** Decay is markedly slower and flattens high, because a direct
seed to node edge exists at every depth and bypasses the chain entirely.

**If it looks the same as Mission 1**, your seed stripping did not work, or the
generator is carrying the seed implicitly through the parent text. Either way
Mission 1 is invalid until this is fixed.

### Control B: severed chain

Generate chains normally, then rebuild each chain by replacing node k with a
node taken from a different chain, and continue generation from that
substitute.

**Expected.** Seed influence at all depths beyond k collapses to the floor.
Not reduced. At the floor. A chain that has been cut carries nothing across the
cut.

**If influence survives past the cut**, there is leakage: something other than
the parent node is carrying seed information forward, likely the system prompt
or a stylistic signature. Find it before proceeding.

---

## Mission 3: Weak link annihilation

This is the sharpest test in the document. It tests the fingerprint that
additive composition cannot imitate, and its prediction is genuinely
counterintuitive.

**Objective.** Test whether one weak link destroys the whole upstream path
independent of where that link sits.

### Design

Build chains of fixed depth 12 where every link is strong, then force exactly
one link to be weak by instructing that generation step to make a large topical
jump. Vary the position k of the weak link across 2, 4, 6, 8 and 10. Twenty
chains per position.

Measure seed influence at the terminal node, I(12).

### Expected result if the theory holds

Under multiplication, I(12) is the product of all twelve gains. Multiplication
is commutative, so **the position of the weak link does not matter.**

```
I(12) = g_weak * g_strong^11     for every k
```

Plotted against k, this is a **flat line at a low value**. Every position
equally destructive. That is a strange prediction and it is exactly why it is
worth testing.

### Expected result if the null holds

Additive composition predicts that each link contributes independently, so one
damaged link costs a fixed share of the total and the rest survives. That gives
a **high line**, near the unbroken control, possibly with a mild slope if
later links matter more.

### Decision rule

The two predictions differ in level by a large margin, not by a subtle
statistical amount. Include an unbroken control chain set as the reference.

- Flat and low across all k: multiplicative annihilation confirmed. This is the
  strongest available evidence for the theory in this document.
- High, near the unbroken control: additive. Core claim refuted.
- Sloped: neither. Position matters, which means the composition is neither a
  clean product nor a clean sum, and the model needs a positional term it does
  not currently have.

### Known difficulty

Forcing a link to be weak by instruction is a blunt intervention and you should
verify it worked. Measure the parent to child similarity at the forced step and
confirm it is genuinely low relative to unforced steps. If the instruction did
not take, the experiment measures nothing.

---

## Mission 4: The critical point

**Objective.** Test whether the settle versus runaway boundary at Γ = 1 exists
as a sharp transition rather than a smooth change.

### Design

Move from chains to trees. Each node generates b children rather than one.
Sweep b across 1, 1.5, 2, 3 and 4 by varying how many continuations are
generated per node, and hold g roughly fixed by keeping the generation
instruction constant.

Count **surviving nodes**: nodes whose seed influence I is still above a
coherence threshold θ = 0.1.

### Expected result if the theory holds

Total surviving nodes should be finite and approximately 1/(1 − Γ) when
Γ = b·g < 1, and should grow without bound once Γ crosses 1. With g around
0.85, the boundary sits near b = 1.18, so the sweep should straddle it.

The signature is not that survival increases with b. That is trivially true.
The signature is that the increase is **not smooth**. Below the boundary the
tree exhausts itself and stops producing coherent nodes at a finite count.
Above it, coherent nodes keep appearing at every depth you are willing to
generate.

### Expected result if the null holds

Smooth monotone growth in surviving nodes with b, with no identifiable
threshold and no divergence. That means Γ = 1 is a stipulation you imposed
rather than a property of the system.

### Honest assessment of this mission

This is the weakest test in the document and you should know that going in.
Divergence is a statement about infinite processes and you will observe a
finite one, so you cannot observe divergence directly. You can only observe
whether the growth curve bends where the theory says it bends. Treat a
positive result here as suggestive and a negative result as inconclusive
rather than fatal.

---

## Mission 5: Verdict

Write the verdict before running anything, then fill it in.

| Mission | Result if theory holds | Result if null holds | Observed | Verdict |
|---|---|---|---|---|
| 0 | non zero floor, no depth trend | depth trend in cross pairs | | |
| 1 | log linear at registered slope | linear or plateau | | |
| 2A | slower decay, high plateau | identical to Mission 1 | | |
| 2B | collapse to floor past cut | influence survives cut | | |
| 3 | flat and low across k | high, near control | | |
| 4 | bend near b = 1/g | smooth, no threshold | | |

### What each outcome licences you to claim

- **Missions 1 and 3 both pass.** Multiplicative composition is confirmed in a
  system where you could measure it exactly. You may claim the mathematical
  core is validated computationally. You may not claim anything about human
  affect, because the node was defined by fiat here and cannot be in a person.
- **Mission 1 passes, Mission 3 fails.** Decay is geometric but annihilation
  does not hold. Something other than a pure product is operating. The theory
  needs a term it does not have.
- **Mission 1 fails on rate but holds on shape.** Amend to g(n) and rerun the
  registration with a depth varying model. This is a normal scientific outcome
  and not a defeat.
- **Mission 1 fails on shape.** The core claim is refuted for generators. That
  says nothing directly about people, but it removes the one system where the
  theory could be checked cheaply, and the honest move is to say so publicly.

---

## Appendix A: repository layout

```
mission/
  00_instrument/
    generate_chains.py       # chain generator, seed stripped after step 1
    measure_floor.py         # cross pairing baseline
    validate_instrument.py   # depth trend check on unrelated pairs
  01_decay/
    split.py                 # calibration 15 / test 45, fixed seed
    estimate_g.py            # g_hat from calibration depths 1-2 only
    REGISTERED.md            # predictions, committed before test analysis
    analyse.py               # observed vs registered
  02_controls/
    seed_visible.py
    severed_chain.py
  03_weaklink/
    forced_weak.py           # weak link at position k in 2,4,6,8,10
    verify_intervention.py   # confirm the forced link is actually weak
  04_branching/
    tree_generate.py
    count_survivors.py
  05_verdict/
    VERDICT.md
  data/
  figures/
```

## Appendix B: parameters, fixed in advance

| Parameter | Value | Reason |
|---|---|---|
| chains, Mission 1 | 60 | 15 calibration, 45 test |
| depth | 20 | far enough for g^n to separate models |
| node size | 2 to 3 sentences | fixed so length does not confound similarity |
| chains per position, Mission 3 | 20 | five positions, 100 chains total |
| coherence threshold θ | 0.10 | floor corrected |
| generation temperature | fixed across all missions | temperature changes g |
| embedder | one model, never swapped mid study | swapping changes the floor |

## Appendix C: the honest caveat to carry throughout

Mission 1's basic finding, that influence decays with depth, is close to
guaranteed and proves nothing on its own. Almost any iterative process decays.
The informative content lives entirely in three places: whether the **rate** is
predictable in advance from one early measurement, whether **annihilation** is
position independent, and whether the transition in Mission 4 is **sharp**.

If you report "influence decays with depth, theory confirmed," you will have
learned nothing and will believe you learned something. That is the failure
this document exists to prevent.
