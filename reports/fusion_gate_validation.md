# Lightweight Fusion Gate Validation

## Purpose

These checks isolate likely causes of the near-zero difference between `M0-FT`,
the parameter-matched control `C`, and fusion `M1`. They are architectural and
optimization diagnostics, not VOC accuracy results.

## Runtime and inputs

- CPU runtime
- PyTorch SSD300 implementation from repository `main`
- released VOC SSD300 checkpoint
- deterministic seed 11
- one synthetic `3 x 300 x 300` Gaussian input unless stated otherwise
- synthetic SSD target boxes for gradient checks

## Results

### Zero-gate identity

The maximum absolute difference between baseline and fusion training outputs
was exactly `0.0` at initialization.

### Feature-scale mismatch

With pretrained backbone weights and a randomly initialized fusion branch:

- `conv4_3` RMS after L2 normalization: `0.585718`
- fusion branch delta RMS at gate `1.0`: `0.026990`
- branch / shallow RMS ratio: `0.04608`
- gate value needed to match shallow RMS: about `21.70`

The same random-backbone stress check produced a branch / shallow RMS ratio of
`0.001598`, requiring a gate of about `626` to match the shallow RMS.

This means the global scalar gate does not start in a numerically comparable
range for the two feature streams. A gate near the usual `0.001-0.1` scale
leaves the branch much weaker than the normalized conv4_3 feature.

### First-step gradient behavior

At gate zero, the gradient with respect to fusion branch weights was exactly
zero:

```text
fusion.project.weight   0.0
fusion.depthwise.weight 0.0
fusion.pointwise.weight 0.0
```

Only the gate receives a gradient on the first optimizer step. In a
multi-target pretrained-backbone check, the first gate gradient was
`6.97e-05`, giving a scale change of only `6.97e-08` with LR `1e-3`.

After a synthetic gate step, the branch gradients become nonzero, but their
magnitude is scaled by the still-small gate. This is a slow-start mechanism,
not a zero-gradient bug.

### Control FLOPs mismatch

The proposal described matching both parameter count and FLOPs. The current
control branch matches parameters but not compute:

- M1 branch: about `574M` MACs
- C branch: about `1,143M` MACs
- C / M1 ratio: `1.99`

The control has roughly twice the branch computation because its wider
bottleneck operates at `38 x 38`, whereas M1 projects `conv7` at `19 x 19`
before upsampling.

### Learning-rate budget

The head and new branch receive:

- `3,600` updates at LR `1e-3`
- `1,500` updates at LR `1e-4`
- `600` updates at LR `1e-5`
- `300` updates at LR `1e-6`

Only the first 3,600 updates provide strong adaptation. The full backbone is
unfrozen at iteration 2,400, so only 1,200 strong-LR updates occur with an
unfrozen backbone.

## Interpretation

The most likely primary cause is the interaction between the zero-sized
scalar gate and the branch's feature scale. The gate starts at zero, the
branch parameters initially receive no gradient, and even a gate value of one
does not make the branch numerically comparable to the L2-normalized
`conv4_3` map. Under only 6,000 warm-started fine-tuning iterations, the
branch therefore has little opportunity to change the already strong detector.

The much smaller proposal budget relative to FSSD/Feature-Fused SSD training,
the single seed, and the small size of the observed metric differences are
secondary explanations or measurement limitations. They are not needed to
explain why this particular gate remains effectively inactive.

## Recommended next validation

Before another full 6k comparison, run one short GPU run with:

1. normalized fusion-branch features,
2. a small nonzero gate such as `0.01`,
3. per-100-iteration logging of gate magnitude and branch/shallow RMS ratio.

That run should determine whether the fusion path becomes materially active.
Only after that check is it worth paying for another full 6k accuracy run.
