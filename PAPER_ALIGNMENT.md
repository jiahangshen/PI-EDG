# Paper–implementation alignment audit

The latest author-supplied manuscript describes the voxel PI-EDG method, not
the subsequent F3/S3 two-stage core-correction work. This release candidate
copies the actual method implementation from the research workspace. Items
below should be resolved in the manuscript or release notes before claiming
that the public code *exactly* reproduces the paper's numerical results.

| Item | Manuscript | Research implementation / recorded run |
| --- | --- | --- |
| Core architecture | 64³ PhysicsResUNet, hybrid FNO–CNN, 37 learned density experts | `model_softmoe_phys.py` base model has 50,661,546 parameters and 37 experts with `decade_exact`; this matches the intended model family |
| Training-time charge constraint | Disabled; neutral-electron scaling once after inference | Reviewer run passed `--constraint_start_epoch -1`, `--hard_routing_epoch -1`; paper release defaults retain both disabled |
| Learning rate / duration | Table 8: `1e-5`, 30 epochs | `reviewer_experiments/run_pi_variant.sh`: `1e-4`, `--stop_after_epochs 12` with a 30-epoch scheduler horizon. These are not the same protocol. |
| Router's guide gradient | Text says the guide is detached before routing | Code uses a gradient-scaled guide; recorded run sets `--routing_backbone_grad_scale 0.3`, not zero. The guide loss is computed from features detached from the backbone. |
| Guide loss domain | Equation (14) writes an unmasked full-grid guide loss | Code applies the guide MSE where target density exceeds `1e-10`. |
| Initial finite log lower bound | Text says `t0=-34` | Code initializes a `base_val=-35` with internal thresholds beginning at `-34`; interval sentinel and finite bounds therefore need a notation/code reconciliation. |
| Atom field / charge spike | Equations describe a `Z`-weighted Gaussian sum and a charge-valued spike | Code uses per-element Gaussian channels with max aggregation; `z_spike` is their channel sum, not the literal charge-weighted delta field written in the equation. |
| Raw output nonlinearity | Equation (13) writes the cube of the fused expert output | Code applies a sigmoid gate to the fused cube-root value before cubing, even when the training-time hard constraint is disabled. |
| Spatial input sampling | Manuscript describes voxel-aligned physical fields | Original code generates Gaussian atoms at voxel centers, while `PhysicsUtils.get_grid_coords` uses inclusive box endpoints for `v_ext`; later alignment experiments are **not** in this release. |
| Evaluation split | Paper calls the reported 2,000 molecules a test set | Research `canonical_2000.json` IDs are all a subset of the 2,076-ID manifest validation set. They are train-disjoint but not validation-independent. |
| Three seeds | Appendix D.5 says reported experiments are mean over three independent runs | The recorded reviewer `run_pi_variant.sh` invocation is seed 42; other seeds/results should be independently identified before making a three-seed reproduction claim. |

The reader and preprocessing logic in this release are a small independent
implementation of the single-field CUBE parsing interface used in the
workspace. On EDBench sample `19412`, its `train_ready_grid64` output was
checked against the existing research cache: target grid and centered atomic
coordinates matched exactly. This is a one-sample smoke check, not a full
dataset equivalence proof. The training model and voxelization source are
preserved; no later core correction is silently merged into them.
