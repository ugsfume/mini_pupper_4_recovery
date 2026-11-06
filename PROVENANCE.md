# Provenance

What the shipped files are and where they came from.

| File | What it is | sha256 |
|---|---|---|
| `policies/getup/policy.npz` | The get-up that ran on the robot: made with the kinds of steps in `scripts/recipes/getup.sh` on other seeds (31, clone 21) with slightly different step boundaries, last checkpoint at iteration 4546. The script doesn't reproduce it exactly | `8ef48590a678ef3228867a20d70360c4e89681a4d292655b03aa4c1ab410ff9c` |
| `policies/getup_sim/policy.npz` | The recipe with seed 41, last checkpoint at iteration 4546 | `305842eb8cf431205e286dd9384f384bb64a99bf384319b49413c76fa5748753` |
| `policies/walker/policy.npz` | Fixed weights: the `flat_omni` policy of [mini_pupper_4_sim](https://github.com/mangdangroboticsclub/mini_pupper_4_sim), with further training on rough ground and stairs, not included. Only its metadata differs from those weights: `scripts/recipes/walker.sh` writes the joint limits and the 12° stance | `db812009795f2aa82931a82a6639983d62b345d371595fee87af45c1109461e9` |
| `checkpoints/clone.pt` | Seed 41, after cloning | `6976bce8b121a97969e7a8499efe6ab4cd335b464d2103b54bd5d014cfd9fec3` |
| `checkpoints/wide_1499.pt` | Seed 41, end of the widened-limits step | `e885e99b0d82fad5d7f245db7ee0ea2f4549059f088938f93102da560e607328` |
| `checkpoints/smooth3_3947.pt` | Seed 41, before the last step | `cb49e659c650fb137279295bd8c31ba16f25aceba3ca5581f99cb3a7d643f97f` |

The robot model, the fitted servo parameters, the installer and the servo model code come from mini_pupper_4_sim; the
servo model builds on [BAM](https://github.com/Rhoban/bam). The joint limits were measured by hand on one robot.
