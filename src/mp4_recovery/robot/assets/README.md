# Robot assets

The Mini Pupper v4 SKU2 model. `pupper.xml` is a generated MJCF (12 hinges, 4 knee tendons, 12 position actuators, IMU site,
`stand` keyframe, ghost copy `g_*`).

| File | What it is |
|---|---|
| `sku_2/pupper.xml` | The model |
| `sku_2/scene.xml` | `pupper.xml` plus floor and lights, for the viewer |
| `sku_2/servo_map.json` | Firmware servo angles to model joint angles |
| `sku_2/urdf/` | The repaired URDF `pupper.xml` was generated from |
| `sku_2/meshes/` | CAD meshes referenced by the URDF |
| `sku_2/parts/` | Split meshes referenced by `pupper.xml` |

`pupper.xml` carries a 360 g mass budget. `mp4_recovery.robot.sku2.get_spec` reads it and, without editing the file, removes
the ghost, keyframe, XML actuators and sensor; adds the IMU (`imu_ang_vel`, `imu_lin_vel`, `imu_lin_acc` on site `imu`); and
applies a mass file from `mp4_recovery/data/` (the walking and get-up tasks use `sku2_masses_478g.json`). The stand comes from
`data/sku2_contract.json`, not the keyframe. That file records the hashes of the `pupper.xml` and `servo_map.json` it was
built from.
