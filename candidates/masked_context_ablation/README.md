# Masked-context ablation candidate

This candidate is a controlled derivative of the scratch-trained pilot at Git
commit `3367a666bb77fc1122666a741cf3d6554fb7eb97`. It preserves the network,
masking, targets, optimizer, data loading, score assembly, resolution, and predictor
contract. Only the context inputs differ:

| `context_arm` | First feature input | Second feature input |
| --- | --- | --- |
| `local` | masked local average | masked local average |
| `global` | masked 4×4 image context | masked 4×4 image context |
| `combined` | masked local average | masked 4×4 image context |

All arms have the same 3,420 network parameters and four inference mask passes.
Use the same seed, 128-pixel input, 100 update cap, batch size four, 335 permitted
normal training images, and identical good validation split for each arm. The
checkpoint manifest records actual updates, exposures, losses, and tensor device;
comparison is incomplete if updates or exposures differ.

The testable juice_bottle pilot question is whether the combined arm improves
synthetic localization AP over local-only with no increase in normal-image or
artifact-control false alarms. Global-only probes whether broad context alone is
sufficient. No result from this one-category pilot establishes the original
five-category claim or official LOCO performance.

No pretrained weights, external source code, training-image memory bank, or
network access are required. Torch, NumPy, and Pillow are the installed source
dependencies. Run only through the guarded candidate runner after a separate
operator-authorized bounded campaign.
