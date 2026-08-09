# Phoenix native runtime

`phoenix_core` is the ROS-free, allocation-free deterministic layer of the deploy path. Python
still owns training, evaluation and every offline artifact; this owns the on-robot decision logic
and the policy evaluation.

Nothing here has run on a robot. See `docs/NATIVE_RUNTIME_PLAN.md` §9 for what is and is not
evidenced.

## What is in it

| Module | Responsibility |
|---|---|
| `types.hpp` | Fixed-size PODs, error codes, abort causes. Sizes pinned by `static_assert`. |
| `attitude` | Projected gravity and roll/pitch, one entry point per quaternion convention |
| `gate` | The safety precedence ladder as a pure function of `(snapshot, config, latched)` |
| `filters` | Terminal actuation filters: slew-rate cap, opt-in position clamp |
| `inference` | `InferenceEngine` interface and its ONNX Runtime backend |

There is deliberately **no normalizer**. Normalization is baked into the exported graph for some
checkpoints and absent from others, so a native normalizer driven by a config flag would
double-normalize one of them. The engine feeds raw observations; nothing here can double-apply what
it does not implement.

## Building

```bash
cmake -S runtime/phoenix_core -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/test_phoenix_core
```

C++17, not C++20: ROS 2 Humble's default, and `std::span` was the only C++20 feature the design
wanted. Built with `-Wall -Wextra -Wpedantic -Wshadow -Wconversion -Wsign-conversion` and
`-fno-fast-math` (reassociation and no-NaN assumptions would break both the parity gates and the
fail-closed semantics).

Sanitizers:

```bash
cmake -S runtime/phoenix_core -B build-asan -DCMAKE_BUILD_TYPE=Debug -DPHOENIX_CORE_SANITIZE=ON
cmake --build build-asan -j && ./build-asan/test_phoenix_core
```

## ONNX Runtime

Optional. Without it the deterministic core still builds and its tests still run; the inference
tests skip.

```bash
V=1.23.2
curl -L -o ort.tgz \
  https://github.com/microsoft/onnxruntime/releases/download/v$V/onnxruntime-linux-x64-$V.tgz
tar xzf ort.tgz -C ~/.local/opt/
ln -sfn ~/.local/opt/onnxruntime-linux-x64-$V ~/.local/opt/onnxruntime
```

CMake picks up `~/.local/opt/onnxruntime` automatically, or set `ONNXRUNTIME_ROOT`.

**The version is load-bearing, not incidental.** The inference parity gate asserts bit-exactness
between Python and C++, which holds only when both sides run the same ONNX Runtime version, the
same execution provider, one intra-op and one inter-op thread, and sequential execution. 1.23.2 is
pinned because that is the newest version with a PyPI wheel for this interpreter, and matching
downward is what makes the comparison meaningful. Raising the C++ version without raising the
Python one turns a real gate into a version-difference detector.

On the Jetson this must be an **aarch64** build, not the x86_64 one above.

## Parity harness

The Python deploy path is the oracle. `scripts/generate_parity_fixtures.py` replays it over
boundary-biased inputs and records every input and output as hex floats, so fixtures round-trip
bit-exactly rather than through lossy decimal. `test_parity.cpp` replays the same inputs through the
native code.

```bash
.venv/bin/python scripts/generate_parity_fixtures.py \
  runtime/phoenix_core/test/fixtures/parity_v1.txt
```

Tolerances are declared **before** the comparison runs, in the test header and in the plan, derived
from dtype and operation chain. Measured on this machine:

| Stage | Declared | Measured |
|---|---|---|
| Projected gravity | bit-exact | 0 mismatches / 507 |
| Roll / pitch | ≤ 2 ULP + zero ambiguous band | worst 2 ULP, 0 ambiguous |
| Slew clip | bit-exact | 0 mismatches / 500 |
| Gate decision | exact | 0 mismatches / 4,000 |
| ONNX action + latent | bit-exact | 0 / 118,800 elements |

Roll/pitch is the one stage that is not bit-exact, and the cause is not the port: numpy's `arctan2`
and glibc's `atan2` disagree by 1 ULP on identical double input. Since roll/pitch feed exactly one
decision, the harness additionally asserts that no fixture lies close enough to the attitude
threshold for that drift to change the verdict.

The gate parity test carries coverage assertions requiring every outcome and every reachable abort
cause to appear, so a fixture that only exercised the nominal path cannot pass silently.

## Not yet built

The ROS 2 adapter node, the observation builder, the shield port, `motor_crc`, and the benchmark
programs. No performance measurement has been taken and no performance claim is made anywhere in
this tree.
