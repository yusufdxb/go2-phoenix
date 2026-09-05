# Lab card: verify the Jetson runs the ONNX we shipped

**Scope: no robot motion.** Nothing in this card sends a command to the GO2.
The robot may stay powered down except for the payload computer. This card
answers exactly one question, and refuses to answer any other:

> When the policy node starts from the pinned config, does it open the ONNX file
> we intended, at the version we intended?

**Hardware activation is NOT verified today.** No step below has been executed on
the payload. What HAS been executed, on the workstation on 2026-09-05, is the
same staging/pin/manifest/verify chain against a local destination directory;
that proves the tooling works, and proves nothing about the Jetson.

---

## Why this check exists

Transfer is not activation. `scripts/stage_payload_bundle.sh` rsyncs a bundle and
re-checks `SHA256SUMS` on the payload, which proves the bytes arrived. It proves
nothing about which file the node opens: the deploy configs ship a
workstation-relative `policy.onnx_path`, and `ros2_policy_node.main` resolves it
with a bare `Path()` against the payload's current working directory. Started
from the payload's own repo checkout, that path lands in the payload's
`checkpoints/` tree, which holds an older `phoenix-stand-v3` era export. The
session then runs a policy nobody deployed while every transfer check reports
success.

`src/phoenix/sim2real/activation.py` closes that gap and travels inside the
bundle (stdlib + PyYAML only, no repo checkout and no PYTHONPATH needed on the
payload).

## Artifact under test

| item | value |
|---|---|
| checkpoint | `checkpoints/phoenix-stand-h25-lat-noise` (the deliverable, model_799) |
| deploy config | `configs/sim2real/deploy_stand_h25.yaml` |
| expected `policy.onnx` sha256 | `5331dbb7c4b8194e7f2238c14b985ea16af2f47aa7130e04bade3a0f9e9289eb` |
| expected `policy.onnx.data` sha256 | `cda0e9b4847c4208ec1aadd447c1851856b9d6f2e38dbadfaae1c23e214dee92` |

The export is external-data format: `policy.onnx` is the graph, `policy.onnx.data`
carries every weight, and onnxruntime resolves the sidecar relative to the
`.onnx` path. Both must be present in the same directory or the load fails.

---

## The four steps

### 1. Stage the bundle to the payload

From the repo root on the workstation, with the payload reachable
(`jetson` = 192.168.0.70 wifi, `jetson-cable` = 192.168.123.18; both in
`~/.ssh/config`):

```bash
scripts/stage_payload_bundle.sh \
    checkpoints/phoenix-stand-h25-lat-noise \
    configs/sim2real/deploy_stand_h25.yaml \
    jetson:/home/unitree/phoenix/stand-h25-lat-noise
```

The script refuses to assemble the bundle at all unless
`checkpoints/phoenix-stand-h25-lat-noise/parity_gate.json` records
`"passed": true`, so an ungated policy cannot reach the robot by this route.

### 2. Activate: pin the checkpoint

Nothing extra to run. Step 1 calls `activation.py pin` **before** it writes
`SHA256SUMS`, because pinning rewrites the config and a manifest taken first
would be invalidated by the very step that activates the bundle. Pin rewrites
the bundle's own copy of `deploy_stand_h25.yaml` so every path the node opens is
absolute and inside `/home/unitree/phoenix/stand-h25-lat-noise`. It fails closed
if a path key it must rewrite has no corresponding file in the bundle.

Expect these two lines in step 1's output:

```
[activate] pinned onnx_path -> /home/unitree/phoenix/stand-h25-lat-noise/policy.onnx
[activate] pinned torchscript_path -> /home/unitree/phoenix/stand-h25-lat-noise/policy.pt
```

### 3. Verify the manifest and the activation, ON THE PAYLOAD

Step 1 already runs both remotely and stops if either fails. Re-run them by hand
to confirm state at the start of the session, since the bundle may have been
staged on a previous day:

```bash
ssh jetson
cd /home/unitree/phoenix/stand-h25-lat-noise
sha256sum -c SHA256SUMS
python3 activation.py verify --bundle /home/unitree/phoenix/stand-h25-lat-noise
```

`sha256sum -c` proves the bytes on the payload are the bytes we built.
`activation.py verify` re-reads the config **the way the node does**, resolves
every path it will open, and hashes each against `SHA256SUMS`. It exits non-zero
on any mismatch, any relative path, and any path escaping the bundle.

PASS looks like:

```
[activate] policy.onnx /home/unitree/phoenix/stand-h25-lat-noise/policy.onnx
[activate] sha256      5331dbb7c4b8194e7f2238c14b985ea16af2f47aa7130e04bade3a0f9e9289eb
[activate] ACTIVATION VERIFIED. Bring up with, verbatim:
    python3 -m phoenix.sim2real.ros2_policy_node --config /home/unitree/phoenix/stand-h25-lat-noise/deploy_stand_h25.yaml
```

The printed sha256 must equal the expected value in the table above. If it does
not, the payload is holding a different export; **stop and re-stage**, do not
proceed.

### 4. Start the node from the pinned config and read back what it loaded

Run the bring-up line `verify` printed, verbatim. Use the pinned in-bundle config
path; do NOT pass `--onnx`, because the whole point is to test what the config
resolution does on its own.

```bash
python3 -m phoenix.sim2real.ros2_policy_node \
    --config /home/unitree/phoenix/stand-h25-lat-noise/deploy_stand_h25.yaml
```

Then confirm, from outside the node, that the running process has the intended
file open:

```bash
PID=$(pgrep -f ros2_policy_node)
ls -l /proc/$PID/fd | grep -i onnx        # or, if the fd is already closed:
grep -i onnx /proc/$PID/maps | head
sha256sum /home/unitree/phoenix/stand-h25-lat-noise/policy.onnx
```

**This step is the only one that observes the runtime rather than the filesystem.**
The first three prove the bundle is correct and self-consistent; step 4 proves
the process actually reached into it.

---

## Pass criteria

All four must hold. Any one failing fails the check.

1. `sha256sum -c SHA256SUMS` reports OK for every file on the payload.
2. `activation.py verify` exits 0 and prints the expected `policy.onnx` sha256
   above, character for character.
3. The node starts from the pinned config with no `--onnx` override and does not
   raise on ONNX load.
4. The running process resolves the ONNX to a path under
   `/home/unitree/phoenix/stand-h25-lat-noise/`, not under any `checkpoints/` tree.

## Abort conditions

- `verify` prints a sha256 that differs from the table: the payload holds a
  different export. Re-stage; do not "fix" the config by hand.
- `verify` reports a relative or escaping path: the pin did not take. Re-run
  step 1 rather than editing the YAML on the payload.
- The node loads an ONNX from anywhere other than the bundle directory: that is
  exactly the bug this card exists to detect. Record the resolved path and stop.

## Explicitly out of scope

- Any `/cmd_vel`, `lowcmd` or motion command. The robot does not move.
- Any `motion_switcher` call. Do not select a mode.
- Any claim about stand quality, slew percentage, or transfer performance. Those
  need the separate feet-on-ground stand test and are not touched here.
- Any claim that hardware activation "works" before this card has actually been
  executed on the payload and its output recorded.

## Record on completion

Append to the day's vault log: the four command outputs verbatim, the sha256
`verify` printed, and the resolved ONNX path from step 4. Until that exists,
Phoenix Jetson activation stays **UNVERIFIED** in every document that mentions it.
