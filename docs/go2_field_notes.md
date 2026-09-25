# GO2 field observations

This page records observations that affect Phoenix's interpretation of robot
telemetry. It does not serve as a connection or activation procedure. The
September 2026 H25 stage F1 attempt reached policy control for 0.56 seconds and
then latched a joint-limit fault. It did not establish a stable loaded stand.

## 1. DDS configuration is part of a run

The ROS 2 process must receive a valid `CYCLONEDDS_URI` that names an existing
configuration file. `scripts/harness_preflight.sh` checks that the file exists.
A topic list alone does not prove that the intended sensor publisher is fresh.

## 2. Topic discovery needs freshness checks

After a ROS environment change, discovery may still show old endpoints. Phoenix
checks timestamps and required first messages before it publishes policy
commands. Logs should record the topic source and age used for each verdict.

## 3. Odometry uses a boot-relative frame

The observed odometry position is a displacement from the robot's boot pose.
Its vertical coordinate is not an absolute simulator base height. Captures
using this source declare `odom_boot_relative`; replay refuses to treat them
as simulator world coordinates without an explicit transform. This affects
failure seeding and any reported base-height comparison.

## 4. Low-level control has an ownership boundary

A motors-off dry run validates message flow but does not prove control of the
robot. Live policy authority requires the stock controller to release the
motors and a separate deadman signal. The bridge records whether it was live
and what target was finally sent after safety processing.

## 5. Sensor geometry depends on the robot pose

A sensor transform measured from one posture should not be reused as a fixed
world transform during a different posture. Report the frame and posture used
for calibration when comparing real captures with simulation.

## 6. Wall-clock time can change independently of control time

The payload may start without a trustworthy wall clock. Freshness gates and
controller deadlines use monotonic time, while evidence records identify their
time base. A wall-clock correction must not make a stale sensor appear fresh.
