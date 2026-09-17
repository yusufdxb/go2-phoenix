#!/usr/bin/env bash
# Phoenix loop closure, Gate 9 intermediate.
#
# What this runs, in order:
#   1. Stage the training parquet into a fresh curriculum pool
#   2. Replay it with Halton variations, DRIVEN BY THE BASELINE POLICY, and
#      stage the variant trajectories the replay produced. Zero variants is a
#      hard error, not a silent pass
#   3. Fine-tune the baseline once per seed, each with its OWN training seed and
#      its own curriculum RNG seed, both read back from the artifact the run wrote
#   4. Evaluate the baseline and every adapted policy at one fixed evaluation
#      seed, recorded separately from the training seeds
#   5. Held-out arm: replay the HELD-OUT trajectory under every policy, with a
#      DISJOINT Halton seed, and report the failure rate on it
#   6. Report
#
# Held-out arm (new 2026-09-17). Until phoenix.replay.reconstruct could drive a
# policy and emit per-variant trajectories, nothing here could seed an
# evaluation rollout from a recorded trajectory, so the held-out parquet was
# used for exactly one purpose: proving it never entered the training pool.
# That is now stage 5, and it uses --variation-seed to keep the held-out
# perturbation points disjoint from the ones training saw. Reusing the training
# variation seed would make the "held-out" points identical to the pool's.
#
# What is still NOT evidence here: this whole script is sim-only. A pass is a
# reason to book hardware time, never a substitute for it. One captured
# intervention also means the Halton perturbations are correlated copies of ONE
# state, so a gain can be specific to that seed family; see
# vault AUDIT_2026-09-17_hardware-testability.md.
#
# Usage:
#   ./scripts/loop_closure.sh <TRAINING_PARQUET> <HELDOUT_PARQUET> [options]
#
# Options:
#   --seeds "42 43 44"   training seeds, one fine-tune run each
#   --eval-seed N        evaluation seed, shared by every policy (default 20260911)
#   --baseline PATH      baseline checkpoint to adapt (default: the locked H25 stand)
#   --env-config PATH    env config for replay and evaluation (default: H25 stand)
#   --heldout-variation-seed N
#                        Halton seed for the held-out arm. MUST differ from the
#                        training variations config seed (default 20260917)
#   --skip-heldout       skip stage 5 and say so in the report
#   --allow-unaugmented  proceed when the replay stage produced no variant
#                        trajectories. Records pool_augmented=false in the
#                        report. Without this the run stops there.
#
# Example:
#   ./scripts/loop_closure.sh \
#       data/failures/pqa_push_lat_2026-04-22_15-23-01.parquet \
#       data/failures/pqa_slip_2026-04-22_16-10-44.parquet
#
# Total wall time: ~60-80 min on NVIDIA (Blackwell) consumer GPU (replay ~15 min + 3x fine-tune
# ~36 min + 4x eval ~15 min).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source "$REPO_ROOT/scripts/_activate.sh"

die() {
    echo "[loop_closure] ERROR: $*" >&2
    exit 1
}

# --- arguments -------------------------------------------------------------
SEEDS=(42 43 44)
EVAL_SEED=20260911
# The curriculum RNG seed is a SEPARATE quantity from the training seed. It is
# derived by a fixed offset so the two are never the same integer and each is
# recorded under its own name.
CURRICULUM_SEED_OFFSET=1000
ALLOW_UNAUGMENTED=0
# The locked H25 stand deliverable. This script used to hardcode the walking
# v3b checkpoint and configs/env/flat.yaml, so it adapted and evaluated a
# policy that is NOT the one the hardware gates run.
BASELINE_CKPT="checkpoints/phoenix-stand-h25-lat-noise/2026-06-22_21-08-20/model_799.pt"
ENV_CONFIG="configs/env/stand_v3_h25.yaml"
HELDOUT_VARIATION_SEED=20260917
SKIP_HELDOUT=0
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seeds)
            [[ $# -ge 2 ]] || die "--seeds needs a value"
            read -r -a SEEDS <<<"$2"
            shift 2
            ;;
        --eval-seed)
            [[ $# -ge 2 ]] || die "--eval-seed needs a value"
            EVAL_SEED="$2"
            shift 2
            ;;
        --baseline)
            [[ $# -ge 2 ]] || die "--baseline needs a value"
            BASELINE_CKPT="$2"
            shift 2
            ;;
        --env-config)
            [[ $# -ge 2 ]] || die "--env-config needs a value"
            ENV_CONFIG="$2"
            shift 2
            ;;
        --heldout-variation-seed)
            [[ $# -ge 2 ]] || die "--heldout-variation-seed needs a value"
            HELDOUT_VARIATION_SEED="$2"
            shift 2
            ;;
        --skip-heldout)
            SKIP_HELDOUT=1
            shift
            ;;
        --allow-unaugmented)
            ALLOW_UNAUGMENTED=1
            shift
            ;;
        -*)
            die "unknown option: $1"
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

if [[ ${#POSITIONAL[@]} -ne 2 ]]; then
    echo "[loop_closure] usage: $0 <TRAINING_PARQUET> <HELDOUT_PARQUET> [options]" >&2
    exit 2
fi
TRAIN_PARQUET="${POSITIONAL[0]}"
HELDOUT_PARQUET="${POSITIONAL[1]}"

for p in "$TRAIN_PARQUET" "$HELDOUT_PARQUET"; do
    [[ -f "$p" ]] || die "parquet not found: $p"
done

if [[ "$(realpath "$TRAIN_PARQUET")" == "$(realpath "$HELDOUT_PARQUET")" ]]; then
    die "training and held-out parquets are the same file; the held-out
                scenario MUST differ from training."
fi
TRAIN_SHA=$(sha256sum "$TRAIN_PARQUET" | cut -d' ' -f1)
HELDOUT_SHA=$(sha256sum "$HELDOUT_PARQUET" | cut -d' ' -f1)
if [[ "$TRAIN_SHA" == "$HELDOUT_SHA" ]]; then
    die "training and held-out parquets have identical contents ($TRAIN_SHA);
                two paths to the same data is not a held-out set."
fi

# Every run must have a DISTINCT intended training seed, or the three runs are
# one run reported three times. Check before spending any GPU time.
if [[ ${#SEEDS[@]} -eq 0 ]]; then
    die "--seeds produced an empty list"
fi
UNIQUE_SEEDS=$(printf '%s\n' "${SEEDS[@]}" | sort -u | wc -l)
if [[ "$UNIQUE_SEEDS" -ne "${#SEEDS[@]}" ]]; then
    die "seed list ${SEEDS[*]} is not distinct; independent runs need distinct seeds"
fi

[[ -f "$BASELINE_CKPT" ]] || die "baseline checkpoint missing at $BASELINE_CKPT"
[[ -f "$ENV_CONFIG" ]] || die "env config missing at $ENV_CONFIG"

VARIATIONS_CONFIG="configs/replay/variations.yaml"
[[ -f "$VARIATIONS_CONFIG" ]] || die "variations config missing at $VARIATIONS_CONFIG"

# The held-out arm is only held out if its perturbation points differ from the
# ones the training pool was built with. Same seed, same Halton points, and the
# "held-out" arm is a rerun of training under a different trajectory label.
TRAIN_VARIATION_SEED=$(PYTHONPATH="$REPO_ROOT/src" python3 -c \
    'import sys,yaml; print(int(yaml.safe_load(open(sys.argv[1]))["variations"]["seed"]))' \
    "$VARIATIONS_CONFIG") || die "could not read the variations seed from $VARIATIONS_CONFIG"
if [[ "$SKIP_HELDOUT" -ne 1 && "$HELDOUT_VARIATION_SEED" == "$TRAIN_VARIATION_SEED" ]]; then
    die "--heldout-variation-seed ($HELDOUT_VARIATION_SEED) equals the training
                variation seed from $VARIATIONS_CONFIG. The held-out arm would
                draw the SAME Halton points training saw and would not be held
                out in any sense. Pick a different seed."
fi

# --- fine_tune seed contract ----------------------------------------------
# The previous version of this script looped over three seeds, labelled three
# output directories, and never passed the seed to fine_tune. All three runs
# used cfg["run"]["seed"], so the "3 independent seeds" were the same run three
# times. Refuse to run until fine_tune can actually take the seeds.
FINE_TUNE_HELP=$(PYTHONPATH="$REPO_ROOT/src" python3 -m phoenix.adaptation.fine_tune --help 2>&1) \
    || die "could not read 'phoenix.adaptation.fine_tune --help':
$FINE_TUNE_HELP"
MISSING_FLAGS=()
for flag in --seed --curriculum-seed; do
    grep -q -- "$flag" <<<"$FINE_TUNE_HELP" || MISSING_FLAGS+=("$flag")
done
if [[ ${#MISSING_FLAGS[@]} -gt 0 ]]; then
    die "phoenix.adaptation.fine_tune does not accept ${MISSING_FLAGS[*]}.
                Multi-seed loop closure is impossible without it: every run
                would fall back to cfg['run']['seed'] and the seeds below would
                be labels on identical runs.
                Required contract:
                  --seed INT             training seed; sets env_cfg.seed and
                                         cfg['run']['seed'], overriding the YAML
                  --curriculum-seed INT  FailureCurriculum RNG seed; must default
                                         to --seed, never silently to 0
                  and the run must write <run_dir>/seeds.json holding
                  {training_seed, curriculum_seed, config_seed, resolved_from}
                  so the seed can be verified from the artifact."
fi

TS=$(date +%Y-%m-%d_%H-%M-%S)
TRAIN_DIR="data/failures/loop_closure_train_${TS}"
REPLAY_DIR="data/failures/loop_closure_replay_${TS}"
OUT_DIR="docs/loop_closure_${TS}"
POOL_LINK="data/failures/loop_closure_train"
mkdir -p "$TRAIN_DIR" "$REPLAY_DIR" "$OUT_DIR"
# The symlink below is process state, not an artifact. Remove it on ANY exit,
# including a failure, so a later run cannot pick up a stale pool.
trap 'rm -f "$POOL_LINK"' EXIT

echo "[loop_closure] =============================="
echo "[loop_closure] training parquet : $TRAIN_PARQUET"
echo "[loop_closure] held-out parquet : $HELDOUT_PARQUET (NOT evaluated, see report)"
echo "[loop_closure] baseline policy  : $BASELINE_CKPT"
echo "[loop_closure] env config       : $ENV_CONFIG"
echo "[loop_closure] training pool    : $TRAIN_DIR"
echo "[loop_closure] replay variants  : $REPLAY_DIR"
echo "[loop_closure] training seeds   : ${SEEDS[*]}"
echo "[loop_closure] evaluation seed  : $EVAL_SEED"
echo "[loop_closure] outputs          : $OUT_DIR"
echo "[loop_closure] =============================="

# --- Stage 1: populate training pool ---------------------------------
cp -f "$TRAIN_PARQUET" "$TRAIN_DIR/"

# --- Stage 2: replay with Halton variations --------------------------
echo "[loop_closure] stage 2/5: replay with variations..."
PYTHONPATH="$REPO_ROOT/src" python3 -m phoenix.replay.reconstruct \
    --trajectory "$TRAIN_PARQUET" \
    --variations-config "$VARIATIONS_CONFIG" \
    --env-config "$ENV_CONFIG" \
    --policy "$BASELINE_CKPT" \
    --output-dir "$REPLAY_DIR" \
    --headless \
    2>&1 | tee "$OUT_DIR/replay.log"

# Count what the replay actually produced BEFORE copying, so the number cannot
# be confused with the original trajectory staged in stage 1. The old version
# counted the pool after the copy, where 1 (the original, no variants at all)
# read as success.
NUM_VARIANTS=$(find "$REPLAY_DIR" -name "*.parquet" | wc -l)
if [[ "$NUM_VARIANTS" -eq 0 ]]; then
    if [[ "$ALLOW_UNAUGMENTED" -ne 1 ]]; then
        die "the replay stage produced 0 variant trajectories in $REPLAY_DIR.
                phoenix.replay.reconstruct writes replay_summary.json and render
                output; it does NOT currently emit Parquet variant trajectories,
                so there is nothing to augment the curriculum pool with and the
                'pool of 1 real + N variants' claim would be false.
                Since 2026-09-17 reconstruct DOES emit them, so zero here means
                the replay actually failed: read $OUT_DIR/replay.log. It exits 2
                when the trajectory is a hardware capture (boot-relative
                odometry is not a valid simulator seed). Re-run with
                --allow-unaugmented only to train on the single real trajectory
                and have the report say so."
    fi
    echo "[loop_closure] WARN: 0 variant trajectories; continuing unaugmented on request" >&2
    POOL_AUGMENTED=false
else
    find "$REPLAY_DIR" -name "*.parquet" -exec cp -f {} "$TRAIN_DIR/" \;
    POOL_AUGMENTED=true
fi
NUM_TRAIN=$(find "$TRAIN_DIR" -name "*.parquet" | wc -l)
EXPECTED_TRAIN=$((NUM_VARIANTS + 1))
if [[ "$NUM_TRAIN" -ne "$EXPECTED_TRAIN" ]]; then
    die "training pool holds $NUM_TRAIN parquets, expected $EXPECTED_TRAIN
                (1 real + $NUM_VARIANTS variants); staging did not do what it says."
fi
echo "[loop_closure] training pool: $NUM_TRAIN parquets (1 real + $NUM_VARIANTS variants)"

# The held-out trajectory must not be in the pool, by content and not by name.
while IFS= read -r staged; do
    if [[ "$(sha256sum "$staged" | cut -d' ' -f1)" == "$HELDOUT_SHA" ]]; then
        die "held-out trajectory content found in the training pool at $staged"
    fi
done < <(find "$TRAIN_DIR" -name "*.parquet")

# --- Stage 3: multi-seed adaptation fine-tune ------------------------
# adaptation.yaml hardcodes the un-timestamped pool path, so point it at this
# run's directory with a symlink (removed by the EXIT trap above). If that path
# is a real directory, `ln -sfn` would quietly create the link INSIDE it and the
# curriculum would read a stale pool, so refuse instead.
if [[ -e "$POOL_LINK" && ! -L "$POOL_LINK" ]]; then
    die "$POOL_LINK exists and is not a symlink; refusing to stage the pool
                into it. Move or delete it first."
fi
ln -sfn "$(basename "$TRAIN_DIR")" "$POOL_LINK"

RUN_DIRS=()
ACTUAL_TRAINING_SEEDS=()
ACTUAL_CURRICULUM_SEEDS=()
for seed in "${SEEDS[@]}"; do
    curriculum_seed=$((seed + CURRICULUM_SEED_OFFSET))
    echo "[loop_closure] stage 3/5: fine-tune training_seed=$seed curriculum_seed=$curriculum_seed"
    BEFORE=$(mktemp)
    ls -d checkpoints/phoenix-adapt-loop-closure/20* 2>/dev/null | sort >"$BEFORE" || true
    PYTHONPATH="$REPO_ROOT/src" python3 -m phoenix.adaptation.fine_tune \
        --config configs/train/adaptation_loop_closure.yaml \
        --resume "$BASELINE_CKPT" \
        --trajectory-dir "$POOL_LINK" \
        --num-envs 10240 \
        --seed "$seed" \
        --curriculum-seed "$curriculum_seed" \
        --headless \
        2>&1 | tee "$OUT_DIR/adapt_seed${seed}.log"
    AFTER=$(mktemp)
    ls -d checkpoints/phoenix-adapt-loop-closure/20* 2>/dev/null | sort >"$AFTER" || true
    NEW_RUN=$(comm -13 "$BEFORE" "$AFTER")
    rm -f "$BEFORE" "$AFTER"
    NEW_COUNT=$(grep -c . <<<"$NEW_RUN" || true)
    if [[ "$NEW_COUNT" -ne 1 ]]; then
        die "fine-tune for seed=$seed produced $NEW_COUNT new run
                directories under checkpoints/phoenix-adapt-loop-closure/,
                expected exactly 1. Refusing to guess which checkpoint is this
                seed's."
    fi
    RUN_DIR="${NEW_RUN}_seed${seed}"
    mv "$NEW_RUN" "$RUN_DIR"

    # Verify the seed from the artifact the run wrote, not from our own intent.
    SEEDS_JSON="$RUN_DIR/seeds.json"
    [[ -f "$SEEDS_JSON" ]] || die "no $SEEDS_JSON; cannot verify which seed this
                run actually used. fine_tune must record
                {training_seed, curriculum_seed, config_seed, resolved_from}."
    ACTUAL_TRAIN_SEED=$(python3 -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["training_seed"])' "$SEEDS_JSON")
    ACTUAL_CURR_SEED=$(python3 -c \
        'import json,sys; print(json.load(open(sys.argv[1]))["curriculum_seed"])' "$SEEDS_JSON")
    [[ "$ACTUAL_TRAIN_SEED" == "$seed" ]] || die "run $RUN_DIR recorded
                training_seed=$ACTUAL_TRAIN_SEED, asked for $seed"
    [[ "$ACTUAL_CURR_SEED" == "$curriculum_seed" ]] || die "run $RUN_DIR recorded
                curriculum_seed=$ACTUAL_CURR_SEED, asked for $curriculum_seed"
    RUN_DIRS+=("$RUN_DIR")
    ACTUAL_TRAINING_SEEDS+=("$ACTUAL_TRAIN_SEED")
    ACTUAL_CURRICULUM_SEEDS+=("$ACTUAL_CURR_SEED")
done

# Independence assertion, from the recorded artifacts rather than the loop
# variable: N runs must carry N distinct training seeds and N distinct
# curriculum seeds.
for label in training curriculum; do
    if [[ "$label" == training ]]; then
        values=("${ACTUAL_TRAINING_SEEDS[@]}")
    else
        values=("${ACTUAL_CURRICULUM_SEEDS[@]}")
    fi
    distinct=$(printf '%s\n' "${values[@]}" | sort -u | wc -l)
    if [[ "$distinct" -ne "${#values[@]}" ]]; then
        die "the ${#values[@]} runs recorded only $distinct distinct $label seeds
                (${values[*]}); they are not independent runs."
    fi
done

# --- Stage 4: eval baseline + adapted --------------------------------
echo "[loop_closure] stage 4/5: evaluations (evaluation seed $EVAL_SEED, shared by all)"

eval_policy() {
    local ckpt="$1"
    local label="$2"
    local training_seed="${3:-}"
    [[ -f "$ckpt" ]] || die "checkpoint for $label not found: ${ckpt:-<empty>}"
    echo "[loop_closure]   eval: $label ($ckpt)"
    local extra=()
    if [[ -n "$training_seed" ]]; then
        extra=(--training-seed "$training_seed")
    fi
    PYTHONPATH="$REPO_ROOT/src" python3 -m phoenix.training.evaluate \
        --checkpoint "$ckpt" \
        --env-config "$ENV_CONFIG" \
        --num-envs 16 \
        --num-episodes 32 \
        --seed "$EVAL_SEED" \
        "${extra[@]}" \
        --metrics-out "$OUT_DIR/eval_${label}.json" \
        2>&1 | tail -2
}

eval_policy "$BASELINE_CKPT" "baseline"

LABELS=(baseline)
ADAPTED_CKPTS=()
for i in "${!SEEDS[@]}"; do
    seed="${SEEDS[$i]}"
    RUN_DIR="${RUN_DIRS[$i]}"
    CKPT="${RUN_DIR}/model_499.pt"
    if [[ ! -f "$CKPT" ]]; then
        CKPT=$(ls "$RUN_DIR"/model_*.pt 2>/dev/null | sort -V | tail -1 || true)
    fi
    eval_policy "$CKPT" "adapted_seed${seed}" "$seed"
    LABELS+=("adapted_seed${seed}")
    ADAPTED_CKPTS+=("$CKPT")
done

# --- Stage 5: held-out failure-seeded arm ----------------------------
# Seed a rollout from the HELD-OUT trajectory under every policy and measure
# how often the perturbed replays end in a detected failure. The Halton seed is
# deliberately different from the training one, so these perturbation points
# are disjoint from the ones the curriculum trained on.
HELDOUT_DIR="$OUT_DIR/heldout"
HELDOUT_RAN=false
declare -A HELDOUT_FAIL_RATE
declare -A HELDOUT_COUNTS
if [[ "$SKIP_HELDOUT" -eq 1 ]]; then
    echo "[loop_closure] stage 5/6: held-out arm SKIPPED on request"
else
    echo "[loop_closure] stage 5/6: held-out arm (variation seed $HELDOUT_VARIATION_SEED)"
    mkdir -p "$HELDOUT_DIR"
    HELDOUT_POLICIES=("$BASELINE_CKPT" "${ADAPTED_CKPTS[@]}")
    HELDOUT_RAN=true
    for i in "${!LABELS[@]}"; do
        label="${LABELS[$i]}"
        ckpt="${HELDOUT_POLICIES[$i]}"
        dest="$HELDOUT_DIR/$label"
        echo "[loop_closure]   held-out: $label"
        if ! PYTHONPATH="$REPO_ROOT/src" python3 -m phoenix.replay.reconstruct \
            --trajectory "$HELDOUT_PARQUET" \
            --variations-config "$VARIATIONS_CONFIG" \
            --env-config "$ENV_CONFIG" \
            --policy "$ckpt" \
            --variation-seed "$HELDOUT_VARIATION_SEED" \
            --output-dir "$dest" \
            --headless \
            >"$OUT_DIR/heldout_${label}.log" 2>&1; then
            die "held-out replay failed for $label; see $OUT_DIR/heldout_${label}.log.
                Exit 2 means the held-out parquet is a hardware capture, which is
                not a valid simulator seed. Re-run with --skip-heldout to proceed
                without this arm and have the report say so."
        fi
        INDEX="$dest/variants_index.json"
        [[ -f "$INDEX" ]] || die "held-out replay for $label wrote no $INDEX"
        read -r n_written n_failed <<<"$(python3 -c "
import json,sys
d = json.load(open(sys.argv[1]))
print(d['variants_written'], d['variants_with_failure'])" "$INDEX")"
        [[ "$n_written" -gt 0 ]] || die "held-out replay for $label produced 0 variants;
                there is nothing to measure. See $OUT_DIR/heldout_${label}.log."
        HELDOUT_COUNTS["$label"]="$n_failed/$n_written"
        HELDOUT_FAIL_RATE["$label"]=$(python3 -c \
            "print(f'{$n_failed / $n_written:.4f}')")
        echo "[loop_closure]     $label: ${HELDOUT_COUNTS[$label]} variants failed"
    done
fi

# --- Stage 6: report -------------------------------------------------
echo "[loop_closure] stage 6/6: report"
REPORT="$OUT_DIR/report.md"
SEED_MANIFEST="$OUT_DIR/seeds.json"
SEED_PAIRS=()
for i in "${!SEEDS[@]}"; do
    SEED_PAIRS+=("${ACTUAL_TRAINING_SEEDS[$i]}:${ACTUAL_CURRICULUM_SEEDS[$i]}")
done
python3 - "$SEED_MANIFEST" "$EVAL_SEED" "${SEED_PAIRS[@]}" <<'PYEOF'
import json
import sys

path, eval_seed, *pairs = sys.argv[1:]
runs = []
for pair in pairs:
    training, curriculum = pair.split(":")
    runs.append({"training_seed": int(training), "curriculum_seed": int(curriculum)})
json.dump(
    {
        "evaluation_seed": int(eval_seed),
        "runs": runs,
        "note": (
            "Training seed, curriculum RNG seed and evaluation seed are three "
            "separate quantities. The evaluation seed is deliberately shared "
            "across policies so the comparison is like for like."
        ),
    },
    open(path, "w"),
    indent=2,
)
PYEOF

MISSING_METRICS=0
{
    echo "# Loop closure report, ${TS}"
    echo ""
    echo "## Inputs"
    echo "- Training parquet: \`$TRAIN_PARQUET\` (sha256 ${TRAIN_SHA:0:12})"
    echo "- Held-out parquet: \`$HELDOUT_PARQUET\` (sha256 ${HELDOUT_SHA:0:12})"
    echo "- Baseline policy: \`$BASELINE_CKPT\`"
    echo "- Env config: \`$ENV_CONFIG\`"
    echo "- Training pool: $NUM_TRAIN parquets (1 real + $NUM_VARIANTS Halton variants)"
    echo "- Pool augmented: $POOL_AUGMENTED"
    echo ""
    echo "## Seeds"
    echo ""
    echo "Three separate quantities, recorded separately (\`seeds.json\`)."
    echo ""
    echo "| run | training seed | curriculum RNG seed | evaluation seed |"
    echo "|---|---|---|---|"
    for i in "${!SEEDS[@]}"; do
        echo "| adapted_seed${SEEDS[$i]} | ${ACTUAL_TRAINING_SEEDS[$i]} |" \
             "${ACTUAL_CURRICULUM_SEEDS[$i]} | $EVAL_SEED |"
    done
    echo "| baseline | (not retrained here) | n/a | $EVAL_SEED |"
    echo ""
    echo "Each training seed was read back from \`<run_dir>/seeds.json\` after the"
    echo "run, and the set was asserted distinct. The evaluation seed is shared"
    echo "on purpose."
    echo ""
    echo "## Eval metrics"
    echo ""
    echo "Evaluated on \`$ENV_CONFIG\`. \`slew_sat_pct\` is the"
    echo "deploy-equivalent clip-activation rate; \`legacy_slew_pct\` is the old"
    echo "raw-action-delta number, printed only to line up with results recorded"
    echo "before 2026-09-11."
    echo ""
    echo "| policy | success_rate | mean_ep_length_s | slew_sat_pct | legacy_slew_pct | lin_vel_err | ang_vel_err |"
    echo "|---|---|---|---|---|---|---|"
    for label in "${LABELS[@]}"; do
        J="$OUT_DIR/eval_${label}.json"
        if [[ ! -f "$J" ]]; then
            echo "| $label | MISSING | MISSING | MISSING | MISSING | MISSING | MISSING |"
            MISSING_METRICS=1
            continue
        fi
        python3 -c "
import json
d = json.load(open('$J'))
print('| $label | {:.3f} | {:.2f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |'.format(
    d['success_rate'], d['mean_episode_length_s'], d['slew_saturation_pct'],
    d.get('legacy_raw_action_delta_pct', float('nan')),
    d['mean_lin_vel_error'], d['mean_ang_vel_error']))"
    done
    echo ""
    echo "## Held-out arm"
    echo ""
    echo "The held-out trajectory never entered the training pool, checked by"
    echo "content hash against every staged parquet."
    echo ""
    if [[ "$HELDOUT_RAN" != true ]]; then
        echo "The held-out ARM was SKIPPED (\`--skip-heldout\`), so this run carries"
        echo "no held-out evidence and no generalization claim may be made from it."
    else
        echo "Each policy was used to replay \`$HELDOUT_PARQUET\` under Halton"
        echo "perturbations drawn with variation seed \`$HELDOUT_VARIATION_SEED\`,"
        echo "which differs from the training variation seed"
        echo "\`$TRAIN_VARIATION_SEED\`, so these perturbation points are disjoint"
        echo "from the ones the curriculum trained on. Lower is better."
        echo ""
        echo "| policy | held-out failure rate | variants failed |"
        echo "|---|---|---|"
        for label in "${LABELS[@]}"; do
            echo "| $label | ${HELDOUT_FAIL_RATE[$label]} | ${HELDOUT_COUNTS[$label]} |"
        done
    fi
    echo ""
    echo "## Decision"
    echo ""
    echo "Pre-declared BEFORE the numbers above were read. The rule is stated"
    echo "here rather than chosen after the fact, and it is deliberately not"
    echo "\"any seed that improved\": with N seeds and several metrics, \"any\""
    echo "passes on noise alone, and reporting only the winner is cherry-picking."
    echo ""
    echo "Gate 9 intermediate (sim-only) passes only if ALL of:"
    echo "1. the MEAN adapted success_rate across ALL ${#SEEDS[@]} seeds exceeds the"
    echo "   baseline's, and no individual seed regresses by more than 5 points;"
    echo "2. the MEAN adapted held-out failure rate is BELOW the baseline's"
    echo "   (skipped runs cannot satisfy this, and therefore cannot pass);"
    echo "3. no increase in mean \`slew_sat_pct\` versus baseline."
    echo ""
    echo "Every seed is in the tables above. Report all of them, including the"
    echo "worst. If the rule is not met, this run is negative evidence: iterate"
    echo "failure_sample_fraction or the env config and re-run the whole script."
    echo ""
    echo "## What this run is NOT"
    echo ""
    echo "- Sim-only. A pass books hardware time; it does not replace it."
    echo "- The Halton variants are perturbed copies of ONE captured state, so a"
    echo "  gain may be specific to that seed family rather than to the failure"
    echo "  mode. Generalization needs captures from several independent events."
    echo "- No fresh post-training hardware trial has been run."
    echo ""
    echo "## Next"
    echo "- If pass: re-export the ONNX, run the parity gate, then rsync to T7"
    echo "  and stage the payload for a hardware session. Do NOT pick the best"
    echo "  seed; carry the whole set forward or re-run with more seeds."
    echo "- Held-out parquet \`$HELDOUT_PARQUET\` stays reserved, do NOT train on it."
} > "$REPORT"

echo ""
echo "[loop_closure] done. report: $REPORT"
cat "$REPORT"

if [[ "$MISSING_METRICS" -ne 0 ]]; then
    die "one or more evaluations produced no metrics JSON; the table above has
                MISSING rows and the comparison is incomplete."
fi
