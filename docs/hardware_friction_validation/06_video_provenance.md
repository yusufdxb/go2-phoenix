# 06 — Video provenance: untethered README footage (2026-10-12)

## VERIFIED
- **`IMG_4880.mp4` and `IMG_4881.mp4` weren't found** by those names on the MacBook (Spotlight; `find` over
  home, Pictures, Messages attachments, iCloud Drive, Movies, Downloads, Desktop, Documents), on mewtwo
  (`find ~ -xdev`), or on the Jetson.
- The README demo (`33ef16f`, 2026-10-09 19:40Z) was built on mewtwo from
  `~/workspace/go2-phoenix-demo-v2/` (all files written 2026-10-09 14:57 local) into
  `~/workspace/go2-phoenix-demo-build/brag-output-2026-10-09-150217/brag.mp4` (25.2 s).
- Source clips (contact sheets in `~/Projects/phoenix-friction-data/video_provenance/`):

| clip | duration | format | embedded time | content | used in README cut as |
|---|---|---|---|---|---|
| `run1.mp4` | 40.93 s | 512×910 30 fps H.264 + AAC | `creation_time` 2026-10-09T18:48:04–10Z | green-walled room; GO2 lying → stands → walks left to right across open floor → lies down; seated operator with remote; **no rope visible** | "RUN 2 · FREE FLOOR · NO ROPE — First untethered steps" |
| `run2.mp4` | 42.67 s | same | 2026-10-09T18:47:50–48:04Z | large open lab; GO2 stands, turns in place, sits | "Turns about 10°", sim 87 % vs real 20 % |

- No camera make/model, capture date or location tags survive. Both are 512×910 re-encodes whose
  `creation_time` values are 14 s apart. That's consistent with a transfer/re-encode on 10-09, **not**
  evidence of capture time.
- **Phoenix acquisition logs:** the only Phoenix walking runs anywhere are the four gantry W runs
  (10-06 23:54Z `vx03_2s`; 10-07 00:01Z `vx03_4s`, 00:06Z and 00:10Z `yaw05_3s`). All have checkpoint
  `e03013d1`, ONNX `6d36796f`, manifest `c6d4a320`, code `34104ff`, and the typed "ON GANTRY" W stage. No Phoenix
  session exists after 2026-10-07 01:21Z, and no `bridge.jsonl`/`walk_cmd.jsonl`/`policy.parquet` exists outside them.
  The mewtwo EVIDENCE note (`158ecea`, 10-07 13:24) states "no walking command has been sent
  without the gantry".
- **Timeline match** (motion energy of `run1` vs bridge mode timelines):

| | stand-up → walking onset | walking span | stand-up → sit |
|---|---|---|---|
| run1 video | ≈ 7.1 s | ≈ 3.6 s | ≈ 17.2 s |
| `W_vx03_4s` 10-07 | 6.8 s | 4.55 s | 17.0 s |
| `W_vx03_2s` 10-06 | 6.5 s | 2.55 s | 14.9 s |

- The robot was also run by other stacks on 10-09 (`~/mosaic_sessions/20261009T153348Z_mosaic_lab`,
  `~/fetch_sessions/2026-10-09/demo_5080`).
- The Jetson boots with its clock at 1970 (`last reboot`), so session timestamps depend on the clock being
  set after each boot.

## INFERRED
- `run1` was most likely run through the Phoenix harness sequencing (stand-up, ~11–13 s policy window,
  sit). Its timing fits a 4-s-class forward preset better than `vx03_2s`. The harness timing is
  identical across policies and runs, though, so this **doesn't identify the run or the checkpoint**.
- If `run1` is `W_vx03_4s` of 10-07, its "free floor, no rope" label conflicts with the 10-07 note
  and the typed "ON GANTRY" confirmation. If it's a different run, that run left no Phoenix log on the
  Jetson. Either way the untethered envelope (preset, distance, conditions) is **undocumented**.
- The README cut's "Turns about 10°" disagrees with the logged IMU yaw of 17.6°/19.0° and with the README text (17 of 86°).

## HYPOTHESES
- Operator testimony: the untethered run used the 10-07 checkpoint. Not independently established.

## OPEN (needed to close provenance)
1. The original `IMG_4880.MOV/.mp4` and `IMG_4881` from the capturing phone (Photos keeps original
   capture date, device and location; AirDrop/"Export Unmodified Original" keeps them).
2. Which session/terminal launched the untethered run, and where its logs went.
3. Whether `run1` == `IMG_4880` (durations: run1 40.9 s; the stated ~14 s standing clip matches neither file).
