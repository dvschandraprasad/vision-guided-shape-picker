# Robot and RAPID setup (step by step)

This is for an ABB robot with a controller that has socket messaging (on IRC5 this is the
"PC Interface" option). The values in the RAPID files come from one lab cell. You must teach
your own.

## Files

| File | Load as | What it does |
| --- | --- | --- |
| `ServerCommuncatio.sys` | **System module** | Socket helpers. Robot as client: `RobotAsClientConnect`, `RobotClientReciveMessage`, `RobotClienSendMessage`, `RobotClientCloseAndDisconnect`. Also an unused robot-as-server example. |
| `rapid/MainModule_angle.mod` | **Program module** with `main` | Receive pick lines, check them, wait for the yellow button, pick and place. Also the angle-teach routines and a target check routine. |

Keep the file bytes as they are. `ServerCommuncatio.sys` is Latin-1 text with CRLF line endings
(it has Swedish comments). `.gitattributes` tells Git not to change these files.

## 1. I/O signals

Create (or map) these in the controller's I/O configuration:

| Signal | Type | Use |
| --- | --- | --- |
| `doValve1` | digital output | vacuum on/off |
| `doYellowLamp` | digital output | "press the button to pick" lamp |
| `diYellowBtn` | digital input | operator confirms each pick |

## 2. Tool and work objects

| Data | Meaning |
| --- | --- |
| `toolss` | Vacuum cup tool (TCP at the cup). |
| `wobjS` | **Sheet** frame. Origin exactly at the **ArUco marker centre**, x to the camera image's right, y to the image's up. All pick coordinates from Python are in this frame. |
| `wobjD` | **Destination** frame: the tray with one pocket per shape. |

Teach `wobjS` with the 3-point user frame method on the sheet. If the sheet moves, re-teach
`wobjS` and re-run `calibrate.py --live`.

## 3. Targets

| Target | Frame | Meaning |
| --- | --- | --- |
| `Start` | `wobj0` | Home, out of the camera view. |
| `pickBase` | `wobjS` | Cup-down orientation and arm configuration for every pick. Only its orientation and config are used; Python gives x, y, z. |
| `pCircle`, `pSquare`, `pHex`, `pTriangle`, `pStar` | `wobjD` | Place pose inside each pocket. |

Check the pocket targets with **PP to Routine -> `CheckTargets`**: the robot stops 60 mm above
each pocket in turn.

## 4. Network

The robot is the TCP **client**, the Pi is the **server**.

1. Give the Pi a fixed address on the robot's subnet (the lab used `192.168.125.205`).
   The ABB service port default is `192.168.125.1`; the code's hints call the robot
   `192.168.125.2`. Use the addresses of your own network.
2. In `ServerCommuncatio.sys`, procedure `RobotAsClientConnect`:
   ```
   SocketConnect robotClient, "192.168.125.205", 5000;
   ```
   Put the Pi's address and port here.
3. On the Pi, use the same values: `--host <pi-address> --port 5000`.
   (`grade_levels/run_grade.sh` and `run_dashboard.sh` have `192.168.125.205` written in them.)
4. If a firewall runs on the Pi, open TCP port 5000 (robot) and 8080 (browser).

How the connection behaves:

- `ConnectToPC` (in `MainModule_angle.mod`) calls `RobotAsClientConnect`. That procedure retries
  a timeout up to 5 times (1 s apart); any other error goes back to `ConnectToPC`, which closes
  the socket, waits 2 s and tries again. So the robot keeps trying until the Pi listens.
- The Pi side waits only **30 s** for the robot, so start the Pi first and RAPID soon after.
- On connect the pendant shows "I LIVE!" (from `ServerCommuncatio.sys`) and "Connected to PC.".
- Every received line is also written on the pendant.

## 5. Run

1. On the Pi: `./grade_levels/run_grade.sh 1` (or another grade / `dashboard_web.py --send`).
   These need `scripts/dashboard_new.py`, see [Missing files](../README.md#missing-files).
   Until it is back, `live_pick.py --send --host <pi-address>` can send from one frame (best
   with only one part in view: several lines sent at once can arrive joined together).
   Wait for "model ready". The Pi now waits **30 s** for the robot.
2. On the pendant: **PP to Main**, then **PLAY**. The robot goes to `Start` and shows
   "Waiting for PC", then "Connected to PC.".
3. For each part:
   1. Press **Pick** on the web page.
   2. Pendant: `Pick X=.. Y=.. Z=.. A=..` and "Press the yellow button to pick."; the lamp is on.
   3. Press the **yellow button** (the program waits for it to be released, then pressed).
   4. The robot builds the pick target from `pickBase` at (X, Y, Z) in `wobjS`, already turned by
      A about the cup axis (`RelTool \Rz`). It moves (MoveJ) to 60 mm above the target, goes
      straight down (MoveL, v50), switches the vacuum on, waits 1 s and lifts 60 mm. Then it
      moves above the pocket target in `wobjD`, goes down, switches the vacuum off, waits 1 s
      and lifts. Pendant: **"Done."**.
4. If the Pi closes the link, the robot shows "PC disconnected - waiting for it again..." and
   reconnects by itself (every 2 s).
5. The line `quit` ends the loop: vacuum off, arm to `Start`, "Program finished.".

## 6. Safety checks in RAPID

A line is ignored (with a message on the pendant) if:

- it does not contain five numbers separated by `;` ("Bad message");
- the slot is not 1 to 5;
- x or y is outside -200..200 mm, or z is outside 0..40 mm ("Outside the sheet");
- the angle is outside -180..180 deg.

Every pick also needs the operator's yellow button. Run the first cycles in manual reduced
speed.

## 7. Angle teach routines

**PP to Routine** -> `TeachAll`, `TeachSquare`, `TeachHexagon`, `TeachTriangle` or `TeachStar`.
Settings at the top of the module:

| Constant | Value | Meaning |
| --- | --- | --- |
| `nTeachX`, `nTeachY` | -120, 60 | Drop point on the sheet (wobjS mm). Must match `teach_angle.py --drop-x/--drop-y`. |
| `nTeachH` | 10 | Part height (= `PICK_Z_MM`). |
| `nTeachLift` | 60 | Approach / lift height. |
| `nTeachClearZ` | 200 | Height the cup rises to after a drop. |
| `nTeachWait` | 1.0 s | Vacuum build-up / release. |
| `nTeachTurn` | 20 deg | Test turn; must be 1..26 deg. |
| `nPocketDepth` | 0 | Extra depth to grip a seated part, 0..10 mm. |

The routine asks "Is the part SEATED in its pocket and the drop point EMPTY?" and does nothing
unless you answer **Yes**. Then follow the pendant: it tells you which `teach_angle.py` command
to run before each **PLAY**. See the README section
[Calibration and angle teaching](../README.md#calibration-and-angle-teaching).

## 8. Check the measured points (no vacuum)

`python scripts/verify_positions.py` prints the pick table and writes `VisionCheck.mod`.
Load it in the same task as `MainModule_angle.mod` (it uses `wobjS`, `toolss`, `oPick`,
`cnfPick` and `ejPick` from there) and run **PP to Routine -> `CheckAll`**. For each part the cup
goes to 50 mm (`--safe-z`) above the point, down to the part height, waits 2 s and lifts. The
vacuum is never switched on. Run it in manual reduced speed and watch that the cup touches the
top of each part.

## 9. Manual link test

`scripts/robot_server.py` listens on `192.168.125.205:5000` (edit `HOST` in the file), waits for
the robot, sends "connection exists", and then sends each line you type. Type `quit` to end.
It expects the robot to answer every line, so use it with a small RAPID test program that
replies, not with `MainModule_angle.mod` (which does not reply).
