from __future__ import annotations

import pipeline

import argparse
import socket
import threading
from pathlib import Path

import cv2

import calibrate

SAMPLE_FRAME = pipeline.REPO_ROOT / "data" / "captured_frames" / "calibration_frame.png"
# The three shapes one grade-I cycle must pick.
ASSIGNED_SHAPES = ("circle", "hexagon", "triangle")

# Keep only the best detection for each shape.
def choose(detections: list[pipeline.Detection],
           shapes: tuple[str, ...]) -> list[tuple[str, pipeline.Detection]]:
    best: dict[str, pipeline.Detection] = {}
    for detection in detections:
        if detection.name not in shapes:
            continue
        held = best.get(detection.name)
        if held is None or detection.confidence > held.confidence:
            best[detection.name] = detection
    return [(shape, best[shape]) for shape in shapes if shape in best]

# A pretend robot on loopback, no hardware touched.
class FakeRobot(threading.Thread):

    def __init__(self, host: str, port: int, parts: int, timeout_s: float = 10.0) -> None:
        super().__init__(daemon=True, name="fake-robot")
        self.host = host
        self.port = port
        self.parts = parts
        self.timeout_s = timeout_s
        self.answers: list[str] = []
        self.errors: list[str] = []
        self.connected = threading.Event()

    def run(self) -> None:
        try:
            with socket.create_connection((self.host, self.port), timeout=self.timeout_s) as link:
                self.connected.set()
                buffer = ""
                # Ask for one part per shape, then read the line.
                for _ in range(self.parts):
                    link.sendall(b"READY\n")
                    while "\n" not in buffer:
                        chunk = link.recv(4096)
                        if not chunk:
                            self.errors.append("the PC closed the link mid-rehearsal")
                            return
                        buffer += chunk.decode("utf-8", errors="replace")
                    line, _, buffer = buffer.partition("\n")
                    self.answers.append(line.strip())
        except OSError as error:
            self.errors.append(f"could not reach {self.host}:{self.port}: {error}")
            self.connected.set()

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="grade-I dry run: three shapes in, three pick lines out, no hardware")
    parser.add_argument("--image", default=str(SAMPLE_FRAME), help="frame with the marker in it")
    parser.add_argument("--model", default=pipeline.MODEL_PATH, help="OBB weights")
    parser.add_argument("--conf", type=float, default=0.25, help="detector confidence")
    parser.add_argument("--shapes", default=",".join(ASSIGNED_SHAPES),
                        help="one part per shape, most confident first")
    parser.add_argument("--host", default="127.0.0.1", help="loopback: no robot is touched")
    parser.add_argument("--port", type=int, default=0, help="0 asks the OS for a free port")
    parser.add_argument("--timeout", type=float, default=10.0, help="seconds per exchange")
    arguments = parser.parse_args(argv)

    # Split the shape list typed on the command line.
    wanted = tuple(name.strip() for name in arguments.shapes.split(",") if name.strip())
    image = cv2.imread(arguments.image, cv2.IMREAD_COLOR)
    if image is None:
        print(f"cannot read {arguments.image}")
        return 2

    print(f"frame   {arguments.image} ({image.shape[1]}x{image.shape[0]})")
    print(f"model   {arguments.model} at conf {arguments.conf:.2f}")
    print(f"wanted  {', '.join(wanted)}")
    print()

    detections = pipeline.ShapeDetector(model_path=arguments.model,
                                        conf=arguments.conf).detect(image)
    try:
        # Use the printed marker to turn pixels into millimetres.
        plane = calibrate.marker_scale(image, marker_mm=calibrate.MARKER_MM)
    except ValueError as error:
        print(f"no usable marker in this frame: {error}")
        return 2

    chosen = choose(detections, wanted)
    missing = [shape for shape in wanted if shape not in {name for name, _ in chosen}]
    for shape in missing:
        print(f"  {shape}: not in this frame - a real robot would sit and wait")

    # Turn each detection into one pick command.
    commands = [(shape, plane.to_pick_command(detection)) for shape, detection in chosen]
    if not commands:
        print("nothing to send")
        return 1

    with pipeline.TcpTransport(host=arguments.host, port=arguments.port) as link:
        link.open()
        # Start the pretend robot before the PC waits for it.
        pendant = FakeRobot(arguments.host, link.port, len(commands), arguments.timeout)
        pendant.start()
        try:
            link.accept(arguments.timeout)
        except pipeline.TransportError as error:
            print(f"the fake pendant never connected: {error}")
            return 1
        client = pipeline.RobotClient(link)
        for _shape, command in commands:
            # Wait for the robot to ask for the next part.
            trigger = client.wait_for_request(arguments.timeout)
            if trigger is None:
                print("the pendant went quiet mid-rehearsal")
                break
            client.send_pick(command)

    pendant.join(arguments.timeout)

    print(f"{'part':<5}{'shape':<10}{'sent':<28}{'read back'}")
    intact = 0
    for index, (shape, command) in enumerate(commands, start=1):
        sent = pipeline.encode_command(command)
        if index <= len(pendant.answers):
            try:
                # Decode the answer to see what came back.
                back = pipeline.decode_command(pendant.answers[index - 1])
            except ValueError as error:
                read = f"corrupted: {error}"
            else:
                intact += 1
                read = f"slot {back.shape}, {back.x_mm:.1f}, {back.y_mm:.1f}, angle {back.angle_deg:.1f}"
        else:
            read = "no answer"
        print(f"{index:<5}{shape:<10}{sent:<28}{read}")

    for error in pendant.errors:
        print(f"  pendant: {error}")

    print()
    print(f"found {len(chosen)} of {len(wanted)} shape(s) | sent {len(commands)} | "
          f"read back intact {intact}")
    print("ORIENTATION is physical: >=2/3 have to land in the pocket - watch the cup. "
          "A circle carries angle 0.0; the folded angle is what the pocket must swallow.")
    # Exit 0 only if all shapes were found and echoed.
    return 0 if not missing and intact == len(commands) else 1

if __name__ == "__main__":
    raise SystemExit(main())
