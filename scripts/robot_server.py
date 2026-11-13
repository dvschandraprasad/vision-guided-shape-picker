from __future__ import annotations

import socket
import sys

# Local PC address used to listen for the robot.
HOST = "192.168.125.205"
PORT = 5000


# Send one line to the robot and read its reply.
def send_line(sock: socket.socket, text: str) -> str | None:
    sock.sendall(text.encode("utf-8"))
    return sock.recv(4096).decode("latin-1", errors="replace").strip()


def main() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
        # Let the server restart on the same port.
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((HOST, PORT))
        server.listen()
        print(f"listening on {HOST}:{PORT}")
        # Wait until one robot connects, then keep it.
        client, address = server.accept()
        with client:
            print(f"robot {address[0]} connected")
            print("type a message to send it, or 'quit' to stop")
            # Send a test line to check the link.
            print("reply:", send_line(client, "connection exists"))
            for line in sys.stdin:
                text = line.strip()
                if text == "quit":
                    break
                if text:
                    print("reply:", send_line(client, text))
            # Tell the robot the session is finished.
            send_line(client, "quit")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
