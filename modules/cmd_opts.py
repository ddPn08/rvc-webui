import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--host", help="Host to connect to", type=str, default="localhost")
parser.add_argument("--port", help="Port to connect to", type=int)
parser.add_argument("--share", help="Enable gradio share", action="store_true")
parser.add_argument(
    "--models-dir", help="Path to models directory", type=str, default=None
)
parser.add_argument(
    "--output-dir", help="Path to output directory", type=str, default=None
)

opts, _ = parser.parse_known_args()
