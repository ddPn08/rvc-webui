import argparse

parser = argparse.ArgumentParser()

parser.add_argument("--host", help="Host to connect to", type=str, default="127.0.0.1")
parser.add_argument("--port", help="Port to connect to", type=int)
parser.add_argument("--share", help="Enable gradio share", action="store_true")
parser.add_argument(
    "--models-dir", help="Path to models directory", type=str, default=None
)
parser.add_argument(
    "--output-dir", help="Path to output directory", type=str, default=None
)
parser.add_argument(
    "--precision",
    help="Precision to use",
    type=str,
    default="fp16",
    choices=["fp32", "fp16"],
)

opts, _ = parser.parse_known_args()
