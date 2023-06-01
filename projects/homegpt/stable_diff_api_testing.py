import os
from pathlib import Path

import webuiapi

HOST = os.environ.get("STABLE_DIFF_API_HOST")
PORT = os.environ.get("STABLE_DIFF_API_PORT")


_output_dir = Path("images/")

def main():
    api = webuiapi.WebUIApi(host=HOST, port=PORT, steps=20)
    result = api.txt2img(
        prompt="A frog wizard holding a wand", 
    )

    # result.image is a PIL.image
    # Save to file
    _output_dir.mkdir(parents=True, exist_ok=True)
    result.image.save(_output_dir / "horse.png")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    main()
