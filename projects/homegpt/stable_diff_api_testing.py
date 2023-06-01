import os
import webuiapi

HOST = os.environ.get("STABLE_DIFF_API_HOST")
PORT = os.environ.get("STABLE_DIFF_API_PORT")


def main():
    api = webuiapi.WebUIApi(host=HOST, port=PORT, steps=50)
    result = api.txt2img(
        prompt="a big hairy horse eating ramen noodles from a trough", 
        negative_prompt="noodles are bad for horses",
    )
    print(result.image)
    # result.image is a PIL.Image

    # Save to file
    result.image.save("images/horse.png")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    main()
