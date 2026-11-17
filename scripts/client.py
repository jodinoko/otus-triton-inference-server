import argparse

import numpy as np
import tritonclient.http as httpclient


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", nargs="?", default="Товар сломался в первый день")
    args = parser.parse_args()

    text = httpclient.InferInput("TEXT", [1], "BYTES")
    text.set_data_from_numpy(np.array([args.text.encode("utf-8")], dtype=object))

    with httpclient.InferenceServerClient(
        url="localhost:8000", connection_timeout=10, network_timeout=240
    ) as client:
        result = client.infer("review_pipeline", inputs=[text], model_version="1")

    label = int(result.as_numpy("LABEL")[0])
    print("Класс:", {0: "негативный", 1: "позитивный"}[label])
    print("Вероятности [негативный, позитивный]:", result.as_numpy("PROBABILITIES")[0])
    print("Ответ:", result.as_numpy("REPLY")[0].decode("utf-8"))
    print("Завершение генерации:", result.as_numpy("FINISH_REASON")[0].decode("utf-8"))


if __name__ == "__main__":
    main()
