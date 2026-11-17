import csv
from pathlib import Path

import numpy as np
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType

ROOT = Path(__file__).resolve().parents[1]
ENCODER_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


def export_classifier(classifier):
    model = convert_sklearn(
        classifier,
        name="review_sentiment",
        initial_types=[("EMBEDDING", FloatTensorType([1, 384]))],
        final_types=[
            ("LABEL", Int64TensorType([1])),
            ("PROBABILITIES", FloatTensorType([1, 2])),
        ],
        options={id(classifier): {"zipmap": False}},
        target_opset={"": 13, "ai.onnx.ml": 1},
    )
    onnx.checker.check_model(model)
    return model


def main():
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression

    with (ROOT / "data/reviews.csv").open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    encoder_dir = ROOT / "model_repository/embeddings/1/encoder"
    if encoder_dir.is_dir():
        encoder = SentenceTransformer(str(encoder_dir), device="cpu", local_files_only=True)
    else:
        encoder = SentenceTransformer(ENCODER_ID, device="cpu")
        encoder.save(str(encoder_dir))
    vectors = encoder.encode(
        [row["text"] for row in rows], normalize_embeddings=True, convert_to_numpy=True
    ).astype(np.float32)
    labels = np.array([int(row["label"]) for row in rows])
    train = np.array([row["split"] == "train" for row in rows])
    classifier = LogisticRegression(C=5.0, max_iter=1000, random_state=42)
    classifier.fit(vectors[train], labels[train])
    model = export_classifier(classifier)
    destination = ROOT / "model_repository/classifier/1/model.onnx"
    destination.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, destination)


if __name__ == "__main__":
    main()
