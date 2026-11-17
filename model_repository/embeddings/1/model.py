from pathlib import Path

import numpy as np
import triton_python_backend_utils as pb_utils
from sentence_transformers import SentenceTransformer


class TritonPythonModel:
    def initialize(self, args):
        encoder_dir = Path(__file__).resolve().parent / "encoder"
        if not encoder_dir.is_dir():
            raise RuntimeError("Encoder is missing: run scripts/prepare_models.py first")
        self.encoder = SentenceTransformer(str(encoder_dir), device="cpu", local_files_only=True)
        if self.encoder.get_sentence_embedding_dimension() != 384:
            raise RuntimeError("Expected an encoder with 384 output features")

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                raw = pb_utils.get_input_tensor_by_name(request, "TEXT").as_numpy()
                if raw.shape != (1,):
                    raise ValueError("TEXT must have shape [1]")
                text = raw[0].decode("utf-8") if isinstance(raw[0], bytes) else str(raw[0])
                if not text.strip():
                    raise ValueError("TEXT must not be empty")
                embedding = self.encoder.encode(
                    [text], normalize_embeddings=True, convert_to_numpy=True
                ).astype(np.float32)
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[pb_utils.Tensor("EMBEDDING", embedding)]
                    )
                )
            except Exception as error:
                responses.append(pb_utils.InferenceResponse(error=pb_utils.TritonError(str(error))))
        return responses
