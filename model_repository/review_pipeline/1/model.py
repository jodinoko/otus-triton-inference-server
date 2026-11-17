import json

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        config = json.loads(args["model_config"])
        params = {key: value["string_value"] for key, value in config["parameters"].items()}
        self.embeddings_model = params["embeddings_model"]
        self.classifier_model = params["classifier_model"]
        self.generation_model = params["generation_model"]
        self.model_version = int(params["model_version"])

    def infer(self, model_name, inputs, output_names):
        response = pb_utils.InferenceRequest(
            model_name=model_name,
            model_version=self.model_version,
            requested_output_names=output_names,
            inputs=inputs,
        ).exec()
        if response.has_error():
            raise pb_utils.TritonModelException(f"{model_name}: {response.error().message()}")
        return [pb_utils.get_output_tensor_by_name(response, name) for name in output_names]

    def execute(self, requests):
        responses = []
        for request in requests:
            text_tensor = pb_utils.get_input_tensor_by_name(request, "TEXT")
            text = text_tensor.as_numpy()[0].decode("utf-8")

            (embedding,) = self.infer(self.embeddings_model, [text_tensor], ["EMBEDDING"])
            label, probabilities = self.infer(
                self.classifier_model, [embedding], ["LABEL", "PROBABILITIES"]
            )
            if label.as_numpy()[0] == 0:
                prompt = f"Напиши короткий вежливый ответ магазина на отзыв: {text}"
                prompt_tensor = pb_utils.Tensor(
                    "PROMPT", np.array([prompt.encode("utf-8")], dtype=object)
                )
                generated_text, finish_reason = self.infer(
                    self.generation_model, [prompt_tensor], ["TEXT", "FINISH_REASON"]
                )
                reply = pb_utils.Tensor("REPLY", generated_text.as_numpy())
            else:
                reply = pb_utils.Tensor(
                    "REPLY",
                    np.array(["Спасибо за ваш отзыв!".encode()], dtype=object),
                )
                finish_reason = pb_utils.Tensor(
                    "FINISH_REASON", np.array([b"not_called"], dtype=object)
                )
            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[label, probabilities, reply, finish_reason]
                )
            )
        return responses
