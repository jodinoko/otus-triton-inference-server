import json
import urllib.error
import urllib.request

import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def initialize(self, args):
        config = json.loads(args["model_config"])
        params = {key: value["string_value"] for key, value in config["parameters"].items()}
        self.url = params["vllm_url"].rstrip("/")
        self.model = params["model"]
        self.max_tokens = int(params["max_tokens"])
        self.timeout = float(params["timeout_seconds"])

        with urllib.request.urlopen(f"{self.url}/models", timeout=10) as response:
            models = json.load(response)
        if self.model not in {item["id"] for item in models["data"]}:
            raise RuntimeError(f"vLLM has not loaded {self.model}")

    def execute(self, requests):
        responses = []
        for request in requests:
            try:
                raw = pb_utils.get_input_tensor_by_name(request, "PROMPT").as_numpy()
                if raw.shape != (1,):
                    raise ValueError("PROMPT must have shape [1]")
                prompt = raw[0].decode("utf-8") if isinstance(raw[0], bytes) else str(raw[0])
                if not prompt.strip():
                    raise ValueError("PROMPT must not be empty")
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": self.max_tokens,
                    "temperature": 0,
                    "stream": False,
                }
                call = urllib.request.Request(
                    f"{self.url}/chat/completions",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(call, timeout=self.timeout) as response:
                    choice = json.load(response)["choices"][0]
                content = choice["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("vLLM returned empty content")
                outputs = [
                    pb_utils.Tensor("TEXT", np.array([content.encode("utf-8")], dtype=object)),
                    pb_utils.Tensor(
                        "FINISH_REASON",
                        np.array([str(choice["finish_reason"]).encode("utf-8")], dtype=object),
                    ),
                ]
                responses.append(pb_utils.InferenceResponse(output_tensors=outputs))
            except urllib.error.HTTPError as error:
                details = error.read().decode("utf-8", errors="replace")[:1000]
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(f"vLLM HTTP {error.code}: {details}")
                    )
                )
            except Exception as error:
                responses.append(pb_utils.InferenceResponse(error=pb_utils.TritonError(str(error))))
        return responses
