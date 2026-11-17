FROM nvcr.io/nvidia/tritonserver:24.01-py3

RUN python3 -m pip install --no-cache-dir \
    numpy==1.26.4 \
    torch==2.6.0 \
    transformers==4.48.3 \
    sentence-transformers==3.4.1 \
    scikit-learn==1.6.1 \
    skl2onnx==1.18.0 \
    onnx==1.17.0
