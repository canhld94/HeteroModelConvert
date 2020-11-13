# Convert models to TensorRT models

TensorRT come with three parser: 

1. Cafe parser: default parser
2. UFF parser: parser models in uff format; only work with tesorflow models
3. ONNX parser: parser models in onnx format

Because I work with tensorflow, I use the uff parser. The workflow is:

1. Convert models from tensorflow `.pb` to `.uff`
2. Parse the `uff` models with TensorRT Python API and write the serialized engine to file `.bin`
3. Use the serialized engine in C++ code

## Convert SSD models

Source file: [engine_builder_ssd.py](engine_builder_ssd.py)

1. Convert model to uff model

In order to convert any model to uff format, all operator in the layer must be support by uff. The subgraph or nodes which is not-supported must be removed or replaced with proper NVIDIA extension.