# Convert models to Openvino Intermediate Representation

In order to run inference with Openvino, models must be converted to Intel intermediate representation (IR for short). The IR include two files:

- `*.xml`: define the computation graph, human-interpretable 
- `*.bin`: store the weight of the models

> **_NOTE:_** Two above files should have same name (except the extension) and should be place in a same location.

When convert to OpenVino models, the conversion tool will first build the graph from the original models, optimize the graph by removing redundant nodes or fusing layers, then create a mapping between nodes in original graph and the IR graph. In addition, there there are several options users can add:

1. Reverse input channel: If the original model input format is `NHWC`, `--reverse_input_channel` will convert the IR's input format to `NCHW`
2. Custom data type: `FP32` is floating point 32 bits, while `FP16` is floating point 16 bits. In order to run some layers on FPGA, those layers must be `FP16`
3. Custom operator config: When there are operators in original framework that are not supported by OpenVino, or the operators are not even exist in the original framework, users can create custom operator mapping for those operators (and create the extension for the operator as well)

> **_NOTE:_** Remind that the deep learning models is a graph with nodes are operators and data is streamed on edges.

The good thing is almost all operators in famous deep learning frameworks are supported by OpenVino and their open source CPU extension. The bad thing is:

1. If you are using the A10 Arria 10 dev kit, OpenVino is limited to version 2019R1.1, that are also limited the support to newer layers and operators.
2. When you work with object detection models, there are many non-standard operators. Even most of them are implemented in the CPU extension, you will need to replace some sub-graph in the original models by some sub-graph that OpenVino can interpret. That say, you need to understand the model at operator-level, not layer-level. More information [here](https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_Object_Detection_API_Models.html)