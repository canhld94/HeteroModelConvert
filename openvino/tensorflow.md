# Convert Tensorflow models to Openvino IR

Reference [Converting a TensorFlow* Model](https://docs.openvinotoolkit.org/latest/openvino_docs_MO_DG_prepare_model_convert_model_Convert_Model_From_TensorFlow.html)

## General Tensorflow models

The conversion is straightforward if the model does not employ any custom layers. In general, most of popular CNN network (Resnet, Inception, VGG , etc) are convertible.

**Example: Convert resnet-50 models**

Source models: Pre-trained on ImageNet

Compile command (Jupyter Notebook):

```
!mo_tf.py --input_shape=[16,224,224,3] --mean_values=[123.8,116.78,103.94] --input=map/TensorArrayStack/TensorArrayGatherV3 --input_model=resnet_v1-50.pb --reverse_input_channels --data_type=FP16
```

Download models: [Source](https://drive.google.com/drive/u/0/folders/1r7i_FX3ILdwZaBeZZH26ekUSNlzuCpOx) or [IR](https://drive.google.com/drive/u/0/folders/1aHlhZgq6GHltVizsTrvnCbZihp4ygyAL)

## Tensorflow Object Detection models

### Single Shot Detector with TF object detection

The back-bone of the graph are keep, while the pre-processor are removed and the post-processor is replaced with a sub-graph call `DetectionOutput` from inference engine. The custom removal and replacement are specified in the `ssd_v2_support.json` file.

**Example: Convert SSD Mobilenet v2 models**

Source models: Pre-trained on DOTA dataset with object detection API

Conversion command: (Jupyter Notebook):

```
pipeline = 'deploy/tf_model/DOTA/ssd_mobilenet_v2_pipeline.config'
graph = 'deploy/tf_model/DOTA/ssd_mobilenet_v2.pb'
dtype = 'FP16' # FP32 for CPU & GPU, FP16 for GPU and FPGA
custom_ops_config = 'deploy/tf_model/DOTA/ssd_v2_support.json'
!mo_tf.py --input_model $graph --tensorflow_use_custom_operations_config $custom_ops_config --tensorflow_object_detection_api_pipeline_config $pipeline --batch 1 --reverse_input_channels --data_type=$dtype
```

Download models: [Source](https://drive.google.com/drive/u/0/folders/19JQdsr2iSQbOR2sbvzol0FSnKI9DhxrU) or [IR](https://drive.google.com/drive/u/0/folders/1Ni7DpIwfJK7kUibE2hQuIrJMG0rX68Ry)

### Faster-RCNN models with TF object detection

Similar to SSD, the pre-processor blocks are removed. In addition, OpenVino will intercept the output of the original model and perform post-processing like in SSD. In order to do that, the model converter will add `DetectionOutput` sub-graph before the output of original model (`num_detections`, `detection_boxes`, `detection_classes`, `detection_scores`). The removal and replacement are specified in `faster_rcnn_support.json`

**Example: Faster RCNN Resnet50 models**

Source models: Pre-trained on DOTA dataset with object detection API

Conversion command: (Jupyter Notebook):

```
pipeline = 'deploy/tf_model/DOTA/faster_rcnn_resnet50_pipeline.config'
graph = 'deploy/tf_model/DOTA/faster_rcnn_resnet50.pb'
dtype = 'FP16' # FP32 for CPU & GPU, FP16 for GPU and FPGA
custom_ops_config = 'deploy/tf_model/DOTA/faster_rcnn_support.json'
!mo_tf.py --input_model $graph --tensorflow_use_custom_operations_config $custom_ops_config --tensorflow_object_detection_api_pipeline_config $pipeline --batch 1 --reverse_input_channels --data_type=$dtype
```

Download models: [Source](https://drive.google.com/drive/u/0/folders/19JQdsr2iSQbOR2sbvzol0FSnKI9DhxrU) or [IR](https://drive.google.com/drive/u/0/folders/1Ni7DpIwfJK7kUibE2hQuIrJMG0rX68Ry)