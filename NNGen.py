import numpy as np
import coremltools as ct
from coremltools.models.neural_network import NeuralNetworkBuilder, SgdParams, flexible_shape_utils
from coremltools.models.datatypes import Array
from coremltools.models import MLModel
from coremltools.proto.FeatureTypes_pb2 import ImageFeatureType

# ------------------------------
# 1. Define Input/Output Features (with explicit batch dimension)
# ------------------------------
# Prediction Input: an RGB image with shape (1, 3, 224, 224)
input_features = [("image", Array(1, 3, 224, 224))]
# Prediction Output: class probabilities as a multi-array with shape (1, 10)
output_features = [("final_output", Array(1, 10))]

# ------------------------------
# 2. Build the Neural Network
# ------------------------------
# Architecture:
#   - Convolution (8 filters, 3x3, same padding)
#   - ReLU activation
#   - Global average pooling (output shape: (1,8,1,1))
#   - Flatten (converts (1,8,1,1) to (1,8))
#   - Fully connected ("fc") mapping (1,8) to (1,10)
#   - Softmax (produces output of shape (1,10))
builder = NeuralNetworkBuilder(input_features, output_features, disable_rank5_shape_mapping=True)

# Convolution Layer
W_conv = np.random.rand(8, 3, 3, 3).astype(np.float32)
b_conv = np.random.rand(8).astype(np.float32)
builder.add_convolution(
    name="conv1",
    kernel_channels=3,
    output_channels=8,
    height=3,
    width=3,
    stride_height=1,
    stride_width=1,
    border_mode="same",
    input_name="image",
    output_name="conv1",
    groups=1,
    W=W_conv,
    b=b_conv,
    has_bias=True
)

# ReLU Activation
builder.add_activation(
    name="relu1",
    non_linearity="RELU",
    input_name="conv1",
    output_name="relu1"
)

# Global Average Pooling (output shape: (1,8,1,1))
builder.add_pooling(
    name="global_pool",
    height=0,
    width=0,
    stride_height=1,
    stride_width=1,
    layer_type="AVERAGE",
    input_name="relu1",
    output_name="pool",
    padding_type="VALID",
    is_global=True
)

# Flatten Layer: (1,8,1,1) --> (1,8)
builder.add_flatten(
    name="flatten",
    input_name="pool",
    output_name="flat",
    mode=0
)

# Fully Connected Layer ("fc"): Maps (1,8) --> (1,10)
W_fc = np.random.rand(10, 8).astype(np.float32)
b_fc = np.random.rand(10).astype(np.float32)
builder.add_inner_product(
    name="fc",
    input_name="flat",
    output_name="fc_output",
    W=W_fc,
    b=b_fc,
    input_channels=8,
    output_channels=10,
    has_bias=True
)

# Softmax Layer: Converts logits to probabilities (output shape: (1,10))
builder.add_softmax(
    name="softmax",
    input_name="fc_output",
    output_name="final_output"
)

# ------------------------------
# 3. Mark the Model as Updatable and Set Training Parameters
# ------------------------------
spec = builder.spec
spec.isUpdatable = True
spec.specificationVersion = 4

# Mark the fully connected layer "fc" as updatable.
builder.make_updatable(["fc"])

# Set a categorical cross entropy loss layer on the softmax output.
builder.set_categorical_cross_entropy_loss(name="lossLayer", input="final_output")

# Set an SGD optimizer.
builder.set_sgd_optimizer(SgdParams(lr=0.01, batch=32))

# Set the number of training epochs.
builder.set_epochs(10)

# ------------------------------
# 4. Convert the Input to an Image Type
# ------------------------------
# VNCoreMLModel requires an image input.
# Clear the current type and set an image type with the desired properties.
spec.description.input[0].ClearField("type")
spec.description.input[0].type.imageType.width = 224
spec.description.input[0].type.imageType.height = 224
spec.description.input[0].type.imageType.colorSpace = ImageFeatureType.RGB

# ------------------------------
# 5. Fix Softmax Output Rank in the Spec
# ------------------------------
# The softmax layer outputs shape (1,10) (rank 2), but VNCoreMLModel expects rank >= 3.
# Update the output multi-array shape to [1,10,1].
spec.description.output[0].type.multiArrayType.shape[:] = [1, 10, 1]

# ------------------------------
# 6. Define Training Inputs (after input conversion)
# ------------------------------
# Clear any pre-existing training inputs.
spec.description.ClearField("trainingInput")

# Add the model's input "image" as a training input.
spec.description.trainingInput.add().MergeFrom(spec.description.input[0])

# Create and add a training target for the loss layer.
# The loss layer expects a target named "final_output_true".
true_target = spec.description.output[0].__class__()
true_target.CopyFrom(spec.description.output[0])
true_target.name = "final_output_true"
true_target.shortDescription = "Ground truth for final_output (one-hot vector or class index)"
spec.description.trainingInput.add().MergeFrom(true_target)

# ------------------------------
# 7. Add Metadata
# ------------------------------
spec.description.metadata.author = "Your Company"
spec.description.metadata.license = "MIT"
spec.description.metadata.shortDescription = (
    "An updatable neural network classifier for photos. "
    "The model accepts an input image of shape (1,3,224,224) and outputs class probabilities of shape (1,10) (adjusted to rank 3)."
)

# ------------------------------
# 8. (Optional) Flexible Input Size
# ------------------------------
try:
    flexible_shape_utils.set_input_shape_range(
        spec,
        "image",
        lower_bounds=(1, 3, 32, 32),
        upper_bounds=(1, 3, 1024, 1024),
        default_input_size=(1, 3, 224, 224)
    )
    print("Flexible input shape range set for 'image'.")
except Exception as e:
    print("Flexible shape utilities not available; using default input size.")

# ------------------------------
# 9. Save the Updatable Model (using your specified output location)
# ------------------------------
output_path = "/Your/File/Output/Location/Here/MLTrainee_Updatable.mlmodel"
mlmodel = MLModel(spec)
mlmodel.save(output_path)
print("Updatable neural network model saved to", output_path)
