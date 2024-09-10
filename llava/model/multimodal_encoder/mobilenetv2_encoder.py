import torch
import torch.nn as nn
import numpy as np
import tflite_runtime.interpreter as tflite
from huggingface_hub import hf_hub_download
from PIL import Image
from torchvision import transforms

from transformers import CLIPImageProcessor
import pdb

class MobileNetV2VisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, "mm_vision_select_feature", "patch")

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            raise ValueError("Unexpected model configuration.")

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = tflite.Interpreter(model_path=hf_hub_download(repo_id="mikarbx/mobilenetv2", filename="mobilenet_v2_0.35_128_tl_without_classification_head.tflite"))
        self.vision_tower.allocate_tensors()

        self.input_details = self.vision_tower.get_input_details()
        self.output_details = self.vision_tower.get_output_details()

        self.is_loaded = True

    def save_images(self, tensor_batch, save_path):
        # Loop through the batch and save each image
        for i in range(tensor_batch.shape[0]):  # Iterate over the batch (32 images)
            image_tensor = tensor_batch[i]  # Extract the i-th image tensor, shape [3, 128, 128]

            # Convert the tensor from [3, 128, 128] (C, H, W) to [128, 128, 3] (H, W, C)
            image_np = image_tensor.permute(1, 2, 0).numpy()

            # Scale the values from [0, 1] (PyTorch format) to [0, 255] for saving
            image_np = (image_np * 255).astype(np.uint8)

            # Convert the NumPy array to a PIL Image
            image_pil = Image.fromarray(image_np)

            # Save the image as a PNG file
            image_pil.save(f"{save_path}/image_{i}.png")

    @torch.no_grad()
    def forward(self, images):
        if isinstance(images, list):
            return [self.process_single_image(image) for image in images]
        else:
            return self.process_single_image(images)

    def process_single_image(self, image):
        # Ensure correct dtype and move to CPU if needed
        if image.dtype == torch.bfloat16:
            image = image.to(torch.float32)
        if image.is_cuda:
            image = image.cpu()

        # test if images are prepared correctly
        # self.save_images(image, './test_images')
        
        image = image.permute(0, 2, 3, 1)  # Convert from NCHW to NHWC format
        image_numpy = image.numpy()
        
        # Accumulate output tensors for batch processing
        output_list = [
            self.run_inference_on_single_image(image_numpy[i:i+1])
            for i in range(image_numpy.shape[0])
        ]
        output_batch_tensor = torch.cat(output_list, dim=0)

        # Ensure correct dtype and move to GPU if needed
        if output_batch_tensor.dtype == torch.float32:
            output_batch_tensor = output_batch_tensor.to(torch.bfloat16)
        if torch.cuda.is_available():
            output_batch_tensor = output_batch_tensor.to('cuda')

        return output_batch_tensor


    def run_inference_on_single_image(self, single_image):
        """Run TFLite inference on a single image and return the output tensor."""
        # Set the input tensor, invoke the vision_tower, and get the output tensor
        self.vision_tower.set_tensor(self.input_details[0]['index'], single_image)
        self.vision_tower.invoke()
        output_data = self.vision_tower.get_tensor(self.output_details[0]['index'])

        # Convert to PyTorch tensor and add batch dimension
        return torch.tensor(output_data).unsqueeze(1)


    @property
    def hidden_size(self):
        # Get output size from TFLite model details
        return self.output_details[0]["shape"][-1]