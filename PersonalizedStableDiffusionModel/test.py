from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
import torch

model_id = "CompVis/stable-diffusion-v1-4"

pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(
    "cuda:0"
)
pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
from comcat_diffusion import monkeypatch_comcat, tune_comcat_scale

monkeypatch_comcat(pipe.unet, torch.load("output_example/lora_weight_e83_s500.pt"))

pipe.unet.eval().half()
pipe.unet.cuda()
prompt = "a V* dog is reading a book"
tune_comcat_scale('unet', 0.8)
torch.manual_seed(12)
image = pipe(prompt, num_inference_steps=50, guidance_scale=7).images[0]