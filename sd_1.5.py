from diffusers import AutoencoderKL
from transformers import T5EncoderModel

vae = AutoencoderKL.from_pretrained('runwayml/stable-diffusion-v1-5', subfolder='vae')
text_encoder = T5EncoderModel.from_pretrained("google/t5-large")
from transformers import CLIPTextModel
