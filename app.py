from diffusers import AutoPipelineForText2Image
import torch
import gradio as gr

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

def get_model():
    pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
    return pipe

model = get_model()
model.to(device)

if device=="cpu":
    model.enable_model_cpu_offload()


def return_image(text,inf_steps,gScale):
    image = model(prompt=text, num_inference_steps=inf_steps, guidance_scale=gScale).images[0]
    return image

piece = gr.Interface(
    return_image,
    [
        gr.Textbox(label = "Enter a prompt to describe the image you want to be generated!",placeholder = "A cinematic shot of a baby racoon wearing an intricate italian priest robe"),
        gr.Slider(minimum = 1, maximum = 50, step = 1,label = "number of inference steps"),
        gr.Slider(minimum = 0, maximum = 1, step = 0.1, label = "guidance scale"),
    ],
    outputs = [gr.Image(label = "Image Output")],
    title = "Stable Diffusion XL Turbo Model Demo"
)
piece.launch()
