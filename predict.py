import gradio as gr
import numpy as np
# import spaces # No longer used after removing @spaces.GPU decorator
import torch
import random
from PIL import Image
from diffusers import FluxKontextPipeline
# from diffusers.utils import load_image # Not directly used by Predictor, PIL.Image.open is used.
from cog import BasePredictor, Input, Path
import tempfile # For saving output image

MAX_SEED = np.iinfo(np.int32).max
# DEVICE will be set in Predictor's setup

# --- Cog Predictor Class ---
class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = FluxKontextPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Kontext-dev",
            torch_dtype=torch.bfloat16
        ).to(self.device)

    def predict(
        self,
        input_image: Path = Input(description="Input image for editing. Leave blank for text-to-image.", default=None),
        prompt: str = Input(description="Text prompt for the image generation or editing."),
        seed: int = Input(description="Random seed. Leave blank to randomize.", default=None),
        guidance_scale: float = Input(description="Scale for classifier-free guidance.", ge=1.0, le=10.0, default=2.5),
        num_inference_steps: int = Input(description="Number of inference steps.", ge=1, le=30, default=28),
        # randomize_seed is implicitly handled by seed=None
        # width and height for text-to-image could be added if desired, defaulting to e.g. 1024
        # For image editing, width/height are derived from the input image.
    ) -> Path:
        """Run a single prediction on the model"""
        if seed is None:
            seed = random.randint(0, MAX_SEED)

        generator = torch.Generator(device=self.device).manual_seed(seed)

        pil_image = None
        if input_image:
            pil_image = Image.open(str(input_image)).convert("RGB")

        if pil_image:
            generated_image = self.pipe(
                image=pil_image,
                prompt=prompt,
                guidance_scale=guidance_scale,
                width=pil_image.size[0],
                height=pil_image.size[1],
                num_inference_steps=num_inference_steps,
                generator=generator,
            ).images[0]
        else:
            # Text-to-image: requires default width/height or additional inputs in cog.yaml
            # For now, let's use a common default size like 1024x1024 if no input image
            # This part of the HF app was less defined, as it focused on image editing.
            generated_image = self.pipe(
                prompt=prompt,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
                width=1024, # Default for T2I
                height=1024, # Default for T2I
            ).images[0]

        output_path = Path(tempfile.mkdtemp()) / "output.png"
        generated_image.save(str(output_path))
        return output_path

# --- Gradio UI (for local testing) ---
# Note: The @spaces.GPU decorator is Hugging Face specific and not used by Cog.
# We define a separate infer function for Gradio that can use the loaded pipeline.

# Global pipe for Gradio, loaded if not in Cog environment (e.g. __main__)
gradio_pipe = None
GRADIO_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def ensure_gradio_pipe_loaded():
    global gradio_pipe
    if gradio_pipe is None:
        print("Loading model for Gradio UI...")
        gradio_pipe = FluxKontextPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-Kontext-dev",
            torch_dtype=torch.bfloat16
        ).to(GRADIO_DEVICE)
        print("Model loaded for Gradio.")

# @spaces.GPU # This decorator is for HF Spaces, remove or comment out for local/Cog
def gradio_infer(input_image_pil, prompt_text, seed_val, randomize_seed_bool, guidance_val, steps_val, progress=gr.Progress(track_tqdm=True)):
    ensure_gradio_pipe_loaded()

    if randomize_seed_bool or seed_val < 0: # HF app used 0 as default, here we match randomize_seed checkbox
        seed_val = random.randint(0, MAX_SEED)

    generator = torch.Generator(device=GRADIO_DEVICE).manual_seed(int(seed_val))

    if input_image_pil:
        input_image_pil = input_image_pil.convert("RGB")
        image = gradio_pipe(
            image=input_image_pil,
            prompt=prompt_text,
            guidance_scale=guidance_val,
            width=input_image_pil.size[0],
            height=input_image_pil.size[1],
            num_inference_steps=int(steps_val),
            generator=generator,
        ).images[0]
    else:
        image = gradio_pipe(
            prompt=prompt_text,
            guidance_scale=guidance_val,
            num_inference_steps=int(steps_val),
            generator=generator,
            width=1024, # Default for T2I in Gradio if no image
            height=1024, # Default for T2I in Gradio if no image
        ).images[0]

    return image, seed_val, gr.update(visible=True)

css="""
#col-container {
    margin: 0 auto;
    max-width: 960px;
}
"""

with gr.Blocks(css=css) as demo:
    with gr.Column(elem_id="col-container"):
        gr.Markdown(f"""# FLUX.1 Kontext [dev]
        Image editing and manipulation model guidance-distilled from FLUX.1 Kontext [pro],
        [[blog]](https://bfl.ai/announcements/flux-1-kontext-dev)
        [[model]](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev)
        """)
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Upload the image for editing", type="pil")
                with gr.Row():
                    prompt = gr.Text(
                        label="Prompt",
                        show_label=False,
                        max_lines=1,
                        placeholder="Enter your prompt for editing (e.g., 'Remove glasses', 'Add a hat')",
                        container=False,
                    )
                    run_button = gr.Button("Run", scale=0)
                with gr.Accordion("Advanced Settings", open=False):
                    seed = gr.Slider(
                        label="Seed",
                        minimum=0,
                        maximum=MAX_SEED,
                        step=1,
                        value=0, # HF app.py uses 0 as default, not 42 like in infer func default
                    )
                    randomize_seed = gr.Checkbox(label="Randomize seed", value=True) # HF app.py has this True by default
                    guidance_scale = gr.Slider(
                        label="Guidance Scale",
                        minimum=1.0,
                        maximum=10.0,
                        step=0.1,
                        value=2.5,
                    )
                    steps = gr.Slider(
                        label="Steps",
                        minimum=1,
                        maximum=30, # HF app.py uses 30 max
                        value=28,
                        step=1
                    )
            with gr.Column():
                result = gr.Image(label="Result", show_label=False, interactive=False)
                reuse_button = gr.Button("Reuse this image", visible=False)

        gr.on(
            triggers=[run_button.click, prompt.submit],
            fn=infer,
            inputs=[input_image, prompt, seed, randomize_seed, guidance_scale, steps],
            outputs=[result, seed, reuse_button]
        )
        reuse_button.click(
            fn=lambda image: image,
            inputs=[result],
            outputs=[input_image]
        )

if __name__ == "__main__":
    demo.launch()
