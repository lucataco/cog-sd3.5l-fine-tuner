
from cog import BaseModel, Input, Path, Secret  # pyright: ignore
import os
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
import sys
sys.path.append("ai-toolkit")
from submodule_patches import patch_submodules
patch_submodules()

import sys
import time
import torch
import shutil
import subprocess
from typing import Optional, OrderedDict
from zipfile import ZipFile, is_zipfile
from extensions_built_in.sd_trainer.SDTrainer import SDTrainer
from huggingface_hub import HfApi
from jobs import BaseJob
from toolkit.config import get_config
from wandb_client import WeightsAndBiasesClient, logout_wandb
from layer_match import match_layers_to_optimize, available_layers_to_optimize


MODEL_URL = "https://weights.replicate.delivery/default/stabilityai/stable-diffusion-3.5-large/full.tar"
JOB_NAME = "sd3.5_train_replicate"
WEIGHTS_PATH = Path("./stable-diffusion-3.5-large")
INPUT_DIR = Path("input_images")
OUTPUT_DIR = Path("output")
JOB_DIR = OUTPUT_DIR / JOB_NAME

print(f"Environment language: {os.environ.get('LANG', 'Not set')}")
os.environ["LANG"] = "en_US.UTF-8"
print(f"Updated environment language: {os.environ.get('LANG', 'Not set')}")


class CustomSDTrainer(SDTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen_samples = set()
        self.wandb: WeightsAndBiasesClient | None = None

    def hook_train_loop(self, batch):
        loss_dict = super().hook_train_loop(batch)
        if self.wandb:
            self.wandb.log_loss(loss_dict, self.step_num)
        return loss_dict

    def sample(self, step=None, is_first=False):
        super().sample(step=step, is_first=is_first)
        output_dir = JOB_DIR / "samples"
        all_samples = set([p.name for p in output_dir.glob("*.jpg")])
        new_samples = all_samples - self.seen_samples
        if self.wandb:
            image_paths = [output_dir / p for p in sorted(new_samples)]
            self.wandb.log_samples(image_paths, step)
        self.seen_samples = all_samples

    def post_save_hook(self, save_path):
        super().post_save_hook(save_path)
        # final lora path
        lora_path = JOB_DIR / f"{JOB_NAME}.safetensors"
        if not lora_path.exists():
            # intermediate saved weights
            lora_path = sorted(JOB_DIR.glob("*.safetensors"))[-1]
        if self.wandb:
            print(f"Saving weights to W&B: {lora_path.name}")
            self.wandb.save_weights(lora_path)


class CustomJob(BaseJob):
    def __init__(
        self, config: OrderedDict, wandb_client: WeightsAndBiasesClient | None
    ):
        super().__init__(config)
        self.device = self.get_conf("device", "cpu")
        self.process_dict = {"custom_sd_trainer": CustomSDTrainer}
        self.load_processes(self.process_dict)
        for process in self.process:
            process.wandb = wandb_client

    def run(self):
        super().run()
        # Keeping this for backwards compatibility
        print(
            f"Running  {len(self.process)} process{'' if len(self.process) == 1 else 'es'}"
        )
        for process in self.process:
            process.run()


class TrainingOutput(BaseModel):
    weights: Path


def train(
    input_images: Path = Input(
        description="A zip file containing the images that will be used for training. We recommend a minimum of 10 images. If you include captions, include them as one .txt file per image, ex: my-photo.jpg should have a caption file named my-photo.txt.",
        default=None,
    ),
    trigger_word: str = Input(
        description="The trigger word refers to the object, style or concept you are training on. Pick a string that isn’t a real word, like TOK or something related to what’s being trained, like CYBRPNK. The trigger word you specify here will be associated with all images during training. Then when you use your LoRA, you can include the trigger word in prompts to help activate the LoRA.",
        default="TOK",
    ),
    steps: int = Input(
        description="Number of training steps. Recommended range 700-4000",
        ge=3,
        le=6000,
        default=700,
    ),
    learning_rate: float = Input(
        description="Learning rate, if you’re new to training you probably don’t need to change this.",
        default=4e-4,
    ),
    batch_size: int = Input(
        description="Batch size, you can leave this as 1", default=1
    ),
    resolution: str = Input(
        description="Image resolutions for training", default="512,768"
    ),
    lora_rank: int = Input(
        description="Higher ranks take longer to train but can capture more complex features. Caption quality is more important for higher ranks.",
        default=16,
        ge=1,
        le=128,
    ),
    caption_dropout_rate: float = Input(
        description="Advanced setting. Determines how often a caption is ignored. 0.05 means for 5% of all steps an image will be used without its caption. 0 means always use captions, while 1 means never use them. Dropping captions helps capture more details of an image, and can prevent over-fitting words with specific image elements. Try higher values when training a style.",
        default=0.05,
        ge=0,
        le=1,
    ),
    optimizer: str = Input(
        description="Optimizer to use for training. Supports: prodigy, adam8bit, adamw8bit, lion8bit, adam, adamw, lion, adagrad, adafactor.",
        default="adamw8bit",
    ),
    cache_latents_to_disk: bool = Input(
        description="Use this if you have lots of input images and you hit out of memory errors",
        default=False,
    ),
    layers_to_optimize_regex: str = Input(
        description="Regular expression to match specific layers to optimize. Optimizing fewer layers results in shorter training times, but can also result in a weaker LoRA. For example, To target layers 7, 12, 16, 20 which seems to create good likeness with faster training (as discovered by lux in the Ostris discord, inspired by The Last Ben), use `transformer.single_transformer_blocks.(7|12|16|20).proj_out`.",
        default=None,
    ),
    hf_repo_id: str = Input(
        description="Hugging Face repository ID, if you'd like to upload the trained LoRA to Hugging Face. For example, lucataco/SD3.5-Large-Violeta. If the given repo does not exist, a new public repo will be created.",
        default=None,
    ),
    hf_token: Secret = Input(
        description="Hugging Face token, if you'd like to upload the trained LoRA to Hugging Face.",
        default=None,
    ),
    wandb_api_key: Secret = Input(
        description="Weights and Biases API key, if you'd like to log training progress to W&B.",
        default=None,
    ),
    wandb_project: str = Input(
        description="Weights and Biases project name. Only applicable if wandb_api_key is set.",
        default=JOB_NAME,
    ),
    wandb_run: str = Input(
        description="Weights and Biases run name. Only applicable if wandb_api_key is set.",
        default=None,
    ),
    wandb_entity: str = Input(
        description="Weights and Biases entity name. Only applicable if wandb_api_key is set.",
        default=None,
    ),
    wandb_sample_interval: int = Input(
        description="Step interval for sampling output images that are logged to W&B. Only applicable if wandb_api_key is set.",
        default=100,
        ge=1,
    ),
    wandb_sample_prompts: str = Input(
        description="Newline-separated list of prompts to use when logging samples to W&B. Only applicable if wandb_api_key is set.",
        default=None,
    ),
    wandb_save_interval: int = Input(
        description="Step interval for saving intermediate LoRA weights to W&B. Only applicable if wandb_api_key is set.",
        default=100,
        ge=1,
    ),
) -> TrainingOutput:
    clean_up()
    output_path = "/tmp/trained_model.tar"

    if not input_images:
        raise ValueError("input_images must be provided")

    layers_to_optimize = None
    if layers_to_optimize_regex:
        layers_to_optimize = match_layers_to_optimize(layers_to_optimize_regex)
        if not layers_to_optimize:
            raise ValueError(
                f"The regex '{layers_to_optimize_regex}' didn't match any layers. These layers can be optimized:\n"
                + "\n".join(available_layers_to_optimize)
            )

    sample_prompts = []
    if wandb_sample_prompts:
        sample_prompts = [p.strip() for p in wandb_sample_prompts.split("\n")]

    train_config = OrderedDict(
        {
            "job": "custom_job",
            "config": {
                "name": JOB_NAME,
                "process": [
                    {
                        "type": "custom_sd_trainer",
                        "training_folder": str(OUTPUT_DIR),
                        "device": "cuda:0",
                        "trigger_word": trigger_word,
                        "network": {
                            "type": "lora",
                            "linear": lora_rank,
                            "linear_alpha": lora_rank,
                        },
                        "save": {
                            "dtype": "float16",
                            "save_every": (
                                wandb_save_interval if wandb_api_key else steps + 1
                            ),
                            "max_step_saves_to_keep": 1,
                        },
                        "datasets": [
                            {
                                "folder_path": str(INPUT_DIR),
                                "caption_ext": "txt",
                                "caption_dropout_rate": caption_dropout_rate,
                                "shuffle_tokens": False,
                                # TODO: Do we need to cache to disk? It's faster not to.
                                "cache_latents_to_disk": cache_latents_to_disk,
                                "cache_latents": True,
                                "resolution": [
                                    int(res) for res in resolution.split(",")
                                ],
                            }
                        ],
                        "train": {
                            "batch_size": batch_size,
                            "steps": steps,
                            "gradient_accumulation_steps": 1,
                            "train_unet": True,
                            "train_text_encoder": False,
                            "content_or_style": "balanced",
                            "gradient_checkpointing": True,
                            "noise_scheduler": "flowmatch",
                            "optimizer": optimizer,
                            "lr": learning_rate,
                            "ema_config": {"use_ema": True, "ema_decay": 0.99},
                            "dtype": "bf16",
                        },
                        "model": {
                            "name_or_path": str(WEIGHTS_PATH),
                            "is_v3": True,
                            "quantize": True,
                        },
                        "sample": {
                            "sampler": "flowmatch",
                            "sample_every": (
                                wandb_sample_interval
                                if wandb_api_key and sample_prompts
                                else steps + 1
                            ),
                            "width": 1024,
                            "height": 1024,
                            "prompts": sample_prompts,
                            "neg": "",
                            "seed": 42,
                            "walk_seed": True,
                            "guidance_scale": 3.5,
                            "sample_steps": 28,
                        },
                    }
                ],
            },
            "meta": {"name": "[name]", "version": "1.0"},
        }
    )

    if layers_to_optimize:
        train_config["config"]["process"][0]["network"]["network_kwargs"] = {
            "only_if_contains": layers_to_optimize
        }

    wandb_client = None
    if wandb_api_key:
        wandb_config = {
            "trigger_word": trigger_word,
            "steps": steps,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "resolution": resolution,
            "lora_rank": lora_rank,
            "caption_dropout_rate": caption_dropout_rate,
            "optimizer": optimizer,
        }
        wandb_client = WeightsAndBiasesClient(
            api_key=wandb_api_key.get_secret_value(),
            config=wandb_config,
            sample_prompts=sample_prompts,
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_run,
        )

    download_weights()
    extract_zip(input_images, INPUT_DIR)
    torch.cuda.empty_cache()

    if not trigger_word:
        del train_config["config"]["process"][0]["trigger_word"]

    print("Starting train job")
    job = CustomJob(get_config(train_config, name=None), wandb_client)
    job.run()

    if wandb_client:
        wandb_client.finish()

    job.cleanup()

    lora_file = JOB_DIR / f"{JOB_NAME}.safetensors"
    lora_file.rename(JOB_DIR / "lora.safetensors")

    samples_dir = JOB_DIR / "samples"
    if samples_dir.exists():
        shutil.rmtree(samples_dir)

    # Remove any intermediate lora paths
    lora_paths = JOB_DIR.glob("*.safetensors")
    for path in lora_paths:
        if path.name != "lora.safetensors":
            path.unlink()

    # Optimizer is used to continue training, not needed in output
    optimizer_file = JOB_DIR / "optimizer.pt"
    if optimizer_file.exists():
        optimizer_file.unlink()

    # Copy generated captions to the output tar
    # But do not upload publicly to HF
    captions_dir = JOB_DIR / "captions"
    captions_dir.mkdir(exist_ok=True)
    for caption_file in INPUT_DIR.glob("*.txt"):
        shutil.copy(caption_file, captions_dir)

    os.system(f"tar -cvf {output_path} {JOB_DIR}")

    if hf_token is not None and hf_repo_id is not None:
        if captions_dir.exists():
            shutil.rmtree(captions_dir)

        try:
            handle_hf_readme(hf_repo_id, trigger_word)
            print(f"Uploading to Hugging Face: {hf_repo_id}")
            api = HfApi()

            repo_url = api.create_repo(
                hf_repo_id,
                private=False,
                exist_ok=True,
                token=hf_token.get_secret_value(),
            )

            print(f"HF Repo URL: {repo_url}")

            api.upload_folder(
                repo_id=hf_repo_id,
                folder_path=JOB_DIR,
                repo_type="model",
                use_auth_token=hf_token.get_secret_value(),
            )
        except Exception as e:
            print(f"Error uploading to Hugging Face: {str(e)}")

    return TrainingOutput(weights=Path(output_path))


def handle_hf_readme(hf_repo_id: str, trigger_word: Optional[str]):
    readme_path = JOB_DIR / "README.md"
    license_path = Path("lora-license.md")
    shutil.copy(license_path, readme_path)

    content = readme_path.read_text()
    content = content.replace("[hf_repo_id]", hf_repo_id)

    repo_parts = hf_repo_id.split("/")
    if len(repo_parts) > 1:
        title = repo_parts[1].replace("-", " ").title()
        content = content.replace("[title]", title)
    else:
        content = content.replace("[title]", hf_repo_id)

    if trigger_word:
        content = content.replace(
            "[trigger_section]",
            f"\n## Trigger words\nYou should use `{trigger_word}` to trigger the image generation.\n",
        )
        content = content.replace(
            "[instance_prompt]", f"instance_prompt: {trigger_word}"
        )
    else:
        content = content.replace("[trigger_section]", "")
        content = content.replace("[instance_prompt]", "")

    print(content)

    readme_path.write_text(content)


def extract_zip(input_images: Path, input_dir: Path):
    if not is_zipfile(input_images):
        raise ValueError("input_images must be a zip file")

    input_dir.mkdir(parents=True, exist_ok=True)
    image_count = 0
    with ZipFile(input_images, "r") as zip_ref:
        for file_info in zip_ref.infolist():
            if not file_info.filename.startswith(
                "__MACOSX/"
            ) and not file_info.filename.startswith("._"):
                zip_ref.extract(file_info, input_dir)
                image_count += 1

    print(f"Extracted {image_count} files from zip to {input_dir}")


def clean_up():
    logout_wandb()

    if INPUT_DIR.exists():
        shutil.rmtree(INPUT_DIR)

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)


def download_weights():
    if not WEIGHTS_PATH.exists():
        t1 = time.time()
        subprocess.check_output(
            [
                "pget",
                "-x",
                MODEL_URL,
                str(WEIGHTS_PATH),
            ]
        )
        t2 = time.time()
        print(f"Downloaded base weights in {t2 - t1} seconds")
