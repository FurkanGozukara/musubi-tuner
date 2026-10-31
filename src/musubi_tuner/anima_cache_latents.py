"""Cache Anima latents: the Qwen-Image VAE's normalized 16-channel latents at 1/8 (Wan 2.1 statistics, the latent
format ComfyUI uses for Anima), one frame per image, written under the Anima architecture name. The VAE runs in the
file's dtype (BF16) unless --vae_dtype says otherwise, as kohya's and ComfyUI's encoders do."""

import logging
from typing import List

import torch

import musubi_tuner.cache_latents as cache_latents
from musubi_tuner.anima import anima_utils
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.architectures import ARCHITECTURE_ANIMA
from musubi_tuner.dataset.cache_io import save_latent_cache_anima
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.utils.model_utils import str_to_dtype

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def encode_and_save_batch(vae, batch: List[ItemInfo]):
    contents = []
    for item in batch:
        content = item.content
        content = content[0] if isinstance(content, list) else content  # (H, W, C)
        contents.append(torch.from_numpy(content[..., :3]))
    contents = torch.stack(contents, dim=0).permute(0, 3, 1, 2).unsqueeze(2)  # (B, C, 1, H, W)
    contents = contents / 127.5 - 1.0  # [-1, 1]

    with torch.no_grad():
        latents = vae.encode_pixels_to_latents(contents.to(vae.device, dtype=vae.dtype))  # (B, 16, 1, h, w), normalized

    for b, item in enumerate(batch):
        save_latent_cache_anima(item_info=item, latent=latents[b])


def main():
    parser = cache_latents.setup_parser_common()  # --vae, --vae_dtype
    args = parser.parse_args()

    if args.disable_cudnn_backend:
        logger.info("Disabling cuDNN PyTorch backend.")
        torch.backends.cudnn.enabled = False

    device = args.device if hasattr(args, "device") and args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_ANIMA)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    datasets = train_dataset_group.datasets

    if args.debug_mode is not None:
        cache_latents.show_datasets(datasets, args.debug_mode, args.console_width, args.console_back, args.console_num_images, fps=16)
        return

    assert args.vae is not None, "the Qwen-Image VAE (qwen_image_vae.safetensors) is required"
    vae_dtype = str_to_dtype(args.vae_dtype) if str(args.vae_dtype or "").strip() else None
    logger.info(f"Loading the Qwen-Image VAE from {args.vae}")
    vae = anima_utils.load_vae(args.vae, device=device, dtype=vae_dtype)

    def encode(batch: List[ItemInfo]):
        encode_and_save_batch(vae, batch)

    cache_latents.encode_datasets(datasets, encode, args)


if __name__ == "__main__":
    main()
