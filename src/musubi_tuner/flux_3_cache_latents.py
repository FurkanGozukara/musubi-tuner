"""Cache FLUX.3 [klein] latents: the FLUX.2 VAE's packed 128-channel latents at 1/16 (FLUX.3 uses that VAE unchanged),
written under the FLUX.3 architecture name. Reference images (editing datasets) are cached as ``latents_control_<i>``
like FLUX.2's."""

import logging
from typing import List

import torch

import musubi_tuner.cache_latents as cache_latents
from musubi_tuner.dataset import config_utils
from musubi_tuner.dataset.architectures import ARCHITECTURE_FLUX_3, ARCHITECTURE_FLUX_3_FULL
from musubi_tuner.dataset.config_utils import BlueprintGenerator, ConfigSanitizer
from musubi_tuner.dataset.image_video_dataset import ItemInfo
from musubi_tuner.flux_2.flux2_utils import load_ae
from musubi_tuner.flux_2_cache_latents import encode_and_save_batch
from musubi_tuner.utils.model_utils import str_to_dtype

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def main():
    parser = cache_latents.setup_parser_common()
    args = parser.parse_args()

    if args.disable_cudnn_backend:
        logger.info("Disabling cuDNN PyTorch backend.")
        torch.backends.cudnn.enabled = False

    device = args.device if hasattr(args, "device") and args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device)

    blueprint_generator = BlueprintGenerator(ConfigSanitizer())
    logger.info(f"Load dataset config from {args.dataset_config}")
    user_config = config_utils.load_user_config(args.dataset_config)
    blueprint = blueprint_generator.generate(user_config, args, architecture=ARCHITECTURE_FLUX_3)
    train_dataset_group = config_utils.generate_dataset_group_by_blueprint(blueprint.dataset_group)
    datasets = train_dataset_group.datasets

    if args.debug_mode is not None:
        cache_latents.show_datasets(datasets, args.debug_mode, args.console_width, args.console_back, args.console_num_images, fps=16)
        return

    assert args.vae is not None, "the FLUX.2 VAE (ae) checkpoint is required"
    logger.info(f"Loading the FLUX.2 VAE from {args.vae}")
    vae_dtype = torch.float32 if not str(args.vae_dtype or "").strip() else str_to_dtype(args.vae_dtype)
    ae = load_ae(args.vae, dtype=vae_dtype, device=device, disable_mmap=True)
    ae.to(device)

    def encode(batch: List[ItemInfo]):
        encode_and_save_batch(ae, batch, ARCHITECTURE_FLUX_3_FULL)

    cache_latents.encode_datasets(datasets, encode, args)


if __name__ == "__main__":
    main()
