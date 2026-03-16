"""Utils for evaluating OpenVLA or fine-tuned OpenVLA policies."""

import filecmp
import gc
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import json_numpy
import numpy as np
import requests
import tensorflow as tf
import torch
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

# Apply JSON numpy patch for serialization
json_numpy.patch()

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import DiffusionActionHead, L1RegressionActionHead
from prismatic.models.film_vit_wrapper import FiLMedPrismaticVisionBackbone
from prismatic.models.projectors import NoisyActionProjector, ProprioProjector
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
)
from prismatic.vla.datasets.rlds.utils.data_utils import NormalizationType

from .vla_cache_utils import find_static_patches, task_relevant_selection, get_layer_mask_schedule, token_attention_merge, compute_preprune_mask
from .analysis_utils import AttentionHookCapture, FrameStats

from transformers import DynamicCache

# Initialize important constants
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
OPENVLA_IMAGE_SIZE = 224  # Standard image size expected by OpenVLA

# Configure NumPy print settings
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})


def model_is_on_hf_hub(model_path: str) -> bool:
    """Checks whether a model path points to a model on Hugging Face Hub."""
    # If the API call below runs without error, the model is on the hub
    try:
        HfApi().model_info(model_path)
        return True
    except Exception:
        return False


def update_auto_map(pretrained_checkpoint: str) -> None:
    """
    Update the AutoMap configuration in the checkpoint config.json file.

    This loads the config.json file inside the checkpoint directory and overwrites
    the AutoConfig and AutoModelForVision2Seq fields to use OpenVLA-specific classes.

    Args:
        pretrained_checkpoint: Path to the checkpoint directory
    """
    if not os.path.isdir(pretrained_checkpoint):
        return

    config_path = os.path.join(pretrained_checkpoint, "config.json")
    if not os.path.exists(config_path):
        print(f"Warning: No config.json found at {config_path}")
        return

    # Create timestamped backup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(pretrained_checkpoint, f"config.json.back.{timestamp}")
    shutil.copy2(config_path, backup_path)
    print(f"Created backup of original config at: {os.path.abspath(backup_path)}")

    # Read and update the config
    with open(config_path, "r") as f:
        config = json.load(f)

    config["auto_map"] = {
        "AutoConfig": "configuration_prismatic.OpenVLAConfig",
        "AutoModelForVision2Seq": "modeling_prismatic.OpenVLAForActionPrediction",
    }

    # Write back the updated config
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Updated config.json at: {os.path.abspath(config_path)}")
    print("Changes made:")
    print('  - Set AutoConfig to "configuration_prismatic.OpenVLAConfig"')
    print('  - Set AutoModelForVision2Seq to "modeling_prismatic.OpenVLAForActionPrediction"')


def check_identical_files(path1: Union[str, Path], path2: Union[str, Path]) -> bool:
    """
    Check if two files are identical in content.

    Args:
        path1: Path to the first file
        path2: Path to the second file

    Returns:
        bool: True if files are identical, False otherwise
    """
    path1, path2 = Path(path1), Path(path2)

    # First check if file sizes match
    if path1.stat().st_size != path2.stat().st_size:
        return False

    # Check if contents match
    return filecmp.cmp(path1, path2, shallow=False)


def _handle_file_sync(curr_filepath: str, checkpoint_filepath: str, file_type: str) -> None:
    """
    Handle syncing of files between current directory and checkpoint.

    Creates backups if files exist but differ, and copies current versions to checkpoint.

    Args:
        curr_filepath: Path to the current file version
        checkpoint_filepath: Path where the file should be in the checkpoint
        file_type: Description of the file type for logging
    """
    if os.path.exists(checkpoint_filepath):
        # Check if existing files are identical
        match = check_identical_files(curr_filepath, checkpoint_filepath)

        if not match:
            print(
                "\n------------------------------------------------------------------------------------------------\n"
                f"Found mismatch between:\n"
                f"Current:   {curr_filepath}\n"
                f"Checkpoint: {checkpoint_filepath}\n"
            )

            # Create timestamped backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = f"{checkpoint_filepath}.back.{timestamp}"
            shutil.copy2(checkpoint_filepath, backup_path)
            print(f"Created backup of original checkpoint file at: {os.path.abspath(backup_path)}")

            # Copy current version to checkpoint directory
            shutil.copy2(curr_filepath, checkpoint_filepath)
            print(f"Copied current version to checkpoint at: {os.path.abspath(checkpoint_filepath)}")
            print(
                f"Changes complete. The checkpoint will now use the current version of {file_type}"
                "\n------------------------------------------------------------------------------------------------\n"
            )
    else:
        # If file doesn't exist in checkpoint directory, copy it
        shutil.copy2(curr_filepath, checkpoint_filepath)
        print(
            "\n------------------------------------------------------------------------------------------------\n"
            f"No {file_type} found in checkpoint directory.\n"
            f"Copied current version from: {curr_filepath}\n"
            f"To checkpoint location: {os.path.abspath(checkpoint_filepath)}"
            "\n------------------------------------------------------------------------------------------------\n"
        )


def check_model_logic_mismatch(pretrained_checkpoint: str) -> None:
    """
    Check and sync model logic files between current code and checkpoint.

    Handles the relationship between current and checkpoint versions of both
    modeling_prismatic.py and configuration_prismatic.py:
    - If checkpoint file exists and differs: creates backup and copies current version
    - If checkpoint file doesn't exist: copies current version

    Args:
        pretrained_checkpoint: Path to the checkpoint directory
    """
    if not os.path.isdir(pretrained_checkpoint):
        return

    # Find current files
    curr_files = {"modeling_prismatic.py": None, "configuration_prismatic.py": None}

    for root, _, files in os.walk("./prismatic/"):
        for filename in curr_files.keys():
            if filename in files and curr_files[filename] is None:
                curr_files[filename] = os.path.join(root, filename)

    # Check and handle each file
    for filename, curr_filepath in curr_files.items():
        if curr_filepath is None:
            print(f"WARNING: `{filename}` is not found anywhere in the current directory.")
            continue

        checkpoint_filepath = os.path.join(pretrained_checkpoint, filename)
        _handle_file_sync(curr_filepath, checkpoint_filepath, filename)


def find_checkpoint_file(pretrained_checkpoint: str, file_pattern: str) -> str:
    """
    Find a specific checkpoint file matching a pattern.

    Args:
        pretrained_checkpoint: Path to the checkpoint directory
        file_pattern: String pattern to match in filenames

    Returns:
        str: Path to the matching checkpoint file

    Raises:
        AssertionError: If no files or multiple files match the pattern
    """
    assert os.path.isdir(pretrained_checkpoint), f"Checkpoint path must be a directory: {pretrained_checkpoint}"

    checkpoint_files = []
    for filename in os.listdir(pretrained_checkpoint):
        if file_pattern in filename and "checkpoint" in filename:
            full_path = os.path.join(pretrained_checkpoint, filename)
            checkpoint_files.append(full_path)

    assert len(checkpoint_files) == 1, (
        f"Expected exactly 1 {file_pattern} checkpoint but found {len(checkpoint_files)} in directory: {pretrained_checkpoint}"
    )

    return checkpoint_files[0]


def load_component_state_dict(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """
    Load a component's state dict from checkpoint and handle DDP prefix if present.

    Args:
        checkpoint_path: Path to the checkpoint file

    Returns:
        Dict: The processed state dictionary for loading
    """
    state_dict = torch.load(checkpoint_path, weights_only=True)

    # If the component was trained with DDP, elements in the state dict have prefix "module." which we must remove
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v

    return new_state_dict


def get_vla(cfg: Any) -> torch.nn.Module:
    """
    Load and initialize the VLA model from checkpoint.

    Args:
        cfg: Configuration object

    Returns:
        torch.nn.Module: The initialized VLA model
    """
    print("Instantiating pretrained VLA policy...")

    # If loading a locally stored pretrained checkpoint, check whether config or model files
    # need to be synced so that any changes the user makes to the VLA modeling code will
    # actually go into effect
    # If loading a pretrained checkpoint from Hugging Face Hub, we just assume that the policy
    # will be used as is, with its original modeling logic
    if not model_is_on_hf_hub(cfg.pretrained_checkpoint):
        # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

        # Update config.json and sync model files
        update_auto_map(cfg.pretrained_checkpoint)
        check_model_logic_mismatch(cfg.pretrained_checkpoint)

    
    # Load the model
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.pretrained_checkpoint,
        # attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        load_in_8bit=cfg.load_in_8bit,
        load_in_4bit=cfg.load_in_4bit,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # If using FiLM, wrap the vision backbone to allow for infusion of language inputs
    if cfg.use_film:
        vla = _apply_film_to_vla(vla, cfg)

    # Set number of images in model input
    vla.vision_backbone.set_num_images_in_input(cfg.num_images_in_input)

    vla.eval()

    # Move model to device if not using quantization
    if not cfg.load_in_8bit and not cfg.load_in_4bit:
        vla = vla.to(DEVICE)

    # Load dataset stats for action normalization
    _load_dataset_stats(vla, cfg.pretrained_checkpoint)

    return vla


def _apply_film_to_vla(vla: torch.nn.Module, cfg: Any) -> torch.nn.Module:
    """
    Apply FiLM (Feature-wise Linear Modulation) to the VLA vision backbone.

    Args:
        vla: The VLA model
        cfg: Configuration object with model parameters

    Returns:
        torch.nn.Module: VLA model with FiLM applied
    """
    from peft import LoraConfig, get_peft_model

    # Apply LoRA configuration
    lora_config = LoraConfig(
        r=cfg.lora_rank,
        lora_alpha=min(cfg.lora_rank, 16),
        lora_dropout=0.0,
        target_modules="all-linear",
        init_lora_weights="gaussian",
    )
    vla = get_peft_model(vla, lora_config)

    # Create and apply FiLMed vision backbone
    new_vision_backbone = FiLMedPrismaticVisionBackbone(
        vision_backbone=vla.vision_backbone, llm_dim=vla.llm_dim,
    )
    vla.model.vision_backbone = new_vision_backbone

    # Load vision backbone checkpoint
    checkpoint_path = find_checkpoint_file(cfg.pretrained_checkpoint, "vision_backbone")
    state_dict = torch.load(checkpoint_path, weights_only=True)
    vla.model.vision_backbone.load_state_dict(state_dict)

    # Use the model component instead of wrapper and convert to bfloat16
    vla = vla.model
    vla.vision_backbone = vla.vision_backbone.to(torch.bfloat16)

    return vla


def _load_dataset_stats(vla: torch.nn.Module, checkpoint_path: str) -> None:
    """
    Load dataset statistics used during training for action normalization.

    Args:
        vla: The VLA model
        checkpoint_path: Path to the checkpoint directory
    """
    if model_is_on_hf_hub(checkpoint_path):
        # Download dataset stats directly from HF Hub
        dataset_statistics_path = hf_hub_download(
            repo_id=checkpoint_path,
            filename="dataset_statistics.json",
        )
    else:
        dataset_statistics_path = os.path.join(checkpoint_path, "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        vla.norm_stats = norm_stats
    else:
        print(
            "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
            "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
            "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
        )


def get_processor(cfg: Any) -> AutoProcessor:
    """
    Get the VLA model's Hugging Face processor.

    Args:
        cfg: Configuration object with model parameters

    Returns:
        AutoProcessor: The model's processor
    """
    return AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True)


def get_proprio_projector(cfg: Any, llm_dim: int, proprio_dim: int) -> ProprioProjector:
    """
    Get proprioception projector for the VLA model.

    Args:
        cfg: Configuration object with model parameters
        llm_dim: Dimension of the language model
        proprio_dim: Dimension of proprioception data

    Returns:
        ProprioProjector: The initialized proprio projector
    """
    # Initialize projector and move to device
    proprio_projector = ProprioProjector(
        llm_dim=llm_dim,
        proprio_dim=proprio_dim,
    ).to(DEVICE)
    proprio_projector = proprio_projector.to(torch.bfloat16).to(DEVICE)
    proprio_projector.eval()

    # Find and load checkpoint (may be on Hugging Face Hub or stored locally)
    if model_is_on_hf_hub(cfg.pretrained_checkpoint):
        model_path_to_proprio_projector_name = {
            "moojink/openvla-7b-oft-finetuned-libero-spatial": "proprio_projector--150000_checkpoint.pt",
            "moojink/openvla-7b-oft-finetuned-libero-object": "proprio_projector--150000_checkpoint.pt",
            "moojink/openvla-7b-oft-finetuned-libero-goal": "proprio_projector--50000_checkpoint.pt",
            "moojink/openvla-7b-oft-finetuned-libero-10": "proprio_projector--150000_checkpoint.pt",
            "moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10": "proprio_projector--300000_checkpoint.pt",
        }
        if cfg.pretrained_checkpoint not in model_path_to_proprio_projector_name.keys():
            raise ValueError("Unsupported HF Hub pretrained checkpoint found!")
        # Download proprio projector directly from HF Hub
        proprio_projector_path = hf_hub_download(
            repo_id=cfg.pretrained_checkpoint, filename=model_path_to_proprio_projector_name[cfg.pretrained_checkpoint]
        )
        state_dict = load_component_state_dict(proprio_projector_path)
        proprio_projector.load_state_dict(state_dict)
    else:
        checkpoint_path = find_checkpoint_file(cfg.pretrained_checkpoint, "proprio_projector")
        state_dict = load_component_state_dict(checkpoint_path)
        proprio_projector.load_state_dict(state_dict)

    return proprio_projector


def get_noisy_action_projector(cfg: Any, llm_dim: int) -> NoisyActionProjector:
    """
    Get noisy action projector for diffusion-based action prediction.

    Args:
        cfg: Configuration object with model parameters
        llm_dim: Dimension of the language model

    Returns:
        NoisyActionProjector: The initialized noisy action projector
    """
    # Initialize projector and move to device
    noisy_action_projector = NoisyActionProjector(
        llm_dim=llm_dim,
    ).to(DEVICE)
    noisy_action_projector = noisy_action_projector.to(torch.bfloat16).to(DEVICE)
    noisy_action_projector.eval()

    # Find and load checkpoint
    checkpoint_path = find_checkpoint_file(cfg.pretrained_checkpoint, "noisy_action_projector")
    state_dict = load_component_state_dict(checkpoint_path)
    noisy_action_projector.load_state_dict(state_dict)

    return noisy_action_projector


def get_action_head(cfg: Any, llm_dim: int) -> Union[L1RegressionActionHead, DiffusionActionHead]:
    """
    Get action head for continuous value prediction.

    Args:
        cfg: Configuration object with model parameters
        llm_dim: Dimension of the language model

    Returns:
        Union[L1RegressionActionHead, DiffusionActionHead]: The initialized action head

    Raises:
        AssertionError: If both L1 regression and diffusion are specified
    """
    assert not (cfg.use_l1_regression and cfg.use_diffusion), "Cannot use both L1 regression and diffusion action head!"

    # Initialize appropriate action head based on configuration
    if cfg.use_l1_regression:
        action_head = L1RegressionActionHead(input_dim=llm_dim, hidden_dim=llm_dim, action_dim=ACTION_DIM)
    elif cfg.use_diffusion:
        action_head = DiffusionActionHead(
            input_dim=llm_dim, hidden_dim=llm_dim, action_dim=ACTION_DIM, num_diffusion_steps_train=cfg.num_diffusion_steps_train
        )
        # Set number of diffusion steps for inference
        action_head.noise_scheduler.set_timesteps(cfg.num_diffusion_steps_inference)
    else:
        raise ValueError("Either use_l1_regression or use_diffusion must be True")

    action_head = action_head.to(torch.bfloat16).to(DEVICE)
    action_head.eval()

    # Find and load checkpoint (may be on Hugging Face Hub or stored locally)
    if model_is_on_hf_hub(cfg.pretrained_checkpoint):
        model_path_to_action_head_name = {
            "moojink/openvla-7b-oft-finetuned-libero-spatial": "action_head--150000_checkpoint.pt",
            "moojink/openvla-7b-oft-finetuned-libero-object": "action_head--150000_checkpoint.pt",
            "moojink/openvla-7b-oft-finetuned-libero-goal": "action_head--50000_checkpoint.pt",
            "moojink/openvla-7b-oft-finetuned-libero-10": "action_head--150000_checkpoint.pt",
            "moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10": "action_head--300000_checkpoint.pt",
        }
        if cfg.pretrained_checkpoint not in model_path_to_action_head_name.keys():
            raise ValueError("Unsupported HF Hub pretrained checkpoint found!")
        # Download proprio projector directly from HF Hub
        action_head_path = hf_hub_download(
            repo_id=cfg.pretrained_checkpoint, filename=model_path_to_action_head_name[cfg.pretrained_checkpoint]
        )
        state_dict = load_component_state_dict(action_head_path)
        action_head.load_state_dict(state_dict)
    else:
        checkpoint_path = find_checkpoint_file(cfg.pretrained_checkpoint, "action_head")
        state_dict = load_component_state_dict(checkpoint_path)
        action_head.load_state_dict(state_dict)

    return action_head


def resize_image_for_policy(img: np.ndarray, resize_size: Union[int, Tuple[int, int]]) -> np.ndarray:
    """
    Resize an image to match the policy's expected input size.

    Uses the same resizing scheme as in the training data pipeline for distribution matching.

    Args:
        img: Numpy array containing the image
        resize_size: Target size as int (square) or (height, width) tuple

    Returns:
        np.ndarray: The resized image
    """
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)

    # Resize using the same pipeline as in RLDS dataset builder
    img = tf.image.encode_jpeg(img)  # Encode as JPEG
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)

    return img.numpy()


def crop_and_resize(image: tf.Tensor, crop_scale: float, batch_size: int) -> tf.Tensor:
    """
    Center-crop an image and resize it back to original dimensions.

    Uses the same logic as in the training data pipeline for distribution matching.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) with values in [0,1]
        crop_scale: Area of center crop relative to original image
        batch_size: Batch size

    Returns:
        tf.Tensor: The cropped and resized image
    """
    # Handle 3D inputs by adding batch dimension if needed
    assert image.shape.ndims in (3, 4), "Image must be 3D or 4D tensor"
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Calculate crop dimensions (note: we use sqrt(crop_scale) for h/w)
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Create bounding box for the crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Apply crop and resize
    image = tf.image.crop_and_resize(
        image, bounding_boxes, tf.range(batch_size), (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE)
    )

    # Remove batch dimension if it was added
    if expanded_dims:
        image = image[0]

    return image


def center_crop_image(image: Union[np.ndarray, Image.Image]) -> Image.Image:
    """
    Center crop an image to match training data distribution.

    Args:
        image: Input image (PIL or numpy array)

    Returns:
        Image.Image: Cropped PIL Image
    """
    batch_size = 1
    crop_scale = 0.9

    # Convert to TF Tensor if needed
    if not isinstance(image, tf.Tensor):
        image = tf.convert_to_tensor(np.array(image))

    orig_dtype = image.dtype

    # Convert to float32 in range [0,1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Apply center crop and resize
    image = crop_and_resize(image, crop_scale, batch_size)

    # Convert back to original data type
    image = tf.clip_by_value(image, 0, 1)
    image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

    # Convert to PIL Image
    return Image.fromarray(image.numpy()).convert("RGB")


def check_image_format(image: Any) -> None:
    """
    Validate input image format.

    Args:
        image: Image to check

    Raises:
        AssertionError: If image format is invalid
    """
    is_numpy_array = isinstance(image, np.ndarray)
    has_correct_shape = len(image.shape) == 3 and image.shape[-1] == 3
    has_correct_dtype = image.dtype == np.uint8

    assert is_numpy_array and has_correct_shape and has_correct_dtype, (
        "Incorrect image format detected! Make sure that the input image is a "
        "numpy array with shape (H, W, 3) and dtype np.uint8!"
    )


def normalize_proprio(proprio: np.ndarray, norm_stats: Dict[str, Any]) -> np.ndarray:
    """
    Normalize proprioception data to match training distribution.

    Args:
        proprio: Raw proprioception data
        norm_stats: Normalization statistics

    Returns:
        np.ndarray: Normalized proprioception data
    """
    if ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS:
        mask = norm_stats.get("mask", np.ones_like(norm_stats["min"], dtype=bool))
        proprio_high, proprio_low = np.array(norm_stats["max"]), np.array(norm_stats["min"])
    elif ACTION_PROPRIO_NORMALIZATION_TYPE == NormalizationType.BOUNDS_Q99:
        mask = norm_stats.get("mask", np.ones_like(norm_stats["q01"], dtype=bool))
        proprio_high, proprio_low = np.array(norm_stats["q99"]), np.array(norm_stats["q01"])
    else:
        raise ValueError("Unsupported action/proprio normalization type detected!")

    normalized_proprio = np.clip(
        np.where(
            mask,
            2 * (proprio - proprio_low) / (proprio_high - proprio_low + 1e-8) - 1,
            proprio,
        ),
        a_min=-1.0,
        a_max=1.0,
    )

    return normalized_proprio


def prepare_images_for_vla(images: List[np.ndarray], cfg: Any) -> List[Image.Image]:
    """
    Prepare images for VLA input by resizing and cropping as needed.

    Args:
        images: List of input images as numpy arrays
        cfg: Configuration object with parameters

    Returns:
        List[Image.Image]: Processed images ready for the model
    """
    processed_images = []

    for image in images:
        # Validate format
        check_image_format(image)

        # Resize if needed
        if image.shape != (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE, 3):
            image = resize_image_for_policy(image, OPENVLA_IMAGE_SIZE)

        # Convert to PIL image
        pil_image = Image.fromarray(image).convert("RGB")

        # Apply center crop if configured
        if cfg.center_crop:
            pil_image = center_crop_image(pil_image)

        processed_images.append(pil_image)

    return processed_images


def _kvcache_to_device(cache, device):
    """Move a HuggingFace DynamicCache (or legacy tuple) to the target device, in-place."""
    if cache is None:
        return
    if hasattr(cache, 'key_cache') and hasattr(cache, 'value_cache'):
        cache.key_cache  = [t.to(device) if isinstance(t, torch.Tensor) else t for t in cache.key_cache]
        cache.value_cache = [t.to(device) if isinstance(t, torch.Tensor) else t for t in cache.value_cache]
    elif isinstance(cache, (list, tuple)):
        for k, v in cache:
            k.data = k.data.to(device)
            v.data = v.data.to(device)


def get_vla_action(
    cfg: Any,
    vla: torch.nn.Module,
    processor: Any,
    obs: Dict[str, Any],
    task_label: str,
    action_head: Optional[torch.nn.Module] = None,
    proprio_projector: Optional[torch.nn.Module] = None,
    noisy_action_projector: Optional[torch.nn.Module] = None,
    use_film: bool = False,
    last_caches = None,
    collect_analysis: bool = False,
    analysis_frame_idx: int = 0,
    analysis_task_id: int = 0,
    analysis_episode_idx: int = 0,
) -> List[np.ndarray]:
    """
    Generate action predictions with the VLA policy.

    Args:
        cfg: Configuration object with parameters
        vla: The VLA model
        processor: Model processor for inputs
        obs: Observation dictionary
        task_label: Text description of the task
        action_head: Optional action head for continuous actions
        proprio_projector: Optional proprioception projector
        noisy_action_projector: Optional noisy action projector for diffusion
        use_film: Whether to use FiLM

    Returns:
        List[np.ndarray]: Predicted actions
    """
    # Collect all input images
    all_images = [obs["full_image"]]
    if cfg.num_images_in_input > 1:
        all_images.extend([obs[k] for k in obs.keys() if "wrist" in k])

    # Process images
    all_images = prepare_images_for_vla(all_images, cfg)
    result_image = all_images.copy()
    
    
    prev_images = obs["prev_images"]
    prev_images = prepare_images_for_vla(prev_images, cfg)
    prompt_cache = last_caches['past_key_values'] if last_caches is not None else None
    prev_attn_raw = last_caches['attentions'] if last_caches is not None else None
    # Track whether the first preprune step has already run (so A-class cache is now B-free).
    preprune_warmup_done = last_caches.get('preprune_warmup_done', False) if last_caches is not None else False
    # last_full_attn: last attention from a full-sequence (non-compact) step.
    # Used for get_layer_mask_schedule (needs full-space attn); only valid when not in cascade.
    last_full_attn = last_caches.get('last_full_attn', None) if last_caches is not None else None
    # Per-step attention scores (256-d) updated each cascade step via compact-space remapping.
    # Used as attn_scores_override in task_relevant_selection so B classification stays current.
    last_attn_scores_primary = last_caches.get('last_attn_scores_primary', None) if last_caches is not None else None
    last_attn_scores_wrist   = last_caches.get('last_attn_scores_wrist',   None) if last_caches is not None else None
    # v_global: top-K patch indices from previous step (cascade v3 temporal protection).
    # Stored as set[int] in local patch space [0..255]. None on step 0 (no history).
    v_global_primary = last_caches.get('v_global_primary', None) if last_caches is not None else None
    v_global_wrist   = last_caches.get('v_global_wrist',   None) if last_caches is not None else None
    # prev_query_images: images from the PREVIOUS model query, used as reference for V_dynamic.
    # obs["prev_images"] tracks env-step-level previous frame (not per-query), so it can be stale.
    # Storing in last_caches ensures the reference is always from the previous model call.
    prev_query_images_raw = last_caches.get('prev_query_images', None) if last_caches is not None else None
    # init_attn_scores_primary: step 0 FULL-sequence attention scores [256], frozen throughout episode.
    # Used as V_local for primary pre-prune to avoid compact-attention bias (snowball effect).
    # None until first step completes (step 0 warmup runs full-vision, sets this).
    init_attn_scores_primary = last_caches.get('init_attn_scores_primary', None) if last_caches is not None else None
    # fixed_prune_p_primary: primary prune mask frozen from step 1 (local patch indices 0..255).
    # With constant N_primary_kept, wrist compact positions are stable → wrist KV reuse works (E3).
    fixed_prune_p_primary = last_caches.get('fixed_prune_p_primary', None) if last_caches is not None else None
    # Track which prune_p was used this step (for storing as fixed mask if needed).
    _prune_p_this_step: list = []

    mask_indices = None
    vla.language_model.config.proportion_attn_var = None
    vla.language_model.config.prune_patches = None        # cleared every frame; set below if use_prune
    vla.language_model.config.prepruning_B_indices = None  # cleared every frame; set below if use_preprune
    _skip = getattr(cfg, 'skip_layers', set())  # T12: layer skip
    if isinstance(_skip, str):
        _skip = set(int(x) for x in _skip.split(',') if x.strip()) if _skip else set()
    vla.language_model.config.skip_layers = _skip
    frame_stats = None  # Will hold FrameStats if collect_analysis=True
    use_prune = getattr(cfg, 'use_prune', False)
    use_preprune = getattr(cfg, 'use_preprune', False)
    use_preprune_v3 = getattr(cfg, 'use_preprune_v3', False)  # cascade v3 token selection

    # For cascade: compact attention maps from step≥1 have fewer than 256 vision tokens,
    # which breaks token_attention_merge (expects 256-patch space).
    # Fix: use last_full_attn (full-space) for get_layer_mask_schedule; use per-step
    # last_attn_scores_* overrides (kept current via compact-space remapping) for B classification.
    prev_attn = (last_full_attn if (use_preprune and last_full_attn is not None)
                 else prev_attn_raw)
    # B positions from the CURRENT step; set inside Step 4 cascade block.
    all_prune_positions: list = []

    if cfg.use_vla_cache:
        print(">> VLA-Cache inference mode")
        # Step 1: Identify visually stable patches across frames
        if prompt_cache is not None:
            if collect_analysis:
                stable_patches_primary, sim_scores_primary = find_static_patches(
                    all_images[0], prev_images[0], top_k=150, return_similarity=True)
                stable_patches_wrist, sim_scores_wrist = find_static_patches(
                    all_images[1], prev_images[1], top_k=150, return_similarity=True)
            else:
                stable_patches_primary = find_static_patches(all_images[0], prev_images[0], top_k=150)
                stable_patches_wrist = find_static_patches(all_images[1], prev_images[1], top_k=150)

        # Step 2: Use prior attention to filter out task-relevant tokens
        if prev_attn is not None:
            prune_positions_primary, prune_positions_wrist = [], []  # populated below if use_prune
            if collect_analysis:
                vis_primary, remaining_static_tokens_primary, analysis_primary = task_relevant_selection(
                    prev_attn, result_image[0], stable_patches_primary, primary=True, return_analysis=True
                )
                vis_wrist, remaining_static_tokens_wrist, analysis_wrist = task_relevant_selection(
                    prev_attn, result_image[1], stable_patches_wrist, primary=False, return_analysis=True
                )
                proportion_var = get_layer_mask_schedule(prev_attn)
                frame_stats = FrameStats(
                    frame_idx=analysis_frame_idx,
                    task_id=analysis_task_id,
                    episode_idx=analysis_episode_idx,
                    primary_sim_scores=sim_scores_primary,
                    primary_attn_scores=analysis_primary["attn_scores"],
                    primary_class_A=analysis_primary["class_A_ids"],
                    primary_class_B=analysis_primary["class_B_ids"],
                    primary_class_C=analysis_primary["class_C_ids"],
                    primary_class_D=analysis_primary["class_D_ids"],
                    primary_stable_count=analysis_primary["stable_count"],
                    wrist_sim_scores=sim_scores_wrist,
                    wrist_attn_scores=analysis_wrist["attn_scores"],
                    wrist_class_A=analysis_wrist["class_A_ids"],
                    wrist_class_B=analysis_wrist["class_B_ids"],
                    wrist_class_C=analysis_wrist["class_C_ids"],
                    wrist_class_D=analysis_wrist["class_D_ids"],
                    wrist_stable_count=analysis_wrist["stable_count"],
                    proportion_attn_var=proportion_var.cpu().numpy(),
                )
            else:
                prune_positions_primary, prune_positions_wrist = [], []
                if use_prune or use_preprune:
                    # In cascade mode, last_attn_scores_* are updated each step from compact
                    # attention via token_attention_merge(..., b_positions=...).  Passing them
                    # as override keeps B classification current without requiring full-space attns.
                    vis_primary, remaining_static_tokens_primary, prune_positions_primary = task_relevant_selection(
                        prev_attn, result_image[0], stable_patches_primary, primary=True, return_prune=True,
                        attn_scores_override=last_attn_scores_primary if use_preprune else None
                    )
                    vis_wrist, remaining_static_tokens_wrist, prune_positions_wrist = task_relevant_selection(
                        prev_attn, result_image[1], stable_patches_wrist, primary=False, return_prune=True,
                        attn_scores_override=last_attn_scores_wrist if use_preprune else None
                    )
                else:
                    vis_primary, remaining_static_tokens_primary = task_relevant_selection(
                        prev_attn, result_image[0], stable_patches_primary, primary=True
                    )
                    vis_wrist, remaining_static_tokens_wrist = task_relevant_selection(
                        prev_attn, result_image[1], stable_patches_wrist, primary=False
                    )

            result_image = [vis_primary, vis_wrist]

            # Step 3: Merge remaining static token indices and update model config
            final_static_token_indices = remaining_static_tokens_primary + remaining_static_tokens_wrist
            mask_indices = torch.tensor(final_static_token_indices, device=DEVICE) if final_static_token_indices else None

            vla.language_model.config.reusable_patches = mask_indices
            # proportion_attn_var is computed from CPU tensors (prev_attn is now on CPU);
            # move result to GPU before storing on the model config for use in forward pass.
            sched = get_layer_mask_schedule(prev_attn)
            vla.language_model.config.proportion_attn_var = sched.to(DEVICE)

            # Step 4 (Prune B): select mode based on flags
            if use_preprune_v3:
                # ── Cascade v3 / E3: primary-only pre-prune (V_global ∪ V_dynamic ∪ V_local) ─
                _ref_images = prev_query_images_raw if prev_query_images_raw is not None else prev_images
                _k_local = getattr(cfg, 'preprune_k_local', 80)
                _v_local_primary = init_attn_scores_primary if init_attn_scores_primary is not None else last_attn_scores_primary

                # E3 (use_vla_cache=True) steady state: use frozen prune mask from step 1.
                # Fixed N_primary_kept → wrist compact positions constant → wrist KV reuse valid.
                # Step 0→1 transition: fixed_prune_p_primary not yet set → compute dynamically.
                if cfg.use_vla_cache and fixed_prune_p_primary is not None:
                    prune_p = fixed_prune_p_primary  # frozen, stable N_primary_kept
                else:
                    prune_p, _ = compute_preprune_mask(
                        all_images[0], _ref_images[0],
                        v_global=v_global_primary,
                        last_attn_scores=_v_local_primary,
                        K_local=_k_local,
                    )
                _prune_p_this_step = prune_p  # record for last_caches write-back

                # Wrist: never pre-prune (viewpoint drifts, frozen V_local unsafe)
                all_prune_positions = [p + 1 for p in prune_p]  # primary only, LLM space
                if all_prune_positions:
                    b_emb_indices = torch.tensor(
                        [p - 1 for p in all_prune_positions], dtype=torch.long
                    )
                    vla.language_model.config.prepruning_B_indices = b_emb_indices

                if cfg.use_vla_cache and fixed_prune_p_primary is not None and prompt_cache is not None:
                    # E3 steady state: wrist A-class KV reuse in compact space.
                    # Compact wrist position = orig_wrist_pos - 256 + N_primary_kept
                    # (256 wrist tokens shift down by N_primary_kept after BOS + primary block).
                    N_primary_kept = 256 - len(prune_p)
                    wrist_compact_A = [p - 256 + N_primary_kept for p in remaining_static_tokens_wrist]
                    mask_indices = torch.tensor(wrist_compact_A, device=DEVICE) if wrist_compact_A else None
                    vla.language_model.config.reusable_patches = mask_indices
                    # proportion_attn_var already set above from get_layer_mask_schedule(prev_attn)
                    # (keep the existing schedule — no override needed)
                else:
                    # E2_fixed or E3 cold start (step 0→1): no KV reuse, fresh cache.
                    vla.language_model.config.reusable_patches = None
                    vla.language_model.config.proportion_attn_var = None
                    mask_indices = None
                    prompt_cache = DynamicCache()

            elif (use_prune or use_preprune) and (prune_positions_primary or prune_positions_wrist):
                all_prune_positions = prune_positions_primary + prune_positions_wrist

                if use_preprune:
                    # ── Cascade v2: Pre-prune B at projector level (no VLA-Cache A reuse) ──
                    b_emb_indices = torch.tensor(
                        [p - 1 for p in all_prune_positions], dtype=torch.long
                    )
                    vla.language_model.config.prepruning_B_indices = b_emb_indices
                    vla.language_model.config.prune_patches = None

                    vla.language_model.config.reusable_patches = None
                    vla.language_model.config.proportion_attn_var = None
                    mask_indices = None
                    prompt_cache = DynamicCache()
                else:
                    # ── Stale KV mode (no mask) ──────────────────────────────────────────
                    vla.language_model.config.prune_patches = torch.tensor(
                        all_prune_positions, dtype=torch.long, device=DEVICE
                    )

    else:
        print(">> VLA-Cache disabled")
        prompt_cache = None
        mask_indices = None

    # E2 mode: v3 pre-prune without VLA-Cache
    # Runs after the VLA-Cache disabled branch; skipped when use_vla_cache=True (handled in Step 4).
    if use_preprune_v3 and not cfg.use_vla_cache and last_attn_scores_primary is not None:
        _ref_images = prev_query_images_raw if prev_query_images_raw is not None else prev_images
        _k_local = getattr(cfg, 'preprune_k_local', 80)
        _v_local_primary = init_attn_scores_primary if init_attn_scores_primary is not None else last_attn_scores_primary
        prune_p, _ = compute_preprune_mask(
            all_images[0], _ref_images[0],
            v_global=v_global_primary,
            last_attn_scores=_v_local_primary,
            K_local=_k_local,
        )
        prune_w = []  # wrist: skip pre-prune (viewpoint changes, frozen V_local unsafe)
        all_prune_positions = [p + 1 for p in prune_p] + [p + 257 for p in prune_w]
        if all_prune_positions:
            b_emb_indices = torch.tensor([p - 1 for p in all_prune_positions], dtype=torch.long)
            vla.language_model.config.prepruning_B_indices = b_emb_indices
        prompt_cache = DynamicCache()

    if prompt_cache is None:
        prompt_cache = DynamicCache()


    # Extract primary image and additional images
    primary_image = all_images.pop(0)

    # Build VLA prompt
    prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

    # Process primary image
    inputs = processor(prompt, primary_image).to(DEVICE, dtype=torch.bfloat16)

    # Process additional wrist images if any
    if all_images:
        all_wrist_inputs = [
            processor(prompt, image_wrist).to(DEVICE, dtype=torch.bfloat16) for image_wrist in all_images
        ]
        # Concatenate all images
        primary_pixel_values = inputs["pixel_values"]
        all_wrist_pixel_values = [wrist_inputs["pixel_values"] for wrist_inputs in all_wrist_inputs]
        inputs["pixel_values"] = torch.cat([primary_pixel_values] + all_wrist_pixel_values, dim=1)

    # Process proprioception data if used
    proprio = None
    if cfg.use_proprio:
        proprio = obs["state"]
        proprio_norm_stats = vla.norm_stats[cfg.unnorm_key]["proprio"]
        obs["state"] = normalize_proprio(proprio, proprio_norm_stats)
        proprio = obs["state"]

    # Move KV cache back to GPU right before forward pass, then free CPU copy.
    # (It was offloaded to CPU after the previous query to make room for the
    # bitsandbytes INT4 dequantisation workspace ~86 MiB per FFN layer.)
    _kvcache_to_device(prompt_cache, DEVICE)
    # Release GPU memory before forward pass: return reserved-but-unallocated
    # blocks to CUDA so the allocator has a contiguous pool to draw from.
    gc.collect()
    torch.cuda.empty_cache()

    # Start timer; reset FLOPs counter if collecting analysis
    if collect_analysis:
        vla.language_model.all_FLOPs = 0
    start_time = time.time()
    metrics = {}
    # Generate action
    if action_head is None:
        # Standard VLA output (single-image inputs, discrete actions)
        action, _ = vla.predict_action(**inputs, unnorm_key=cfg.unnorm_key, do_sample=False)
    else:
        # Custom action head for continuous actions.
        # AttentionHookCapture intercepts each LlamaDecoderLayer's attention output,
        # moves it to CPU immediately, and returns None to LlamaModel's accumulator.
        # This keeps peak GPU attention memory at ~47 MB (1 layer) instead of ~1.5 GB
        # (32 layers accumulated before the forward pass returns).
        num_layers = len(vla.language_model.model.layers)
        with AttentionHookCapture(vla.language_model, num_layers) as _cap:
            with torch.no_grad():
                action, _, last_caches = vla.predict_action(
                    **inputs,
                    unnorm_key=cfg.unnorm_key,
                    do_sample=False,
                    proprio=proprio,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    action_head=action_head,
                    use_film=use_film,
                    past_key_values=prompt_cache,
                )
        # Reconstruct attentions tuple: 32 CPU tensors + cache_position (CPU).
        # The LlamaModel filled last_caches['attentions'] with (None, ..., cache_pos)
        # because our hooks replaced each layer's output with None.  We restore the
        # real attention tensors from the hook capture and keep the cache_position,
        # which token_attention_merge needs for text-token masking.
        if last_caches is not None:
            original_attns = last_caches.get('attentions') or ()
            cache_pos = original_attns[-1] if original_attns else None
            # Cascade warmup: mark done after the first preprune step (prev_attn_raw existed).
            # Use prev_attn_raw (not the possibly-overridden prev_attn) to detect cascade steps.
            _preprune_active_this_step = (use_preprune or use_preprune_v3) and bool(all_prune_positions)
            _new_attns = _cap.make_attentions_tuple(cache_pos)
            # last_full_attn: only update on full-sequence (non-cascade) steps.
            _new_full_attn = last_full_attn if _preprune_active_this_step else _new_attns
            # Per-step attn scores for cascade B classification (updated via compact remapping).
            # On cascade steps: recompute from compact _new_attns + current B positions.
            # On full steps: recompute from full _new_attns (b_positions=None).
            if _preprune_active_this_step and all_prune_positions:
                _new_scores_primary = token_attention_merge(
                    _new_attns, primary=True, b_positions=all_prune_positions)
                _new_scores_wrist = token_attention_merge(
                    _new_attns, primary=False, b_positions=all_prune_positions)
            else:
                _new_scores_primary = token_attention_merge(_new_attns, primary=True)
                _new_scores_wrist   = token_attention_merge(_new_attns, primary=False)
            # V_global: top-K indices from current step's attention (for next step's pre-prune)
            _K_GLOBAL = 60
            _new_v_global_primary = set(_new_scores_primary.topk(_K_GLOBAL).indices.tolist())
            _new_v_global_wrist   = set(_new_scores_wrist.topk(_K_GLOBAL).indices.tolist())
            # init_attn_scores_primary: frozen at step 0 (full-sequence, unbiased).
            # Carry forward unchanged once set; _new_scores_primary at step 0 is full-sequence
            # because _preprune_active_this_step is False (no pruning yet).
            _existing_init = last_caches.get('init_attn_scores_primary', None)
            _init_scores_primary = (
                _existing_init if _existing_init is not None  # already set → persist
                else (_new_scores_primary if not _preprune_active_this_step else None)  # step 0: capture
            )
            last_caches = {
                'past_key_values': last_caches['past_key_values'],
                'attentions': _new_attns,
                'preprune_warmup_done': _preprune_active_this_step or preprune_warmup_done,
                'last_full_attn': _new_full_attn,
                'last_attn_scores_primary': _new_scores_primary,
                'last_attn_scores_wrist':   _new_scores_wrist,
                'v_global_primary': _new_v_global_primary,
                'v_global_wrist':   _new_v_global_wrist,
                'init_attn_scores_primary': _init_scores_primary,
                # Store current images so next query can use them for V_dynamic pixel-sim.
                # result_image = all_images before pop(0), i.e. [primary, wrist] PIL images.
                'prev_query_images': result_image,
            }
            # Offload KV cache to CPU so the next forward pass has room for
            # the bitsandbytes dequantisation workspace (~86 MiB per FFN layer).
            _kvcache_to_device(last_caches['past_key_values'], 'cpu')
            gc.collect()
            torch.cuda.empty_cache()
    # End timer
    end_time = time.time()
    time_elapsed = end_time - start_time
    metrics.update({"time_elapsed": time_elapsed})
    metrics.update({"num_static_tokens_primary": len(remaining_static_tokens_primary) if mask_indices is not None else 0})
    metrics.update({"num_static_tokens_wrist": len(remaining_static_tokens_wrist) if mask_indices is not None else 0})
    if collect_analysis:
        llm_flops = getattr(vla.language_model, "all_FLOPs", 0)
        if frame_stats is not None:
            frame_stats.llm_flops = float(llm_flops)
            frame_stats.time_ms = time_elapsed * 1000.0
        metrics.update({"frame_stats": frame_stats, "llm_flops": llm_flops})
    
    # Extract subset of actions for open loop steps
    action_list = [action[i] for i in range(min(len(action), cfg.num_open_loop_steps))]
    result_image = [np.array(image) for image in result_image]
    return action_list, last_caches, result_image, metrics


def get_action_from_server(
    observation: Dict[str, Any], server_endpoint: str = "http://0.0.0.0:8777/act"
) -> Dict[str, Any]:
    """
    Get VLA action from remote inference server.

    Args:
        observation: Observation data to send to server
        server_endpoint: URL of the inference server

    Returns:
        Dict[str, Any]: Action response from server
    """
    response = requests.post(
        server_endpoint,
        json=observation,
    )
    return response.json()
