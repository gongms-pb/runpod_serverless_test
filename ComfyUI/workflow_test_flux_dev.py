import os
import random
import sys
from typing import Sequence, Mapping, Any, Union
import torch


def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    """Returns the value at the given index of a sequence or mapping.

    If the object is a sequence (like list or string), returns the value at the given index.
    If the object is a mapping (like a dictionary), returns the value at the index-th key.

    Some return a dictionary, in these cases, we look for the "results" key

    Args:
        obj (Union[Sequence, Mapping]): The object to retrieve the value from.
        index (int): The index of the value to retrieve.

    Returns:
        Any: The value at the given index.

    Raises:
        IndexError: If the index is out of bounds for the object and the object is not a mapping.
    """
    try:
        return obj[index]
    except KeyError:
        return obj["result"][index]


def find_path(name: str, path: str = None) -> str:
    """
    Recursively looks at parent folders starting from the given path until it finds the given name.
    Returns the path as a Path object if found, or None otherwise.
    """
    # If no path is given, use the current working directory
    if path is None:
        path = os.getcwd()

    # Check if the current directory contains the name
    if name in os.listdir(path):
        path_name = os.path.join(path, name)
        print(f"{name} found: {path_name}")
        return path_name

    # Get the parent directory
    parent_directory = os.path.dirname(path)

    # If the parent directory is the same as the current directory, we've reached the root and stop the search
    if parent_directory == path:
        return None

    # Recursively call the function with the parent directory
    return find_path(name, parent_directory)


def add_comfyui_directory_to_sys_path() -> None:
    """
    Add 'ComfyUI' to the sys.path
    """
    comfyui_path = find_path("ComfyUI")
    if comfyui_path is not None and os.path.isdir(comfyui_path):
        sys.path.append(comfyui_path)
        print(f"'{comfyui_path}' added to sys.path")


def add_extra_model_paths() -> None:
    """
    Parse the optional extra_model_paths.yaml file and add the parsed paths to the sys.path.
    """
    try:
        from main import load_extra_path_config
    except ImportError:
        print(
            "Could not import load_extra_path_config from main.py. Looking in utils.extra_config instead."
        )
        from utils.extra_config import load_extra_path_config

    extra_model_paths = find_path("extra_model_paths.yaml")

    if extra_model_paths is not None:
        load_extra_path_config(extra_model_paths)
    else:
        print("Could not find the extra_model_paths config file.")

# add_comfyui_directory_to_sys_path()
# add_extra_model_paths()

def import_custom_nodes() -> None:
    """Find all custom nodes in the custom_nodes folder and add those node objects to NODE_CLASS_MAPPINGS

    This function sets up a new asyncio event loop, initializes the PromptServer,
    creates a PromptQueue, and initializes the custom nodes.
    """
    import asyncio
    import execution
    from nodes import init_extra_nodes
    import server

    # Creating a new event loop and setting it as the default loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Creating an instance of PromptServer with the loop
    server_instance = server.PromptServer(loop)
    execution.PromptQueue(server_instance)

    # Initializing custom nodes
    init_extra_nodes()


class NodeManager:
    def __init__(self):
        from nodes import NODE_CLASS_MAPPINGS
        self.checkpoint_loader_simple = NODE_CLASS_MAPPINGS["CheckpointLoaderSimple"]()
        self.clip_text_encode = NODE_CLASS_MAPPINGS["CLIPTextEncode"]()
        self.empty_sd3_latent_image = NODE_CLASS_MAPPINGS["EmptySD3LatentImage"]()
        self.flux_guidance = NODE_CLASS_MAPPINGS["FluxGuidance"]()
        self.k_sampler = NODE_CLASS_MAPPINGS["KSampler"]()
        self.vae_decode = NODE_CLASS_MAPPINGS["VAEDecode"]()
        self.save_image = NODE_CLASS_MAPPINGS["SaveImage"]()
        
        # Load the checkpoint during initialization
        self.checkpoint_data = self.checkpoint_loader_simple.load_checkpoint(
            ckpt_name="flux1-dev-fp8.safetensors"
        )

def init():
    add_comfyui_directory_to_sys_path()
    add_extra_model_paths()

    import_custom_nodes()
    
    nodes = NodeManager()
    return nodes

def main(nodes: NodeManager, input_prompt: str = None):
    if input_prompt is None or input_prompt == "":
        prompt = "cute anime girl with massive fluffy fennec ears and a big fluffy tail blonde messy long hair blue eyes wearing a maid outfit with a long black gold leaf pattern dress and a white apron mouth open placing a fancy black forest cake with candles on top of a dinner table of an old dark Victorian mansion lit by candlelight with a bright window to the foggy forest and very expensive stuff everywhere there are paintings on the walls"
    else:
        prompt = input_prompt
    with torch.inference_mode():
        checkpoint_data = nodes.checkpoint_data

        cliptextencode_6 = nodes.clip_text_encode.encode(
            text=prompt,
            clip=get_value_at_index(checkpoint_data, 1),
        )

        emptysd3latentimage_27 = nodes.empty_sd3_latent_image.generate(
            width=1024, height=1024, batch_size=1
        )

        cliptextencode_33 = nodes.clip_text_encode.encode(
            text="", clip=get_value_at_index(checkpoint_data, 1)
        )

        for q in range(1):
            fluxguidance_35 = nodes.flux_guidance.append(
                guidance=3.5, conditioning=get_value_at_index(cliptextencode_6, 0)
            )

            ksampler_31 = nodes.k_sampler.sample(
                seed=random.randint(1, 2**64),
                steps=20,
                cfg=1,
                sampler_name="euler",
                scheduler="simple",
                denoise=1,
                model=get_value_at_index(checkpoint_data, 0),
                positive=get_value_at_index(fluxguidance_35, 0),
                negative=get_value_at_index(cliptextencode_33, 0),
                latent_image=get_value_at_index(emptysd3latentimage_27, 0),
            )

            vaedecode_8 = nodes.vae_decode.decode(
                samples=get_value_at_index(ksampler_31, 0),
                vae=get_value_at_index(checkpoint_data, 2),
            )

            saveimage_9 = nodes.save_image.save_images(
                filename_prefix="ComfyUI", images=get_value_at_index(vaedecode_8, 0)
            )
    
    return saveimage_9

if __name__ == "__main__":
    nodes = init()
    output = main(nodes)
    print(output)
