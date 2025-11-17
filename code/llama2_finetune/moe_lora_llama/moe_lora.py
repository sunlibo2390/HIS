import os
import enum
import json
import math
import torch
from torch import nn
import torch.nn.functional as F
from safetensors.torch import save_file as safe_save_file
import warnings
from enum import Enum
from typing import Union, List, Any, Optional
import inspect
from peft import PeftModel, LoraConfig, __version__
from peft.tuners.tuners_utils import check_target_module_exists
from peft.config import PeftConfig, PromptLearningConfig
from peft.utils import PeftType, transpose, _freeze_adapter, _get_submodules,\
    SAFETENSORS_WEIGHTS_NAME, WEIGHTS_NAME, TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from dataclasses import asdict, dataclass, field
from transformers import PreTrainedModel, LlamaForCausalLM
from transformers.cache_utils import Cache
from transformers.pytorch_utils import Conv1D
from transformers.utils import logging

from transformers.generation.utils import *
GreedySearchOutput = Union[GreedySearchEncoderDecoderOutput, GreedySearchDecoderOnlyOutput]
SampleOutput = Union[SampleEncoderDecoderOutput, SampleDecoderOnlyOutput]
BeamSearchOutput = Union[BeamSearchEncoderDecoderOutput, BeamSearchDecoderOnlyOutput]
BeamSampleOutput = Union[BeamSampleEncoderDecoderOutput, BeamSampleDecoderOnlyOutput]
ContrastiveSearchOutput = Union[ContrastiveSearchEncoderDecoderOutput, ContrastiveSearchDecoderOnlyOutput]
GenerateOutput = Union[GreedySearchOutput, SampleOutput, BeamSearchOutput, BeamSampleOutput, ContrastiveSearchOutput]

logger = logging.get_logger(__name__)

if is_bnb_available():
    import bitsandbytes as bnb

class PeftType(str, enum.Enum):
    PROMPT_TUNING = "PROMPT_TUNING"
    P_TUNING = "P_TUNING"
    PREFIX_TUNING = "PREFIX_TUNING"
    LORA = "LORA"
    ADALORA = "ADALORA"
    ADAPTION_PROMPT = "ADAPTION_PROMPT"
    IA3 = "IA3"
    MLORA = "MLORA"

DECODER_LAYER_NUM = 32

def get_peft_model_state_dict(model, state_dict=None):
    """
    Get the state dict of the Peft model.

    Args:
        model ([`PeftModel`]): The Peft model. When using torch.nn.DistributedDataParallel, DeepSpeed or FSDP,
        the model should be the underlying model/unwrapped model (i.e. model.module).
        state_dict (`dict`, *optional*, defaults to `None`):
            The state dict of the model. If not provided, the state dict of the model
        will be used.
    """
    config = model.config
    if state_dict is None:
        state_dict = model.state_dict()
    if config.peft_type in (PeftType.LORA, PeftType.ADALORA, PeftType.MLORA):
        # adapted from `https://github.com/microsoft/LoRA/blob/main/loralib/utils.py`
        # to be used directly with the state dict which is necessary when using DeepSpeed or FSDP
        bias = config.bias
        if bias == "none":
            to_return = {k: state_dict[k] for k in state_dict if "lora_" in k or ("self_attn" in k and "gate" in k)}
        elif bias == "all":
            to_return = {k: state_dict[k] for k in state_dict if "lora_" in k or "bias" in k or ("self_attn" in k and "gate" in k)}
        elif bias == "lora_only":
            to_return = {}
            for k in state_dict:
                if "lora_" in k or ("self_attn" in k and "gate" in k):
                    to_return[k] = state_dict[k]
                    bias_name = k.split("lora_")[0] + "bias"
                    if bias_name in state_dict:
                        to_return[bias_name] = state_dict[bias_name]
        else:
            raise NotImplementedError

    else:
        raise NotImplementedError
    return to_return


@dataclass
class MLoraConfig(PromptLearningConfig):
    """Extends PromptLearningConfig with extra metadata for multi-adapter routing."""

    r: int = field(default=8, metadata={"help": "Lora attention dimension"})
    target_modules: Optional[Union[List[str], str]] = field(
        default=None,
        metadata={
            "help": "List of module names or regex expression of the module names to replace with Lora."
            "For example, ['q', 'v'] or '.*decoder.*(SelfAttention|EncDecAttention).*(q|v)$' "
        },
    )
    lora_alpha: int = field(default=8, metadata={"help": "Lora alpha"})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout"})
    fan_in_fan_out: bool = field(
        default=False,
        metadata={"help": "Set this to True if the layer to replace stores weight like (fan_in, fan_out)"},
    )
    bias: str = field(default="none", metadata={"help": "Bias type for Lora. Can be 'none', 'all' or 'lora_only'"})
    modules_to_save: Optional[List[str]] = field(
        default=None,
        metadata={
            "help": "List of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint. "
            "For example, in Sequence Classification or Token Classification tasks, "
            "the final layer `classifier/score` are randomly initialized and as such need to be trainable and saved."
        },
    )
    init_lora_weights: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to initialize the weights of the Lora layers with their default initialization. Don't change "
                "this setting, except if you know exactly what you're doing."
            ),
        },
    )
    layers_to_transform: Optional[Union[List, int]] = field(
        default=None,
        metadata={
            "help": "The layer indexes to transform, is this argument is specified, PEFT will transform only the layers indexes that are specified inside this list. If a single integer is passed, PEFT will transform only the layer at this index."
        },
    )
    layers_pattern: Optional[str] = field(
        default=None,
        metadata={
            "help": "The layer pattern name, used only if `layers_to_transform` is different to None and if the layer pattern is not in the common layers pattern."
        },
    )

    loraconfig = LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    adapter_names: Union[List[List[str]], List[str]] = field(default=None, metadata={"help": "The names of the experts to create."})
    insert_mode: str = field(default="flat", metadata={"help": "The insert mode of expert layers."})
    sparse_adapter: bool = field(default=True, metadata={"help": "Whether to use sparse adapters"})
    
    def __post_init__(self):
        self.peft_type = PeftType.MLORA
        if self.insert_mode not in {"flat", "layered", "alternate"}:
            raise ValueError(f"Unsupported insert_mode: {self.insert_mode}")

        if self.insert_mode == "flat":
            if isinstance(self.adapter_names, list) and self.adapter_names and isinstance(self.adapter_names[0], list):
                flattened = [name for group in self.adapter_names for name in group]
                self.adapter_names = [flattened]
            elif isinstance(self.adapter_names, list) and self.adapter_names and isinstance(self.adapter_names[0], str):
                self.adapter_names = [self.adapter_names]
        elif self.insert_mode == "layered":
            if isinstance(self.adapter_names, list) and self.adapter_names and isinstance(self.adapter_names[0], str):
                self.adapter_names = [self.adapter_names]
        elif self.insert_mode == "alternate":
            if not (
                isinstance(self.adapter_names, list)
                and self.adapter_names
                and isinstance(self.adapter_names[0], list)
                and isinstance(self.adapter_names[0][0], str)
            ):
                raise ValueError("alternate mode requires a list of adapter groups.")

    @property
    def is_prompt_learning(self) -> bool:
        return False

    @property
    def is_prompt_learning(self) -> bool:
        return False

    @classmethod
    def from_pretrained(self, pretrained_model_name_or_path: str, subfolder: Optional[str] = None, **kwargs):
        CONFIG_NAME = "adapter_config.json"
        from huggingface_hub import hf_hub_download
        r"""
        This method loads the configuration of your adapter model from a directory.

        Args:
            pretrained_model_name_or_path (`str`):
                The directory or the Hub repository id where the configuration is saved.
            kwargs (additional keyword arguments, *optional*):
                Additional keyword arguments passed along to the child class initialization.
        """
        path = (
            os.path.join(pretrained_model_name_or_path, subfolder)
            if subfolder is not None
            else pretrained_model_name_or_path
        )

        hf_hub_download_kwargs, class_kwargs, _ = self._split_kwargs(kwargs)

        if os.path.isfile(os.path.join(path, CONFIG_NAME)):
            config_file = os.path.join(path, CONFIG_NAME)
        else:
            try:
                config_file = hf_hub_download(
                    pretrained_model_name_or_path, CONFIG_NAME, subfolder=subfolder, **hf_hub_download_kwargs
                )
            except Exception as exc:
                raise ValueError(f"Can't find '{CONFIG_NAME}' at '{pretrained_model_name_or_path}'") from exc

        loaded_attributes = self.from_json_file(config_file)
        kwargs = {**class_kwargs, **loaded_attributes}
        return self(**kwargs)

    def to_json_string(self):
        """将配置转换为 JSON 字符串"""
        config_dict = {k: v for k, v in self.__dict__.items()}
        return json.dumps(config_dict, sort_keys=True)

class MLoraModel(PeftModel, nn.Module):
    def __init__(
        self,
        model: PreTrainedModel,
        config: MLoraConfig,
        adapter_name: str = "default",
        autocast_adapter_dtype: bool = True,
        low_cpu_mem_usage: bool = False,
        **kwargs,
    ):
        nn.Module.__init__(self)
        self.special_peft_forward_args = {"adapter_names"}
        self.peft_type = PeftType.MLORA
        self.base_model = model
        self.base_model.active_adapter = config.adapter_names
        self.base_model._is_prompt_learning = True
        self.config = config
        self.config._is_prompt_learning = True
        self.peft_config = {}
        self.active_adapter = adapter_name
        self.adapter_names: Union[List[List[str]], List[str]] = config.adapter_names
        self.insert_mode: str = config.insert_mode
        self.add_adapters(self.adapter_names, self.config)

    def _setup_prompt_encoder(self, adapter_name: str):
        return

    def print_trainable_parameters(self):
        """
        Prints the number of trainable parameters in the model.
        """
        trainable_params = 0
        all_param = 0
        for name, param in self.named_parameters():
            num_params = param.numel()
            # if using DS Zero 3 and the weights are initialized empty
            if num_params == 0 and hasattr(param, "ds_numel"):
                num_params = param.ds_numel

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
                # print(name, param.shape)
        print(
            f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
        )

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        if peft_config.target_modules is None:
            if model_config["model_type"] not in TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING:
                raise ValueError("Please specify `target_modules` in `peft_config`")
            peft_config.target_modules = set(
                TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING[model_config["model_type"]]
            )
        return peft_config

    def add_adapters(self, adapter_names, config=None):
        if config is not None:
            model_config = self.base_model.config.to_dict() if hasattr(self.base_model.config, "to_dict") else self.base_model.config
            config = self._prepare_adapter_config(config, model_config)
            for layer in adapter_names:
                for name in layer:
                    self.peft_config[name] = config
                    self.active_adapter = name
        self._find_and_replace(adapter_names, self.insert_mode) # insert adapters and gates accroding to insert_mode and adapter_names
        if len(self.peft_config) > 1 and self.config.bias != "none":
            raise ValueError(
                "LoraModel supports only 1 adapter with bias. When using multiple adapters, set bias to 'none' for all adapters."
            )
        mark_only_lora_as_trainable(self.base_model, self.config.bias)
        # mark_gate_as_trainable(self.base_model)
        for layer_names in adapter_names:
            for adapter_name in layer_names:
                if self.config.inference_mode:
                    _freeze_adapter(self.base_model, adapter_name)

    def _check_quantization_dependency(self):
        loaded_in_4bit = getattr(self.base_model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.base_model, "is_loaded_in_8bit", False)
        if (loaded_in_4bit or loaded_in_8bit) and not is_bnb_available():
            raise ImportError(
                "To use Lora with 8-bit or 4-bit quantization, please install the `bitsandbytes` package. "
                "You can install it with `pip install bitsandbytes`."
            )

    def _create_new_module(self, lora_config, adapter_names: List[List[str]], target):
        bias = hasattr(target, "bias") and target.bias is not None
        kwargs = {
            "r": lora_config.r,
            "lora_alpha": lora_config.lora_alpha,
            "lora_dropout": lora_config.lora_dropout,
            "fan_in_fan_out": lora_config.fan_in_fan_out,
            "init_lora_weights": lora_config.init_lora_weights,
        }
        loaded_in_4bit = getattr(self.base_model, "is_loaded_in_4bit", False)
        loaded_in_8bit = getattr(self.base_model, "is_loaded_in_8bit", False)

        if loaded_in_8bit and isinstance(target, bnb.nn.Linear8bitLt):
            eightbit_kwargs = kwargs.copy()
            eightbit_kwargs.update(
                {
                    "has_fp16_weights": getattr(target.state, "has_fp16_weights", True),
                    "memory_efficient_backward": getattr(target.state, "memory_efficient_backward", False),
                    "threshold": getattr(target.state, "threshold", 0.0),
                    "index": getattr(target, "index", None),
                }
            )
            new_module = MLinear8bitLt(
                adapter_names, target.in_features, target.out_features, self.config, bias=bias, **eightbit_kwargs
            )
        elif loaded_in_4bit and is_bnb_4bit_available() and isinstance(target, bnb.nn.Linear4bit):
            fourbit_kwargs = kwargs.copy()
            fourbit_kwargs.update(
                {
                    "compute_dtype": target.compute_dtype,
                    "compress_statistics": target.weight.compress_statistics,
                    "quant_type": target.weight.quant_type,
                }
            )
            new_module = MLinear4bit(adapter_names, target.in_features, target.out_features, self.config, bias=bias, **fourbit_kwargs)
        else:
            if isinstance(target, torch.nn.Linear):
                in_features, out_features = target.in_features, target.out_features
                if kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to True but the target module is `torch.nn.Linear`. "
                        "Setting fan_in_fan_out to False."
                    )
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = False
            elif isinstance(target, Conv1D):
                in_features, out_features = (
                    target.weight.ds_shape if hasattr(target.weight, "ds_shape") else target.weight.shape
                )
                if not kwargs["fan_in_fan_out"]:
                    warnings.warn(
                        "fan_in_fan_out is set to False but the target module is `Conv1D`. "
                        "Setting fan_in_fan_out to True."
                    )
                    kwargs["fan_in_fan_out"] = lora_config.fan_in_fan_out = True
            else:
                raise ValueError(
                    f"Target module {target} is not supported. "
                    f"Currently, only `torch.nn.Linear` and `Conv1D` are supported."
                )
            new_module = MLinear(adapter_names, in_features, out_features, self.config, bias=bias, **kwargs)

        return new_module

    def _find_and_replace(self, adapter_names, insert_mode):
        """Insert MOE adapters according to the chosen layout strategy."""
        lora_config = self.config
        self._check_quantization_dependency()
        is_target_modules_in_base_model = False
        key_list = [key for key, _ in self.base_model.named_modules()]

        for key in key_list:
            if not check_target_module_exists(lora_config, key):
                continue

            is_target_modules_in_base_model = True
            parent, target, target_name = _get_submodules(self.base_model, key)
            if insert_mode=="alternate":
                if "layers" in key:
                    decoder_layer_idx = int(key.split("layers.")[-1].split(".")[0])
                    layer_idx = decoder_layer_idx % len(adapter_names)
                    new_module = self._create_new_module(lora_config, [adapter_names[layer_idx]], target)
            elif insert_mode in ["flat", "layered"]:
                new_module = self._create_new_module(lora_config, adapter_names, target)

            self._replace_module(parent, target_name, new_module, target)

        if not is_target_modules_in_base_model:
            raise ValueError(
                f"Target modules {lora_config.target_modules} not found in the base model. "
                f"Please check the target modules and try again."
            )

    def _replace_module(self, parent_module, child_name, new_module, old_module):
        setattr(parent_module, child_name, new_module)
        new_module.weight = old_module.weight
        if hasattr(old_module, "bias"):
            if old_module.bias is not None:
                new_module.bias = old_module.bias

        if getattr(old_module, "state", None) is not None:
            new_module.state = old_module.state
            new_module.to(old_module.weight.device)

        # dispatch to correct device
        for name, module in new_module.named_modules():
            if "lora_" in name:
                module.to(old_module.weight.device)
            if "ranknum" in name:
                module.to(old_module.weight.device)
            if "gate" in name:
                module.to(old_module.weight.device)

    def get_peft_config_as_dict(self, inference: bool = False):
        config_dict = {}
        for key, value in self.peft_config.items():
            config = {k: v.value if isinstance(v, Enum) else v for k, v in asdict(value).items()}
            if inference:
                config["inference_mode"] = True
        config_dict[key] = config
        return config

    def save_pretrained(
        self,
        save_directory: str,
        safe_serialization: bool = False,
        selected_adapters: Optional[List[str]] = None,
        **kwargs: Any,
    ):
        r"""
        This function saves the adapter model and the adapter configuration files to a directory, so that it can be
        reloaded using the [`LoraModel.from_pretrained`] class method, and also used by the [`LoraModel.push_to_hub`]
        method.

        Args:
            save_directory (`str`):
                Directory where the adapter model and configuration files will be saved (will be created if it does not
                exist).
            kwargs (additional keyword arguments, *optional*):
                Additional keyword arguments passed along to the `push_to_hub` method.
        """
        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")

        if selected_adapters is None:
            selected_adapters = list(self.peft_config.keys())
        else:
            if any(
                selected_adapter_name not in list(self.peft_config.keys())
                for selected_adapter_name in selected_adapters
            ):
                raise ValueError(
                    f"You passed an invalid `selected_adapters` arguments, current supported adapter names are"
                    f" {list(self.peft_config.keys())} - got {selected_adapters}."
                )

        os.makedirs(save_directory, exist_ok=True)
        self.create_or_update_model_card(save_directory)

        peft_config = self.config
        # save only the trainable weights
        output_state_dict = get_peft_model_state_dict(
            self, state_dict=kwargs.get("state_dict", None)
        )
        output_dir = os.path.join(save_directory)
        os.makedirs(output_dir, exist_ok=True)

        if safe_serialization:
            safe_save_file(
                output_state_dict,
                os.path.join(output_dir, SAFETENSORS_WEIGHTS_NAME),
                metadata={"format": "pt"},
            )
        else:
            torch.save(output_state_dict, os.path.join(output_dir, WEIGHTS_NAME))

        # save the config and change the inference mode to `True`
        if peft_config.base_model_name_or_path is None:
            peft_config.base_model_name_or_path = (
                self.base_model.__dict__.get("name_or_path", None)
                if isinstance(peft_config, PromptLearningConfig)
                else self.base_model.model.__dict__.get("name_or_path", None)
            )
        inference_mode = peft_config.inference_mode
        peft_config.inference_mode = True

        if peft_config.task_type is None:
            # deal with auto mapping
            base_model_class = self._get_base_model_class(
                is_prompt_tuning=isinstance(peft_config, PromptLearningConfig)
            )
            parent_library = base_model_class.__module__

            auto_mapping_dict = {
                "base_model_class": base_model_class.__name__,
                "parent_library": parent_library,
            }
        else:
            auto_mapping_dict = None

        peft_config.save_pretrained(output_dir, auto_mapping_dict=auto_mapping_dict)
        peft_config.inference_mode = inference_mode

    @classmethod
    def from_pretrained(
        cls,
        model: PreTrainedModel,
        model_id: Union[str, os.PathLike],
        is_trainable: bool = False,
        config: Optional[PeftConfig] = None,
        **kwargs: Any,
    ):
        r"""
        Instantiate a [`LoraModel`] from a pretrained Lora configuration and weights.

        Args:
            model ([`~transformers.PreTrainedModel`]):
                The model to be adapted. The model should be initialized with the
                [`~transformers.PreTrainedModel.from_pretrained`] method from the 🤗 Transformers library.
            model_id (`str` or `os.PathLike`):
                The name of the Lora configuration to use. Can be either:
                    - A string, the `model id` of a Lora configuration hosted inside a model repo on the Hugging Face
                      Hub.
                    - A path to a directory containing a Lora configuration file saved using the `save_pretrained`
                      method (`./my_lora_config_directory/`).
            adapter_name (`str`, *optional*, defaults to `"default"`):
                The name of the adapter to be loaded. This is useful for loading multiple adapters.
            is_trainable (`bool`, *optional*, defaults to `False`):
                Whether the adapter should be trainable or not. If `False`, the adapter will be frozen and use for
                inference
            config ([`~peft.PeftConfig`], *optional*):
                The configuration object to use instead of an automatically loaded configuation. This configuration
                object is mutually exclusive with `model_id` and `kwargs`. This is useful when configuration is already
                loaded before calling `from_pretrained`.
            kwargs: (`optional`):
                Additional keyword arguments passed along to the specific Lora configuration class.
        """
        # load the config
        if config is None:
            config = MLoraConfig.from_pretrained(model_id, subfolder=kwargs.get("subfolder", None), **kwargs)
        elif isinstance(config, PeftConfig):
            config.inference_mode = not is_trainable
        else:
            raise ValueError(f"The input config must be a PeftConfig, got {config.__class__}")

        if (getattr(model, "hf_device_map", None) is not None) and len(
            set(model.hf_device_map.values()).intersection({"cpu", "disk"})
        ) > 0:
            remove_hook_from_submodules(model)

        if isinstance(config, PromptLearningConfig) and is_trainable:
            raise ValueError("Cannot set a prompt learning adapter to trainable when loading pretrained adapter.")
        else:
            config.inference_mode = not is_trainable

        model = MLoraModelForCausalLM(model, config)
        adapter_path = os.path.join(model_id, "adapter_model.bin")
        adapter_weights = torch.load(adapter_path)
        new_model_state_dict = {key:value if key not in adapter_weights.keys() else adapter_weights[key].to(value.device) for key, value in model.named_parameters()}
        model.load_state_dict(new_model_state_dict)
        return model, config

    def get_prompt(self, batch_size: int):
        """
        Returns the virtual prompts to use for Peft. Only applicable when `peft_config.peft_type != PeftType.LORA`.
        """
        peft_config = self.config
        prompt_encoder = self.prompt_encoder[self.active_adapter]
        prompt_tokens = (
            self.prompt_tokens[self.active_adapter]
            .unsqueeze(0)
            .expand(batch_size, -1)
            .to(prompt_encoder.embedding.weight.device)
        )
        if peft_config.peft_type == PeftType.PREFIX_TUNING:
            prompt_tokens = prompt_tokens[:, : peft_config.num_virtual_tokens]
            if peft_config.inference_mode:
                past_key_values = prompt_encoder.embedding.weight.repeat(batch_size, 1, 1)
            else:
                past_key_values = prompt_encoder(prompt_tokens)
            if self.base_model_torch_dtype is not None:
                past_key_values = past_key_values.to(self.base_model_torch_dtype)
            past_key_values = past_key_values.view(
                batch_size,
                peft_config.num_virtual_tokens,
                peft_config.num_layers * 2,
                peft_config.num_attention_heads,
                peft_config.token_dim // peft_config.num_attention_heads,
            )
            if peft_config.num_transformer_submodules == 2:
                past_key_values = torch.cat([past_key_values, past_key_values], dim=2)
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(
                peft_config.num_transformer_submodules * 2
            )
            if TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING.get(self.config.model_type, None) is not None:
                post_process_fn = TRANSFORMERS_MODELS_TO_PREFIX_TUNING_POSTPROCESS_MAPPING[self.config.model_type]
                past_key_values = post_process_fn(past_key_values)
            return past_key_values
        else:
            if peft_config.inference_mode:
                prompts = prompt_encoder.embedding.weight.repeat(batch_size, 1, 1)
            else:
                prompts = prompt_encoder(prompt_tokens)
            return prompts


class MLoraModelForCausalLM(MLoraModel):
    def __init__(self, model: LlamaForCausalLM, peft_config: MLoraConfig):
        super().__init__(model, peft_config)

    def prepare_inputs_for_generation(
        self, input_ids, active_adapters, past_key_values=None, attention_mask=None, inputs_embeds=None, cache_position=None, **kwargs
    ):
        # With static cache, the `past_key_values` is None
        # TODO joao: standardize interface for the different Cache classes and remove of this if
        has_static_cache = False
        if past_key_values is None:
            past_key_values = getattr(getattr(self.model.layers[0], "self_attn", {}), "past_key_value", None)
            has_static_cache = past_key_values is not None

        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                past_length = cache_position[0] if cache_position is not None else past_key_values.get_seq_length()
                max_cache_length = (
                    torch.tensor(past_key_values.get_max_length(), device=input_ids.device)
                    if past_key_values.get_max_length() is not None
                    else None
                )
                cache_length = past_length if max_cache_length is None else torch.min(max_cache_length, past_length)
            # TODO joao: remove this `else` after `generate` prioritizes `Cache` objects
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # The `contiguous()` here is necessary to have a static stride during decoding. torchdynamo otherwise
            # recompiles graphs as the stride of the inputs is a guard. Ref: https://github.com/huggingface/transformers/pull/29114
            # TODO: use `next_tokens` directly instead.
            model_inputs = {"input_ids": input_ids.contiguous()}

        input_length = position_ids.shape[-1] if position_ids is not None else input_ids.shape[-1]
        if cache_position is None:
            cache_position = torch.arange(past_length, past_length + input_length, device=input_ids.device)
        else:
            cache_position = cache_position[-input_length:]

        if has_static_cache:
            past_key_values = None

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "active_adapters": active_adapters,
            }
        )
        return model_inputs
        
    def generate(self, **kwargs):
        self.config.active_adapters = copy.copy(kwargs['active_adapters'])
        kwargs.pop('active_adapters')
        return self.base_model.generate(**kwargs)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        active_adapters: List[str]=[],
        return_dict=None,
        **kwargs,
    ):
        # print(f"forward xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx {active_adapters}")
        peft_config = self.config
        self.config.active_adapters = active_adapters
        
        return self.base_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )


class MLoraLayer:
    def __init__(self, in_features: int, out_features: int, **kwargs):
        self.r = {}
        self.lora_alpha = {}
        self.scaling = {}
        self.lora_dropout = nn.ModuleDict({})
        self.lora_A = nn.ModuleDict({})
        self.lora_B = nn.ModuleDict({})
        self.gate = nn.ModuleList({})
        # Mark the weight as unmerged
        self.merged = False
        self.disable_adapters = False
        self.in_features = in_features
        self.out_features = out_features
        self.kwargs = kwargs

    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights):
        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout.update(nn.ModuleDict({adapter_name: lora_dropout_layer}))
        # Actual trainable parameters
        if r > 0:
            self.lora_A.update(nn.ModuleDict({adapter_name: nn.Linear(self.in_features, r, bias=False)}))
            self.lora_B.update(nn.ModuleDict({adapter_name: nn.Linear(r, self.out_features, bias=False)}))
            self.scaling[adapter_name] = lora_alpha / r
        if init_lora_weights:
            self.reset_lora_parameters(adapter_name)

        self.to(self.weight.device)

    def reset_lora_parameters(self, adapter_name):
        if adapter_name in self.lora_A.keys():
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[adapter_name].weight)


class MLinear(nn.Linear, MLoraLayer):
    # Lora implemented in a dense layer
    def __init__(
        self,
        adapter_names: List[List[str]],
        in_features: int,
        out_features: int,
        config, 
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        **kwargs,
    ):
        init_lora_weights = kwargs.pop("init_lora_weights", True)

        nn.Linear.__init__(self, in_features, out_features, **kwargs)
        MLoraLayer.__init__(self, in_features=in_features, out_features=out_features)
        self.config = config
        # Freezing the pre-trained weight matrix
        self.weight.requires_grad = False

        self.fan_in_fan_out = fan_in_fan_out
        if fan_in_fan_out:
            self.weight.data = self.weight.data.T

        nn.Linear.reset_parameters(self)
        self.adapter_names = []
        for layer_idx, layer_names in enumerate(adapter_names):
            names = [f"{layer_idx}_{name}" for name in layer_names]
            self.adapter_names.append(names)
            for adapter_name in names:
                self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
            self.gate.append(
                nn.Linear(in_features=in_features, out_features=len(layer_names), bias=True, device=self.weight.device)
            )

    def single_lora_forward(self, x: torch.Tensor, active_adapter: str):
        # print(active_adapter)
        previous_dtype = x.dtype
        if self.r[active_adapter] > 0:
            # result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)

            x = x.to(self.lora_A[active_adapter].weight.dtype)

            result = (
                self.lora_B[active_adapter](
                    self.lora_A[active_adapter](self.lora_dropout[active_adapter](x))
                )
                * self.scaling[active_adapter]
            )
        else:
            AssertionError("The expert to be activated does not exist.")
        result = result.to(previous_dtype)
        return result

    def forward(self, x: torch.Tensor, active_adapters_idx: List[List[int]]):
        # print(active_adapters_idx)
        active_adapters: List[List[str]] = []
        for adapter_group in active_adapters_idx:
            adapter_names = []
            for active_adapter_idx in adapter_group:
                layer_idx, expert_idx = active_adapter_idx
                if layer_idx > -1 and expert_idx > -1:
                    adapter_names.append(self.config.adapter_names[layer_idx][expert_idx])
            if adapter_names:
                active_adapters.append(adapter_names)
        batch_size, token_size, hidden_size = x.shape
        previous_dtype = x.dtype
        for layer_idx, layer_adapters in enumerate(self.adapter_names):
            layer_result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            gate = self.gate[layer_idx]
            layer_lora_output = []
            layer_active_idx_list = []
            # find all active adapters in this layer adapters
            for adapter_group in active_adapters:
                for active_adapter in adapter_group:
                    active_adapter_dict = {adapter: i for i, adapter in enumerate(layer_adapters) if active_adapter in adapter}
                    if len(active_adapter_dict) > 0:
                        active_adapter_name, active_idx = next(iter(active_adapter_dict.items()))

                        lora_output = self.single_lora_forward(x, active_adapter_name) # [batch_size, token_size, hidden_size]
                        layer_active_idx_list.append(active_idx)
                        layer_lora_output.append(lora_output.unsqueeze(2))
                    else:
                        continue
            if len(layer_lora_output) > 0:
                layer_lora_output = torch.cat(
                    layer_lora_output, dim=2
                ).view(batch_size*token_size, -1, hidden_size) # [batch_size*token_size, layer_active_expert_num, hidden_size]
                x = x.to(gate.weight.dtype)
                layer_expert_weights = torch.softmax(
                    gate(x), dim=-1
                ) # [batch_size, token_size, layer_expert_num]
                layer_expert_weights = layer_expert_weights.to(previous_dtype)
                layer_expert_weights = torch.softmax(
                    torch.cat(
                        [
                            layer_expert_weights[:,:,idx].unsqueeze(-1)
                            for idx in layer_active_idx_list
                        ], dim=-1
                    ).view(batch_size*token_size, 1, -1), dim=-1
                ) # [batch_size*token_size, 1, layer_active_expert_num]
                
                layer_result += torch.bmm(
                    layer_expert_weights, layer_lora_output
                ).squeeze(1).view(batch_size, token_size, hidden_size)
            x = layer_result

        return x

if is_bnb_available():

    class MLinear8bitLt(bnb.nn.Linear8bitLt, MLoraLayer):
        # Lora implemented in a dense layer
        def __init__(
            self,
            adapter_names,
            in_features,
            out_features,
            config,
            r: int = 0,
            lora_alpha: int = 1,
            lora_dropout: float = 0.0,
            **kwargs,
        ):
            init_kwargs = {
                "bias": kwargs.get("bias", True),
                "has_fp16_weights": kwargs.get("has_fp16_weights", True),
                "threshold": kwargs.get("threshold", 0.0),
                "index": kwargs.get("index", None),
            }
            signature = inspect.signature(bnb.nn.Linear8bitLt.__init__)
            if "memory_efficient_backward" in signature.parameters:
                init_kwargs["memory_efficient_backward"] = kwargs.get("memory_efficient_backward", False)
            if "device" in signature.parameters and kwargs.get("device", None) is not None:
                init_kwargs["device"] = kwargs.get("device")
            bnb.nn.Linear8bitLt.__init__(self, in_features, out_features, **init_kwargs)
            MLoraLayer.__init__(self, in_features=in_features, out_features=out_features)
            self.config = config
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
            init_lora_weights = kwargs.pop("init_lora_weights", True)

            self.adapter_names = []
            for layer_idx, layer_names in enumerate(adapter_names):
                names = [f"{layer_idx}_{name}" for name in layer_names]
                self.adapter_names.append(names)
                for adapter_name in names:
                    self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
                self.gate.append(
                    nn.Linear(in_features=in_features, out_features=len(layer_names), bias=True, device=self.weight.device)
                )

        def single_expert_forward(self, x: torch.Tensor, active_adapter: str):
            previous_dtype = x.dtype
            active_lora_B = self.lora_B[active_adapter]
            active_lora_A = self.lora_A[active_adapter]
            if self.r[active_adapter] > 0:
                if not torch.is_autocast_enabled():
                    if x.dtype != torch.float32:
                        x = x.float()

                    # 首先应用 dropout 操作
                    dropout_output = self.lora_dropout[active_adapter](x)

                    # 将 dropout 的结果传递给 lora_A
                    lora_A_output = self.lora_A[active_adapter](dropout_output)

                    # 将上一步的输出传递给 lora_B
                    lora_B_output = self.lora_B[active_adapter](lora_A_output)

                    # 最后，将 lora_B 的输出乘以相应的 scaling 因子
                    result = lora_B_output * self.scaling[active_adapter]

                else:
                    x = x.to(self.lora_A[active_adapter].weight.dtype)
                    result = (
                        self.lora_B[active_adapter](
                            self.lora_A[active_adapter](self.lora_dropout[active_adapter](x))
                        )
                        * self.scaling[active_adapter]
                    )
            result = result.to(previous_dtype)
            return result
        
        def forward(self, x: torch.Tensor):
            active_adapters_idx = self.config.active_adapters
            previous_dtype = x.dtype
            batch_size, token_size, hidden_size = x.shape
            result = []
            # print(batch_size, x.shape, active_adapters_idx)
            if batch_size > 1 and active_adapters_idx.shape[0]!=batch_size and self.training==False:
                active_adapters_idx = active_adapters_idx.repeat_interleave(int(batch_size/active_adapters_idx.shape[0]), dim=0)
            for b in range(batch_size):
                _x = x[b] # [token_size, hidden_size]
                _active_adapters_idx = active_adapters_idx[b] # [6, 2]
                active_adapters = []

                for active_adapter_idx in _active_adapters_idx:
                    if active_adapter_idx[0]>-1 and active_adapter_idx[1]>-1:
                        active_adapters.append(
                            self.config.adapter_names[active_adapter_idx[0]][active_adapter_idx[1]] 
                        )
                for layer_idx, layer_adapters in enumerate(self.adapter_names):
                    layer_result = super().forward(_x)
                    gate = self.gate[layer_idx]
                    layer_lora_output = []
                    layer_active_idx_list = []
                    layer_adapter_mask = []
                    
                    for layer_adapter in layer_adapters:
                        exist = False
                        lora_output = self.single_expert_forward(_x.unsqueeze(0), layer_adapter) # [1, token_size, hidden_size]
                        layer_lora_output.append(lora_output.unsqueeze(2))
                        for active_adapter in active_adapters:
                            if active_adapter in layer_adapter:
                                layer_adapter_mask.append(1)
                                exist = True
                                break
                        if exist==False:
                            layer_adapter_mask.append(-5e4)

                    layer_lora_output = torch.cat(
                        layer_lora_output, dim=2
                    ).view(token_size, -1, hidden_size) # [token_size, layer_expert_num, hidden_size]

                    _x = _x.to(gate.weight.dtype)
                    layer_expert_weights = torch.sigmoid(
                        gate(_x.unsqueeze(0))
                    ) # [1, token_size, layer_expert_num]

                    layer_expert_weights = layer_expert_weights.to(previous_dtype)

                    if self.config.sparse_adapter==True:
                        layer_adapter_mask = torch.tensor(layer_adapter_mask).to(previous_dtype).to(layer_expert_weights.device) # [layer_expert_num]
                    else:
                        layer_adapter_mask = torch.ones_like(
                            torch.tensor(layer_adapter_mask)
                        ).to(previous_dtype).to(layer_expert_weights.device) # [layer_expert_num]

                    if sum(layer_adapter_mask>0)>0:
                        # print("layer_expert_weights",layer_expert_weights)
                        layer_expert_weights = torch.softmax(
                            torch.matmul(
                                layer_expert_weights, 
                                torch.diag(layer_adapter_mask)
                            ).view(token_size, 1, -1), dim=-1
                        ) # [token_size, 1, layer_active_expert_num]
                        # print("layer_adapter_mask",layer_adapter_mask)
                        # print("layer_expert_weights",layer_expert_weights)
                        layer_result = layer_result + torch.bmm(
                            layer_expert_weights, layer_lora_output
                        ).view(token_size, hidden_size)
                    else:
                        layer_expert_weights = torch.matmul(
                            layer_expert_weights, 
                            torch.diag(torch.zeros_like(layer_adapter_mask))
                        ).view(token_size, 1, -1)
                        # [token_size, 1, layer_active_expert_num]
                        # print("layer_adapter_mask",layer_adapter_mask)
                        # print("layer_expert_weights",layer_expert_weights)
                        layer_result = layer_result + torch.bmm(
                            layer_expert_weights, layer_lora_output
                        ).view(token_size, hidden_size)

                    _x = layer_result

                result.append(_x.unsqueeze(0))
            result = torch.cat(result, dim=0).to(previous_dtype) # 
            return result

    if is_bnb_4bit_available():

        class MLinear4bit(bnb.nn.Linear4bit, MLoraLayer):
            # Lora implemented in a dense layer
            def __init__(
                self,
                adapter_names,
                in_features,
                out_features,
                config,
                r: int = 0,
                lora_alpha: int = 1,
                lora_dropout: float = 0.0,
                **kwargs,
            ):
                bnb.nn.Linear4bit.__init__(
                    self,
                    in_features,
                    out_features,
                    bias=kwargs.get("bias", True),
                    compute_dtype=kwargs.get("compute_dtype", torch.float32),
                    compress_statistics=kwargs.get("compress_statistics", True),
                    quant_type=kwargs.get("quant_type", "nf4"),
                )
                MLoraLayer.__init__(self, in_features=in_features, out_features=out_features)
                self.config = config
                # Freezing the pre-trained weight matrix
                self.weight.requires_grad = False

                init_lora_weights = kwargs.pop("init_lora_weights", True)
                self.adapter_names = []
                for layer_idx, layer_names in enumerate(adapter_names):
                    names = [f"{layer_idx}_{name}" for name in layer_names]
                    self.adapter_names.append(names)
                    for adapter_name in names:
                        self.update_layer(adapter_name, r, lora_alpha, lora_dropout, init_lora_weights)
                    self.gate.append(
                        nn.Linear(in_features=in_features, out_features=len(layer_names), bias=False, device=self.weight.device)
                    )

            def single_expert_forward(self, x: torch.Tensor, active_adapter: str):
                previous_dtype = x.dtype
                if self.r[active_adapter] > 0:
                    if not torch.is_autocast_enabled():
                        x = x.to(self.lora_A[active_adapter].weight.dtype)
                        result = (
                            self.lora_B[active_adapter](
                                self.lora_A[active_adapter](self.lora_dropout[active_adapter](x))
                            ).to(previous_dtype)
                            * self.scaling[active_adapter]
                        )
                    else:
                        x = x.to(self.lora_A[active_adapter].weight.dtype)
                        result = (
                            self.lora_B[active_adapter](
                                self.lora_A[active_adapter](self.lora_dropout[active_adapter](x))
                            )
                            * self.scaling[active_adapter]
                        )
                result = result.to(previous_dtype)
                return result
            
            def forward(self, x: torch.Tensor, active_adapters_idx: List[List[int]]):
                # print(active_adapters_idx)
                active_adapters: List[List[str]] = []
                for adapter_group in active_adapters_idx:
                    adapter_names = []
                    for active_adapter_idx in adapter_group:
                        layer_idx, expert_idx = active_adapter_idx
                        if layer_idx > -1 and expert_idx > -1:
                            adapter_names.append(self.config.adapter_names[layer_idx][expert_idx])
                    if adapter_names:
                        active_adapters.append(adapter_names)
                previous_dtype = x.dtype
                batch_size, token_size, hidden_size = x.shape
                for layer_idx, layer_adapters in enumerate(self.adapter_names):
                    layer_result = super().forward(x)
                    gate = self.gate[layer_idx]
                    layer_lora_output = []
                    layer_active_idx_list = []
                    # find all active adapters in this layer adapters
                    for adapter_group in active_adapters:
                        for active_adapter in adapter_group:
                            active_adapter_dict = {adapter:i for i, adapter in enumerate(layer_adapters) if active_adapter in adapter}
                            if len(active_adapter_dict)>0:
                                active_adapter_name, active_idx = next(iter(active_adapter_dict.items()))
                                
                                lora_output = self.single_expert_forward(x, active_adapter_name) # [batch_size, token_size, hidden_size]
                                layer_active_idx_list.append(active_idx)
                                layer_lora_output.append(lora_output.unsqueeze(2))
                            else:
                                continue
                    if len(layer_lora_output) > 0:
                        layer_lora_output = torch.cat(
                            layer_lora_output, dim=2
                        ).view(batch_size*token_size, -1, hidden_size) # [batch_size*token_size, layer_active_expert_num, hidden_size]
                        x = x.to(gate.weight.dtype)
                        layer_expert_weights = torch.softmax(
                            gate(x), dim=-1
                        ) # [batch_size, token_size, layer_expert_num]
                        layer_expert_weights = layer_expert_weights.to(previous_dtype)
                        layer_expert_weights = torch.softmax(
                            torch.cat(
                                [
                                    layer_expert_weights[:,:,idx].unsqueeze(-1)
                                    for idx in layer_active_idx_list
                                ], dim=-1
                            ).view(batch_size*token_size, 1, -1), dim=-1
                        ) # [batch_size*token_size, 1, layer_active_expert_num]
                        
                        layer_result += torch.bmm(
                            layer_expert_weights, layer_lora_output
                        ).squeeze(1).view(batch_size, token_size, hidden_size)
                    x = layer_result

                result = x.to(previous_dtype)
                return result


PEFT_TYPE_TO_MODEL_MAPPING = {
    PeftType.MLORA: MLoraModel,
}

PEFT_TYPE_TO_CONFIG_MAPPING = {
    "MLORA": MLoraConfig,
}

# had to adapt it for `lora_only` to work
def mark_only_lora_as_trainable(model: nn.Module, bias: str = "none") -> None:
    for n, p in model.named_parameters():
        if "lora_" not in n:
            p.requires_grad = False
    if bias == "none":
        return
    elif bias == "all":
        for n, p in model.named_parameters():
            if "bias" in n:
                p.requires_grad = True
    elif bias == "lora_only":
        for m in model.modules():
            if isinstance(m, MLoraLayer) and hasattr(m, "bias") and m.bias is not None:
                m.bias.requires_grad = True
    else:
        raise NotImplementedError
    
def mark_gate_as_trainable(model: nn.Module):
    for n, p in model.named_parameters():
        if "gate" in n and "self_attn" in n:
            p.requires_grad = True
