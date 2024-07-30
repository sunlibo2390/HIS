import os
import re
import enum
import math
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from safetensors.torch import save_file as safe_save_file
import warnings
from enum import Enum
from datasets import Dataset
from typing import Union, List, Callable, Any, Tuple, Optional
from peft import PeftModel, LoraConfig, __version__
from peft.tuners.tuners_utils import check_target_module_exists
from peft.config import PeftConfig, PromptLearningConfig
from peft.utils import PeftType, ModulesToSaveWrapper,\
    _prepare_prompt_learning_config, transpose, _freeze_adapter, _get_submodules,\
    SAFETENSORS_WEIGHTS_NAME, WEIGHTS_NAME, TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
from peft.import_utils import is_bnb_4bit_available, is_bnb_available
from dataclasses import asdict, dataclass, field
from transformers import PreTrainedModel, LlamaForCausalLM, Trainer
from transformers.generation.streamers import BaseStreamer
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.models.llama.modeling_llama import BaseModelOutputWithPast, CausalLMOutputWithPast, \
     repeat_kv, apply_rotary_pos_emb
from transformers.pytorch_utils import Conv1D
from transformers.utils import logging
from accelerate.hooks import AlignDevicesHook, add_hook_to_module, remove_hook_from_submodules
from transformers.generation.logits_process import (
    EncoderNoRepeatNGramLogitsProcessor,
    EncoderRepetitionPenaltyLogitsProcessor,
    EpsilonLogitsWarper,
    EtaLogitsWarper,
    ExponentialDecayLengthPenalty,
    ForcedBOSTokenLogitsProcessor,
    ForcedEOSTokenLogitsProcessor,
    ForceTokensLogitsProcessor,
    HammingDiversityLogitsProcessor,
    InfNanRemoveLogitsProcessor,
    LogitNormalization,
    LogitsProcessorList,)

from transformers.generation.stopping_criteria import (
    MaxLengthCriteria,
    MaxTimeCriteria,
    StoppingCriteria,
    StoppingCriteriaList,
    validate_stopping_criteria,
)

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
        # to_return = lora_state_dict(model, bias=model.peft_config.bias)
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
        # to_return = {k: v for k, v in to_return.items() if (("lora_" in k and adapter_name in k) or ("bias" in k))}
        # if config.peft_type == PeftType.ADALORA:
        #     rank_pattern = config.rank_pattern
        #     if rank_pattern is not None:
        #         rank_pattern = {k.replace(f".{adapter_name}", ""): v for k, v in rank_pattern.items()}
        #         config.rank_pattern = rank_pattern
        #         to_return = model.resize_state_dict_by_rank_pattern(rank_pattern, to_return, adapter_name)

    # elif config.peft_type == PeftType.ADAPTION_PROMPT:
    #     to_return = {k: state_dict[k] for k in state_dict if k.split(".")[-1].startswith("adaption_")}
    # elif isinstance(config, PromptLearningConfig):
    #     to_return = {}
    #     if config.inference_mode:
    #         prompt_embeddings = model.prompt_encoder[adapter_name].embedding.weight
    #     else:
    #         prompt_embeddings = model.get_prompt_embedding_to_save(adapter_name)
    #     to_return["prompt_embeddings"] = prompt_embeddings
    # elif config.peft_type == PeftType.IA3:
    #     to_return = {k: state_dict[k] for k in state_dict if "ia3_" in k}
    else:
        raise NotImplementedError
    # if model.modules_to_save is not None:
    #     for key, value in state_dict.items():
    #         if any(f"{module_name}.modules_to_save.{adapter_name}" in key for module_name in model.modules_to_save):
    #             to_return[key.replace("modules_to_save.", "")] = value

    # to_return = {k.replace(f".{adapter_name}", ""): v for k, v in to_return.items()}
    return to_return

# class MLoraDataset:
#     def __init__(self, dataset):
#         self = dataset

#     def 


@dataclass
class MLoraConfig(PromptLearningConfig):
    """
    This is the configuration class to store the configuration of a [`LoraModel`].

    Args:
        r (`int`): Lora attention dimension.
        target_modules (`Union[List[str],str]`): The names of the modules to apply Lora
          to.
        lora_alpha (`int`): The alpha parameter for Lora scaling.
        lora_dropout (`float`): The dropout probability for Lora layers.
        fan_in_fan_out (`bool`): Set this to True if the layer to replace stores weight like (fan_in, fan_out).
        For example, gpt-2 uses `Conv1D` which stores weights like (fan_in, fan_out) and hence this should be set to `True`.:
        bias (`str`): Bias type for Lora. Can be 'none', 'all' or 'lora_only'
        modules_to_save (`List[str]`):List of modules apart from LoRA layers to be set as trainable
            and saved in the final checkpoint.
        layers_to_transform (`Union[List[int],int]`):
            The layer indexes to transform, if this argument is specified, it will apply the LoRA transformations on
            the layer indexes that are specified in this list. If a single integer is passed, it will apply the LoRA
            transformations on the layer at this index.
        layers_pattern (`str`):
            The layer pattern name, used only if `layers_to_transform` is different from `None` and if the layer
            pattern is not in the common layers pattern.
    """

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
        assert self.insert_mode in ["flat", "layered", "alternate"]

        if self.insert_mode=="flat":
            if isinstance(self.adapter_names, list) and isinstance(self.adapter_names[0], list) and isinstance(self.adapter_names[0][0], str):
                self.adapter_names = [[item for layer in self.adapter_names for item in layer ]]
            elif isinstance(self.adapter_names, list) and isinstance(self.adapter_names[0], str):
                self.adapter_names = [self.adapter_names]
        elif self.insert_mode=="layered":
            if isinstance(self.adapter_names, list) and isinstance(self.adapter_names[0], str):
                self.adapter_names = [self.adapter_names]
        elif self.insert_mode=="alternate":
            assert isinstance(self.adapter_names, list) and isinstance(self.adapter_names[0], list) and isinstance(self.adapter_names[0][0], str)
            assert DECODER_LAYER_NUM % len(self.adapter_names) == 0

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

class MLoraModel(PeftModel, nn.Module):
    def __init__(self, model: PreTrainedModel, config: MLoraConfig):
        nn.Module.__init__(self)
        self.special_peft_forward_args = {"adapter_names"}
        self.peft_type = PeftType.MLORA
        self.base_model = model
        self.base_model.active_adapter = self.active_adapter
        self.base_model._is_prompt_learning = True
        # self.forward = self.base_model.forward
        self.config = config
        self.config._is_prompt_learning = True
        self.peft_config = {}
        self.adapter_names: Union[List[List[str]], List[str]] = config.adapter_names
        self.insert_mode: str = config.insert_mode
        self.add_adapters(self.adapter_names, self.config)

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
                    "has_fp16_weights": target.state.has_fp16_weights,
                    "memory_efficient_backward": target.state.memory_efficient_backward,
                    "threshold": target.state.threshold,
                    "index": target.index,
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
        # elif isinstance(target, torch.nn.Embedding):
        #     embedding_kwargs = kwargs.copy()
        #     embedding_kwargs.pop("fan_in_fan_out", None)
        #     in_features, out_features = target.num_embeddings, target.embedding_dim
        #     new_module = Embedding(adapter_name, in_features, out_features, **embedding_kwargs)
        # elif isinstance(target, torch.nn.Conv2d):
        #     out_channels, in_channels = target.weight.size()[:2]
        #     kernel_size = target.weight.size()[2:]
        #     stride = target.stride
        #     padding = target.padding
        #     new_module = Conv2d(adapter_name, in_channels, out_channels, kernel_size, stride, padding, **kwargs)
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
        lora_config = self.config # Default every lora config is same.
        self._check_quantization_dependency()
        is_target_modules_in_base_model = False
        key_list = [key for key, _ in self.base_model.named_modules()]

        for key in key_list:
            if not check_target_module_exists(lora_config, key):
                continue

            is_target_modules_in_base_model = True
            parent, target, target_name = _get_submodules(self.base_model, key)

            # if isinstance(target, MLoraLayer) and isinstance(target, torch.nn.Conv2d):
            #     target.update_layer_conv2d(
            #         adapter_names,
            #         lora_config.r,
            #         lora_config.lora_alpha,
            #         lora_config.lora_dropout,
            #         lora_config.init_lora_weights,
            #     )
            # elif isinstance(target, MLoraLayer) and isinstance(target, torch.nn.Embedding):
            #     target.update_layer_embedding(
            #         adapter_names,
            #         lora_config.r,
            #         lora_config.lora_alpha,
            #         lora_config.lora_dropout,
            #         lora_config.init_lora_weights,
            #     )

            # if isinstance(target, MLoraLayer): # 若已MLora化，则重新
            #     in_features = target.gate[0].in_features
            #     target.gate = nn.ModuleList([])
            #     for layer_idx, layer_names in enumerate(adapter_names):
            #         names = [f"{layer_idx}_{name}" for name in ["default"] + layer_names]
            #         for adapter_name in names:
            #             target.update_layer(
            #                 adapter_name,
            #                 lora_config.r,
            #                 lora_config.lora_alpha,
            #                 lora_config.lora_dropout,
            #                 lora_config.init_lora_weights,
            #             )
            #         target.append(
            #             nn.Linear(in_features=in_features, out_features=len(layer_names), bias=True, device=target.weight.device)
            #         )
            # else:
            if insert_mode=="alternate":
                if "layers" in key:
                    decoder_layer_idx = int(key.split("layers.")[-1].split(".")[0])
                    layer_idx = decoder_layer_idx % len(adapter_names)
                    new_module = self._create_new_module(lora_config, [adapter_names[layer_idx]], target)
            elif insert_mode in ["flat", "layered"]:
                new_module = self._create_new_module(lora_config, adapter_names, target)

            # new_module = self._create_new_module(lora_config, adapter_names, target)
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
        # model.load_adapter(model_id, adapter_name, is_trainable=is_trainable, **kwargs)
        adapter_path = os.path.join(model_id, "adapter_model.bin")
        adapter_weights = torch.load(adapter_path, map_location=kwargs['device_map'])
        # for name, parameters in model.named_parameters():
        #     if name in adapter_weights.keys():
        #         parameters.weight = adapter_weights[name].to(parameters.device)
        #         print(name)
        new_model_state_dict = {key:value if key not in adapter_weights.keys() else adapter_weights[key].to(value.device) for key, value in model.named_parameters()}
        model.load_state_dict(new_model_state_dict)
        return model, config

    # def _setup_prompt_encoder(self, adapter_name: str):
    #     config = self.peft_config[adapter_name]
    #     if not hasattr(self, "prompt_encoder"):
    #         self.prompt_encoder = torch.nn.ModuleDict({})
    #         self.prompt_tokens = {}
    #     transformer_backbone = None
    #     for name, module in self.base_model.named_children():
    #         for param in module.parameters():
    #             param.requires_grad = False
    #         if isinstance(module, PreTrainedModel):
    #             # Make sure to freeze Tranformers model
    #             if transformer_backbone is None:
    #                 transformer_backbone = module
    #                 self.transformer_backbone_name = name
    #     if transformer_backbone is None:
    #         transformer_backbone = self.base_model

    #     if config.num_transformer_submodules is None:
    #         config.num_transformer_submodules = 2 if config.task_type == TaskType.SEQ_2_SEQ_LM else 1

    #     for named_param, value in list(transformer_backbone.named_parameters()):
    #         if value.shape[0] == self.base_model.config.vocab_size:
    #             self.word_embeddings = transformer_backbone.get_submodule(named_param.replace(".weight", ""))
    #             break

    #     if config.peft_type == PeftType.PROMPT_TUNING:
    #         prompt_encoder = PromptEmbedding(config, self.word_embeddings)
    #     elif config.peft_type == PeftType.P_TUNING:
    #         prompt_encoder = PromptEncoder(config)
    #     elif config.peft_type == PeftType.PREFIX_TUNING:
    #         prompt_encoder = PrefixEncoder(config)
    #     else:
    #         raise ValueError("Not supported")

    #     prompt_encoder = prompt_encoder.to(self.device)
    #     self.prompt_encoder.update(torch.nn.ModuleDict({adapter_name: prompt_encoder}))
    #     self.prompt_tokens[adapter_name] = torch.arange(
    #         config.num_virtual_tokens * config.num_transformer_submodules
    #     ).long()

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
        # self.base_model.prepare_inputs_for_generation = self.prepare_inputs_for_generation

    def beam_sample(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        active_adapters: List[str]=[],
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        **model_kwargs,
    ) -> Union[BeamSampleOutput, torch.LongTensor]: # type: ignore
        r"""
        Generates sequences of token ids for models with a language modeling head using **beam search multinomial
        sampling** and can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        <Tip warning={true}>

        In most cases, you do not need to call [`~generation.GenerationMixin.beam_sample`] directly. Use generate()
        instead. For an overview of generation strategies and code examples, check the [following
        guide](../generation_strategies).

        </Tip>

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            beam_scorer (`BeamScorer`):
                A derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
                sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            logits_warper (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step.
            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`Union[int, List[int]]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.BeamSampleDecoderOnlyOutput`], [`~generation.BeamSampleEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.BeamSampleDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.BeamSampleEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForSeq2SeqLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     TopKLogitsWarper,
        ...     TemperatureLogitsWarper,
        ...     BeamSearchScorer,
        ... )
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

        >>> encoder_input_str = "translate English to German: How old are you?"
        >>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids

        >>> # lets run beam search using 3 beams
        >>> num_beams = 3
        >>> # define decoder start token ids
        >>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
        >>> input_ids = input_ids * model.config.decoder_start_token_id

        >>> # add encoder_outputs to model keyword arguments
        >>> model_kwargs = {
        ...     "encoder_outputs": model.get_encoder()(
        ...         encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
        ...     )
        ... }

        >>> # instantiate beam scorer
        >>> beam_scorer = BeamSearchScorer(
        ...     batch_size=1,
        ...     max_length=model.config.max_length,
        ...     num_beams=num_beams,
        ...     device=model.device,
        ... )

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id)]
        ... )
        >>> # instantiate logits processors
        >>> logits_warper = LogitsProcessorList(
        ...     [
        ...         TopKLogitsWarper(50),
        ...         TemperatureLogitsWarper(0.7),
        ...     ]
        ... )

        >>> outputs = model.beam_sample(
        ...     input_ids, beam_scorer, logits_processor=logits_processor, logits_warper=logits_warper, **model_kwargs
        ... )

        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ['Wie alt bist du?']
        ```"""
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.base_model.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False  # used by synced_gpus only
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            model_inputs = self.base_model.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self(
                **model_inputs,
                active_adapters=active_adapters,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :].to(input_ids.device)

            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            next_token_scores_processed = logits_processor(input_ids, next_token_scores)
            next_token_scores_processed = logits_warper(input_ids, next_token_scores_processed)
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(
                next_token_scores_processed
            )

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores_processed,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.base_model.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.base_model.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.base_model.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            probs = nn.functional.softmax(next_token_scores, dim=-1)

            next_tokens = torch.multinomial(probs, num_samples=2 * num_beams)
            next_token_scores = torch.gather(next_token_scores, -1, next_tokens)

            next_token_scores, _indices = torch.sort(next_token_scores, descending=True, dim=1)
            next_tokens = torch.gather(next_tokens, -1, _indices)

            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
            )
            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            model_kwargs = self.base_model._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.base_model.config.is_encoder_decoder
            )
            if model_kwargs["past_key_values"] is not None:
                model_kwargs["past_key_values"] = self.base_model._reorder_cache(model_kwargs["past_key_values"], beam_idx)

            if return_dict_in_generate and output_scores:
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None

            if self.base_model.config.is_encoder_decoder:
                return BeamSampleEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=sequence_outputs["beam_indices"],
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return BeamSampleDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=sequence_outputs["beam_indices"],
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return sequence_outputs["sequences"]

    def beam_search(
        self,
        input_ids: torch.LongTensor,
        beam_scorer: BeamScorer,
        active_adapters: List[str]=[],
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        **model_kwargs,
    ) -> Union[BeamSearchOutput, torch.LongTensor]: # type: ignore
        r"""
        Generates sequences of token ids for models with a language modeling head using **beam search decoding** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        <Tip warning={true}>

        In most cases, you do not need to call [`~generation.GenerationMixin.beam_search`] directly. Use generate()
        instead. For an overview of generation strategies and code examples, check the [following
        guide](../generation_strategies).

        </Tip>

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            beam_scorer (`BeamScorer`):
                An derived instance of [`BeamScorer`] that defines how beam hypotheses are constructed, stored and
                sorted during generation. For more information, the documentation of [`BeamScorer`] should be read.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`Union[int, List[int]]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`generation.BeamSearchDecoderOnlyOutput`], [`~generation.BeamSearchEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.BeamSearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.BeamSearchEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.


        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForSeq2SeqLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     BeamSearchScorer,
        ... )
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("t5-base")
        >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")

        >>> encoder_input_str = "translate English to German: How old are you?"
        >>> encoder_input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids


        >>> # lets run beam search using 3 beams
        >>> num_beams = 3
        >>> # define decoder start token ids
        >>> input_ids = torch.ones((num_beams, 1), device=model.device, dtype=torch.long)
        >>> input_ids = input_ids * model.config.decoder_start_token_id

        >>> # add encoder_outputs to model keyword arguments
        >>> model_kwargs = {
        ...     "encoder_outputs": model.get_encoder()(
        ...         encoder_input_ids.repeat_interleave(num_beams, dim=0), return_dict=True
        ...     )
        ... }

        >>> # instantiate beam scorer
        >>> beam_scorer = BeamSearchScorer(
        ...     batch_size=1,
        ...     num_beams=num_beams,
        ...     device=model.device,
        ... )

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [
        ...         MinLengthLogitsProcessor(5, eos_token_id=model.config.eos_token_id),
        ...     ]
        ... )

        >>> outputs = model.beam_search(input_ids, beam_scorer, logits_processor=logits_processor, **model_kwargs)

        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ['Wie alt bist du?']
        ```"""
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        if len(stopping_criteria) == 0:
            warnings.warn("You don't have defined any stopping_criteria, this will likely loop forever", UserWarning)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        batch_size = len(beam_scorer._beam_hyps)
        num_beams = beam_scorer.num_beams

        batch_beam_size, cur_len = input_ids.shape

        if num_beams * batch_size != batch_beam_size:
            raise ValueError(
                f"Batch dimension of `input_ids` should be {num_beams * batch_size}, but is {batch_beam_size}."
            )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        beam_indices = (
            tuple(() for _ in range(batch_beam_size)) if (return_dict_in_generate and output_scores) else None
        )
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.base_model.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
        # of the first beam are considered to avoid sampling the exact same tokens across all beams.
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view((batch_size * num_beams,))

        this_peer_finished = False  # used by synced_gpus only
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            model_inputs = self.base_model.prepare_inputs_for_generation(input_ids, **model_kwargs)

            outputs = self(
                **model_inputs,
                active_adapters=active_adapters,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                cur_len = cur_len + 1
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :].to(input_ids.device)
            next_token_scores = nn.functional.log_softmax(
                next_token_logits, dim=-1
            )  # (batch_size * num_beams, vocab_size)

            next_token_scores_processed = logits_processor(input_ids, next_token_scores)
            next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(
                next_token_scores_processed
            )

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores_processed,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.base_model.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.base_model.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.base_model.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

            # Sample 1 + len(eos_token_id) next tokens for each beam so we have at least 1 non eos token per beam.
            n_eos_tokens = len(eos_token_id) if eos_token_id else 0
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, max(2, 1 + n_eos_tokens) * num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size

            # stateless
            beam_outputs = beam_scorer.process(
                input_ids,
                next_token_scores,
                next_tokens,
                next_indices,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                beam_indices=beam_indices,
            )

            beam_scores = beam_outputs["next_beam_scores"]
            beam_next_tokens = beam_outputs["next_beam_tokens"]
            beam_idx = beam_outputs["next_beam_indices"]

            input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)

            model_kwargs = self.base_model._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.base_model.config.is_encoder_decoder
            )
            if model_kwargs["past_key_values"] is not None:
                model_kwargs["past_key_values"] = self.base_model._reorder_cache(model_kwargs["past_key_values"], beam_idx)

            if return_dict_in_generate and output_scores:
                beam_indices = tuple((beam_indices[beam_idx[i]] + (beam_idx[i],) for i in range(len(beam_indices))))

            # increase cur_len
            cur_len = cur_len + 1

            if beam_scorer.is_done or stopping_criteria(input_ids, scores):
                if not synced_gpus:
                    break
                else:
                    this_peer_finished = True

        sequence_outputs = beam_scorer.finalize(
            input_ids,
            beam_scores,
            next_tokens,
            next_indices,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            max_length=stopping_criteria.max_length,
            beam_indices=beam_indices,
        )

        if return_dict_in_generate:
            if not output_scores:
                sequence_outputs["sequence_scores"] = None

            if self.base_model.config.is_encoder_decoder:
                return BeamSearchEncoderDecoderOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=sequence_outputs["beam_indices"],
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return BeamSearchDecoderOnlyOutput(
                    sequences=sequence_outputs["sequences"],
                    sequences_scores=sequence_outputs["sequence_scores"],
                    scores=scores,
                    beam_indices=sequence_outputs["beam_indices"],
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return sequence_outputs["sequences"]

    def sample(
        self,
        input_ids: torch.LongTensor,
        active_adapters: List[str]=[],
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[SampleOutput, torch.LongTensor]:
        r"""
        Generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        <Tip warning={true}>

        In most cases, you do not need to call [`~generation.GenerationMixin.sample`] directly. Use generate() instead.
        For an overview of generation strategies and code examples, check the [following
        guide](../generation_strategies).

        </Tip>

        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            logits_warper (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step.
            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`Union[int, List[int]]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.SampleDecoderOnlyOutput`], [`~generation.SampleEncoderDecoderOutput`] or `torch.LongTensor`:
            A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.SampleDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.SampleEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForCausalLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     TopKLogitsWarper,
        ...     TemperatureLogitsWarper,
        ...     StoppingCriteriaList,
        ...     MaxLengthCriteria,
        ... )
        >>> import torch

        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

        >>> # set pad_token_id to eos_token_id because GPT2 does not have a EOS token
        >>> model.config.pad_token_id = model.config.eos_token_id
        >>> model.generation_config.pad_token_id = model.config.eos_token_id

        >>> input_prompt = "Today is a beautiful day, and"
        >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [
        ...         MinLengthLogitsProcessor(15, eos_token_id=model.generation_config.eos_token_id),
        ...     ]
        ... )
        >>> # instantiate logits processors
        >>> logits_warper = LogitsProcessorList(
        ...     [
        ...         TopKLogitsWarper(50),
        ...         TemperatureLogitsWarper(0.7),
        ...     ]
        ... )

        >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

        >>> torch.manual_seed(0)  # doctest: +IGNORE_RESULT
        >>> outputs = model.sample(
        ...     input_ids,
        ...     logits_processor=logits_processor,
        ...     logits_warper=logits_warper,
        ...     stopping_criteria=stopping_criteria,
        ... )

        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ['Today is a beautiful day, and we must do everything possible to make it a day of celebration.']
        ```"""
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        logits_warper = logits_warper if logits_warper is not None else LogitsProcessorList()
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.base_model.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        this_peer_finished = False  # used by synced_gpus only
        # auto-regressive generation
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.base_model.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                active_adapters=active_adapters,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :].to(input_ids.device)

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.base_model.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.base_model.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.base_model.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # sample
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self.base_model._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.base_model.config.is_encoder_decoder
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

            # stop if we exceed the maximum length
            if stopping_criteria(input_ids, scores):
                this_peer_finished = True

            if this_peer_finished and not synced_gpus:
                break

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.base_model.config.is_encoder_decoder:
                return SampleEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return SampleDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return input_ids

    def greedy_search(
        self,
        input_ids: torch.LongTensor,
        active_adapters: List[str]=[],
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[GreedySearchOutput, torch.LongTensor]: # type: ignore
        r"""
        Generates sequences of token ids for models with a language modeling head using **greedy decoding** and can be
        used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        <Tip warning={true}>

        In most cases, you do not need to call [`~generation.GenerationMixin.greedy_search`] directly. Use generate()
        instead. For an overview of generation strategies and code examples, check the [following
        guide](../generation_strategies).

        </Tip>


        Parameters:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.

            max_length (`int`, *optional*, defaults to 20):
                **DEPRECATED**. Use `logits_processor` or `stopping_criteria` directly to cap the number of generated
                tokens. The maximum length of the sequence to be generated.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`Union[int, List[int]]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific keyword arguments will be forwarded to the `forward` function of the model.
                If model is an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.GreedySearchDecoderOnlyOutput`], [`~generation.GreedySearchEncoderDecoderOutput`] or
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.GreedySearchDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.GreedySearchEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        Examples:

        ```python
        >>> from transformers import (
        ...     AutoTokenizer,
        ...     AutoModelForCausalLM,
        ...     LogitsProcessorList,
        ...     MinLengthLogitsProcessor,
        ...     StoppingCriteriaList,
        ...     MaxLengthCriteria,
        ... )

        >>> tokenizer = AutoTokenizer.from_pretrained("gpt2")
        >>> model = AutoModelForCausalLM.from_pretrained("gpt2")

        >>> # set pad_token_id to eos_token_id because GPT2 does not have a PAD token
        >>> model.generation_config.pad_token_id = model.generation_config.eos_token_id

        >>> input_prompt = "It might be possible to"
        >>> input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

        >>> # instantiate logits processors
        >>> logits_processor = LogitsProcessorList(
        ...     [
        ...         MinLengthLogitsProcessor(10, eos_token_id=model.generation_config.eos_token_id),
        ...     ]
        ... )
        >>> stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=20)])

        >>> outputs = model.greedy_search(
        ...     input_ids, logits_processor=logits_processor, stopping_criteria=stopping_criteria
        ... )

        >>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
        ["It might be possible to get a better understanding of the nature of the problem, but it's not"]
        ```"""
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])` instead.",
                UserWarning,
            )
            stopping_criteria = validate_stopping_criteria(stopping_criteria, max_length)
        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # if model is an encoder-decoder, retrieve encoder attention weights and hidden states
        if return_dict_in_generate and self.base_model.config.is_encoder_decoder:
            encoder_attentions = model_kwargs["encoder_outputs"].get("attentions") if output_attentions else None
            encoder_hidden_states = (
                model_kwargs["encoder_outputs"].get("hidden_states") if output_hidden_states else None
            )

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        this_peer_finished = False  # used by synced_gpus only
        while True:
            if synced_gpus:
                # Under synced_gpus the `forward` call must continue until all gpus complete their sequence.
                # The following logic allows an early break if all peers finished generating their sequence
                this_peer_finished_flag = torch.tensor(0.0 if this_peer_finished else 1.0).to(input_ids.device)
                # send 0.0 if we finished, 1.0 otherwise
                dist.all_reduce(this_peer_finished_flag, op=dist.ReduceOp.SUM)
                # did all peers finish? the reduced sum will be 0.0 then
                if this_peer_finished_flag.item() == 0.0:
                    break

            # prepare model inputs
            model_inputs = self.base_model.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                active_adapters=active_adapters,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            if synced_gpus and this_peer_finished:
                continue  # don't waste resources running the code we don't need

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_tokens_scores = logits_processor(input_ids, next_token_logits)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    scores += (next_tokens_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.base_model.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.base_model.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.base_model.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # argmax
            next_tokens = torch.argmax(next_tokens_scores, dim=-1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self.base_model._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.base_model.config.is_encoder_decoder
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

            # stop if we exceed the maximum length
            if stopping_criteria(input_ids, scores):
                this_peer_finished = True

            if this_peer_finished and not synced_gpus:
                break

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            if self.base_model.config.is_encoder_decoder:
                return GreedySearchEncoderDecoderOutput(
                    sequences=input_ids,
                    scores=scores,
                    encoder_attentions=encoder_attentions,
                    encoder_hidden_states=encoder_hidden_states,
                    decoder_attentions=decoder_attentions,
                    cross_attentions=cross_attentions,
                    decoder_hidden_states=decoder_hidden_states,
                )
            else:
                return GreedySearchDecoderOnlyOutput(
                    sequences=input_ids,
                    scores=scores,
                    attentions=decoder_attentions,
                    hidden_states=decoder_hidden_states,
                )
        else:
            return input_ids

    # @torch.no_grad()
    # def generate(
    #     self,
    #     inputs: Optional[torch.Tensor] = None,
    #     active_adapters: List[str] = [],
    #     generation_config: Optional[GenerationConfig] = None,
    #     logits_processor: Optional[LogitsProcessorList] = None,
    #     stopping_criteria: Optional[StoppingCriteriaList] = None,
    #     prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
    #     synced_gpus: Optional[bool] = None,
    #     assistant_model: Optional["PreTrainedModel"] = None,
    #     streamer: Optional["BaseStreamer"] = None,
    #     negative_prompt_ids: Optional[torch.Tensor] = None,
    #     negative_prompt_attention_mask: Optional[torch.Tensor] = None,
    #     **kwargs,
    # ) -> Union[GenerateOutput, torch.LongTensor]: # type: ignore
    #     r"""

    #     Generates sequences of token ids for models with a language modeling head.

    #     <Tip warning={true}>

    #     Most generation-controlling parameters are set in `generation_config` which, if not passed, will be set to the
    #     model's default generation configuration. You can override any `generation_config` by passing the corresponding
    #     parameters to generate(), e.g. `.generate(inputs, num_beams=4, do_sample=True)`.

    #     For an overview of generation strategies and code examples, check out the [following
    #     guide](../generation_strategies).

    #     </Tip>

    #     Parameters:
    #         inputs (`torch.Tensor` of varying shape depending on the modality, *optional*):
    #             The sequence used as a prompt for the generation or as model inputs to the encoder. If `None` the
    #             method initializes it with `bos_token_id` and a batch size of 1. For decoder-only models `inputs`
    #             should of in the format of `input_ids`. For encoder-decoder models *inputs* can represent any of
    #             `input_ids`, `input_values`, `input_features`, or `pixel_values`.
    #         generation_config (`~generation.GenerationConfig`, *optional*):
    #             The generation configuration to be used as base parametrization for the generation call. `**kwargs`
    #             passed to generate matching the attributes of `generation_config` will override them. If
    #             `generation_config` is not provided, the default will be used, which had the following loading
    #             priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
    #             configuration. Please note that unspecified parameters will inherit [`~generation.GenerationConfig`]'s
    #             default values, whose documentation should be checked to parameterize generation.
    #         logits_processor (`LogitsProcessorList`, *optional*):
    #             Custom logits processors that complement the default logits processors built from arguments and
    #             generation config. If a logit processor is passed that is already created with the arguments or a
    #             generation config an error is thrown. This feature is intended for advanced users.
    #         stopping_criteria (`StoppingCriteriaList`, *optional*):
    #             Custom stopping criteria that complement the default stopping criteria built from arguments and a
    #             generation config. If a stopping criteria is passed that is already created with the arguments or a
    #             generation config an error is thrown. If your stopping criteria depends on the `scores` input, make
    #             sure you pass `return_dict_in_generate=True, output_scores=True` to `generate`. This feature is
    #             intended for advanced users.
    #         prefix_allowed_tokens_fn (`Callable[[int, torch.Tensor], List[int]]`, *optional*):
    #             If provided, this function constraints the beam search to allowed tokens only at each step. If not
    #             provided no constraint is applied. This function takes 2 arguments: the batch ID `batch_id` and
    #             `input_ids`. It has to return a list with the allowed tokens for the next generation step conditioned
    #             on the batch ID `batch_id` and the previously generated tokens `inputs_ids`. This argument is useful
    #             for constrained generation conditioned on the prefix, as described in [Autoregressive Entity
    #             Retrieval](https://arxiv.org/abs/2010.00904).
    #         synced_gpus (`bool`, *optional*):
    #             Whether to continue running the while loop until max_length. Unless overridden this flag will be set to
    #             `True` under DeepSpeed ZeRO Stage 3 multiple GPUs environment to avoid hanging if one GPU finished
    #             generating before other GPUs. Otherwise it'll be set to `False`.
    #         assistant_model (`PreTrainedModel`, *optional*):
    #             An assistant model that can be used to accelerate generation. The assistant model must have the exact
    #             same tokenizer. The acceleration is achieved when forecasting candidate tokens with the assistent model
    #             is much faster than running generation with the model you're calling generate from. As such, the
    #             assistant model should be much smaller.
    #         streamer (`BaseStreamer`, *optional*):
    #             Streamer object that will be used to stream the generated sequences. Generated tokens are passed
    #             through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
    #         negative_prompt_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
    #             The negative prompt needed for some processors such as CFG. The batch size must match the input batch
    #             size. This is an experimental feature, subject to breaking API changes in future versions.
    #         negative_prompt_attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
    #             Attention_mask for `negative_prompt_ids`.
    #         kwargs (`Dict[str, Any]`, *optional*):
    #             Ad hoc parametrization of `generate_config` and/or additional model-specific kwargs that will be
    #             forwarded to the `forward` function of the model. If the model is an encoder-decoder model, encoder
    #             specific kwargs should not be prefixed and decoder specific kwargs should be prefixed with *decoder_*.

    #     Return:
    #         [`~utils.ModelOutput`] or `torch.LongTensor`: A [`~utils.ModelOutput`] (if `return_dict_in_generate=True`
    #         or when `config.return_dict_in_generate=True`) or a `torch.FloatTensor`.

    #             If the model is *not* an encoder-decoder model (`model.config.is_encoder_decoder=False`), the possible
    #             [`~utils.ModelOutput`] types are:

    #                 - [`~generation.GreedySearchDecoderOnlyOutput`],
    #                 - [`~generation.SampleDecoderOnlyOutput`],
    #                 - [`~generation.BeamSearchDecoderOnlyOutput`],
    #                 - [`~generation.BeamSampleDecoderOnlyOutput`]

    #             If the model is an encoder-decoder model (`model.config.is_encoder_decoder=True`), the possible
    #             [`~utils.ModelOutput`] types are:

    #                 - [`~generation.GreedySearchEncoderDecoderOutput`],
    #                 - [`~generation.SampleEncoderDecoderOutput`],
    #                 - [`~generation.BeamSearchEncoderDecoderOutput`],
    #                 - [`~generation.BeamSampleEncoderDecoderOutput`]
    #     """

    #     if synced_gpus is None:
    #         if is_deepspeed_zero3_enabled() and dist.get_world_size() > 1:
    #             synced_gpus = True
    #         else:
    #             synced_gpus = False

    #     # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
    #     self.base_model._validate_model_class()

    #     # priority: `generation_config` argument > `model.generation_config` (the default generation config)
    #     if generation_config is None:
    #         # legacy: users may modify the model configuration to control generation. To trigger this legacy behavior,
    #         # two conditions must be met
    #         # 1) the generation config must have been created from the model config (`_from_model_config` field);
    #         # 2) the generation config must have seen no modification since its creation (the hash is the same).
    #         if self.generation_config._from_model_config and self.generation_config._original_object_hash == hash(
    #             self.generation_config
    #         ):
    #             new_generation_config = GenerationConfig.from_model_config(self.base_model.config)
    #             if new_generation_config != self.generation_config:
    #                 warnings.warn(
    #                     "You have modified the pretrained model configuration to control generation. This is a"
    #                     " deprecated strategy to control generation and will be removed soon, in a future version."
    #                     " Please use and modify the model generation configuration (see"
    #                     " https://huggingface.co/docs/transformers/generation_strategies#default-text-generation-configuration )"
    #                 )
    #                 self.generation_config = new_generation_config
    #         generation_config = self.generation_config

    #     generation_config = copy.deepcopy(generation_config)
    #     model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs
    #     generation_config.validate()
    #     self.base_model._validate_model_kwargs(model_kwargs.copy())

    #     # 2. Set generation parameters if not already defined
    #     logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
    #     stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

    #     if generation_config.pad_token_id is None and generation_config.eos_token_id is not None:
    #         if model_kwargs.get("attention_mask", None) is None:
    #             logger.warning(
    #                 "The attention mask and the pad token id were not set. As a consequence, you may observe "
    #                 "unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results."
    #             )
    #         eos_token_id = generation_config.eos_token_id
    #         if isinstance(eos_token_id, list):
    #             eos_token_id = eos_token_id[0]
    #         logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{eos_token_id} for open-end generation.")
    #         generation_config.pad_token_id = eos_token_id

    #     # 3. Define model inputs
    #     # inputs_tensor has to be defined
    #     # model_input_name is defined if model-specific keyword input is passed
    #     # otherwise model_input_name is None
    #     # all model-specific keyword inputs are removed from `model_kwargs`
    #     inputs_tensor, model_input_name, model_kwargs = self.base_model._prepare_model_inputs(
    #         inputs, generation_config.bos_token_id, model_kwargs
    #     )
    #     batch_size = inputs_tensor.shape[0]

    #     # 4. Define other model kwargs
    #     model_kwargs["output_attentions"] = generation_config.output_attentions
    #     model_kwargs["output_hidden_states"] = generation_config.output_hidden_states
    #     # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
    #     # generating the first new token or not, and we only want to use the embeddings for the first new token)
    #     if not self.base_model.config.is_encoder_decoder and model_input_name == "inputs_embeds":
    #         model_kwargs["use_cache"] = True
    #     else:
    #         model_kwargs["use_cache"] = generation_config.use_cache

    #     accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
    #     requires_attention_mask = "encoder_outputs" not in model_kwargs

    #     if model_kwargs.get("attention_mask", None) is None and requires_attention_mask and accepts_attention_mask:
    #         model_kwargs["attention_mask"] = self.base_model._prepare_attention_mask_for_generation(
    #             inputs_tensor, generation_config.pad_token_id, generation_config.eos_token_id
    #         )

    #     # decoder-only models should use left-padding for generation
    #     if not self.base_model.config.is_encoder_decoder:
    #         # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
    #         # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
    #         if (
    #             generation_config.pad_token_id is not None
    #             and len(inputs_tensor.shape) == 2
    #             and torch.sum(inputs_tensor[:, -1] == generation_config.pad_token_id) > 0
    #         ):
    #             logger.warning(
    #                 "A decoder-only architecture is being used, but right-padding was detected! For correct "
    #                 "generation results, please set `padding_side='left'` when initializing the tokenizer."
    #             )

    #     if self.base_model.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
    #         # if model is encoder decoder encoder_outputs are created
    #         # and added to `model_kwargs`
    #         model_kwargs = self.base_model._prepare_encoder_decoder_kwargs_for_generation(
    #             inputs_tensor, model_kwargs, model_input_name
    #         )

    #     # 5. Prepare `input_ids` which will be used for auto-regressive generation
    #     if self.base_model.config.is_encoder_decoder:
    #         input_ids, model_kwargs = self.base_model._prepare_decoder_input_ids_for_generation(
    #             batch_size=batch_size,
    #             model_input_name=model_input_name,
    #             model_kwargs=model_kwargs,
    #             decoder_start_token_id=generation_config.decoder_start_token_id,
    #             bos_token_id=generation_config.bos_token_id,
    #             device=inputs_tensor.device,
    #         )
    #     else:
    #         input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

    #     if streamer is not None:
    #         streamer.put(input_ids.cpu())

    #     # 6. Prepare `max_length` depending on other stopping criteria.
    #     input_ids_length = input_ids.shape[-1]
    #     has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
    #     if generation_config.max_new_tokens is not None:
    #         if not has_default_max_length and generation_config.max_length is not None:
    #             logger.warning(
    #                 f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
    #                 f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
    #                 "Please refer to the documentation for more information. "
    #                 "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
    #             )
    #         generation_config.max_length = generation_config.max_new_tokens + input_ids_length
    #     self.base_model._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

    #     # 7. determine generation mode
    #     generation_mode = self.base_model._get_generation_mode(generation_config, assistant_model)

    #     if streamer is not None and (generation_config.num_beams > 1):
    #         raise ValueError(
    #             "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
    #         )

    #     if self.base_model.device.type != input_ids.device.type:
    #         warnings.warn(
    #             "You are calling .generate() with the `input_ids` being on a device type different"
    #             f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
    #             f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
    #             " Please make sure that you have put `input_ids` to the"
    #             f" correct device by calling for example input_ids = input_ids.to('{self.base_model.device.type}') before"
    #             " running `.generate()`.",
    #             UserWarning,
    #         )

    #     # 8. prepare distribution pre_processing samplers
    #     logits_processor = self.base_model._get_logits_processor(
    #         generation_config=generation_config,
    #         input_ids_seq_length=input_ids_length,
    #         encoder_input_ids=inputs_tensor,
    #         prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
    #         logits_processor=logits_processor,
    #         model_kwargs=model_kwargs,
    #         negative_prompt_ids=negative_prompt_ids,
    #         negative_prompt_attention_mask=negative_prompt_attention_mask,
    #     )

    #     # 9. prepare stopping criteria
    #     stopping_criteria = self.base_model._get_stopping_criteria(
    #         generation_config=generation_config, stopping_criteria=stopping_criteria
    #     )
    #     # 10. go into different generation modes
    #     if generation_mode == GenerationMode.GREEDY_SEARCH:
    #         # 11. run greedy search
    #         return self.greedy_search(
    #             input_ids,
    #             active_adapters=active_adapters,
    #             logits_processor=logits_processor,
    #             stopping_criteria=stopping_criteria,
    #             pad_token_id=generation_config.pad_token_id,
    #             eos_token_id=generation_config.eos_token_id,
    #             output_scores=generation_config.output_scores,
    #             return_dict_in_generate=generation_config.return_dict_in_generate,
    #             synced_gpus=synced_gpus,
    #             streamer=streamer,
    #             **model_kwargs,
    #         )

    #     # elif generation_mode == GenerationMode.CONTRASTIVE_SEARCH:
    #     #     if not model_kwargs["use_cache"]:
    #     #         raise ValueError("Contrastive search requires `use_cache=True`")

    #     #     return self.contrastive_search(
    #     #         input_ids,
    #     #         top_k=generation_config.top_k,
    #     #         penalty_alpha=generation_config.penalty_alpha,
    #     #         logits_processor=logits_processor,
    #     #         stopping_criteria=stopping_criteria,
    #     #         pad_token_id=generation_config.pad_token_id,
    #     #         eos_token_id=generation_config.eos_token_id,
    #     #         output_scores=generation_config.output_scores,
    #     #         return_dict_in_generate=generation_config.return_dict_in_generate,
    #     #         synced_gpus=synced_gpus,
    #     #         streamer=streamer,
    #     #         sequential=generation_config.low_memory,
    #     #         **model_kwargs,
    #     #     )

    #     elif generation_mode == GenerationMode.SAMPLE:
    #         # 11. prepare logits warper
    #         logits_warper = self.base_model._get_logits_warper(generation_config)

    #         # 12. expand input_ids with `num_return_sequences` additional sequences per batch
    #         input_ids, model_kwargs = self.base_model._expand_inputs_for_generation(
    #             input_ids=input_ids,
    #             expand_size=generation_config.num_return_sequences,
    #             is_encoder_decoder=self.base_model.config.is_encoder_decoder,
    #             **model_kwargs,
    #         )

    #         # 13. run sample
    #         return self.sample(
    #             input_ids,
    #             active_adapters=active_adapters,
    #             logits_processor=logits_processor,
    #             logits_warper=logits_warper,
    #             stopping_criteria=stopping_criteria,
    #             pad_token_id=generation_config.pad_token_id,
    #             eos_token_id=generation_config.eos_token_id,
    #             output_scores=generation_config.output_scores,
    #             return_dict_in_generate=generation_config.return_dict_in_generate,
    #             synced_gpus=synced_gpus,
    #             streamer=streamer,
    #             **model_kwargs,
    #         )

    #     elif generation_mode == GenerationMode.BEAM_SEARCH:
    #         # 11. prepare beam search scorer
    #         beam_scorer = BeamSearchScorer(
    #             batch_size=batch_size,
    #             num_beams=generation_config.num_beams,
    #             device=inputs_tensor.device,
    #             length_penalty=generation_config.length_penalty,
    #             do_early_stopping=generation_config.early_stopping,
    #             num_beam_hyps_to_keep=generation_config.num_return_sequences,
    #             max_length=generation_config.max_length,
    #         )
    #         # 12. interleave input_ids with `num_beams` additional sequences per batch
    #         input_ids, model_kwargs = self.base_model._expand_inputs_for_generation(
    #             input_ids=input_ids,
    #             expand_size=generation_config.num_beams,
    #             is_encoder_decoder=self.base_model.config.is_encoder_decoder,
    #             **model_kwargs,
    #         )
    #         active_adapters = active_adapters * generation_config.num_beams
    #         # 13. run beam search
    #         return self.beam_search(
    #             input_ids,
    #             beam_scorer,
    #             active_adapters=active_adapters,
    #             logits_processor=logits_processor,
    #             stopping_criteria=stopping_criteria,
    #             pad_token_id=generation_config.pad_token_id,
    #             eos_token_id=generation_config.eos_token_id,
    #             output_scores=generation_config.output_scores,
    #             return_dict_in_generate=generation_config.return_dict_in_generate,
    #             synced_gpus=synced_gpus,
    #             **model_kwargs,
    #         )

    #     elif generation_mode == GenerationMode.BEAM_SAMPLE:
    #         # 11. prepare logits warper
    #         logits_warper = self.base_model._get_logits_warper(generation_config)

    #         # 12. prepare beam search scorer
    #         beam_scorer = BeamSearchScorer(
    #             batch_size=batch_size,
    #             num_beams=generation_config.num_beams,
    #             device=inputs_tensor.device,
    #             length_penalty=generation_config.length_penalty,
    #             do_early_stopping=generation_config.early_stopping,
    #             num_beam_hyps_to_keep=generation_config.num_return_sequences,
    #             max_length=generation_config.max_length,
    #         )

    #         # 13. interleave input_ids with `num_beams` additional sequences per batch
    #         input_ids, model_kwargs = self.base_model._expand_inputs_for_generation(
    #             input_ids=input_ids,
    #             expand_size=generation_config.num_beams,
    #             is_encoder_decoder=self.base_model.config.is_encoder_decoder,
    #             **model_kwargs,
    #         )
    #         active_adapters = torch.tensor(active_adapters).repeat_interleave(generation_config.num_beams, dim=0)
    #         # 14. run beam sample
    #         return self.beam_sample(
    #             input_ids,
    #             beam_scorer,
    #             active_adapters=active_adapters,
    #             logits_processor=logits_processor,
    #             logits_warper=logits_warper,
    #             stopping_criteria=stopping_criteria,
    #             pad_token_id=generation_config.pad_token_id,
    #             eos_token_id=generation_config.eos_token_id,
    #             output_scores=generation_config.output_scores,
    #             return_dict_in_generate=generation_config.return_dict_in_generate,
    #             synced_gpus=synced_gpus,
    #             **model_kwargs,
    #         )

    #     # elif generation_mode == GenerationMode.GROUP_BEAM_SEARCH:
    #     #     # 11. prepare beam search scorer
    #     #     beam_scorer = BeamSearchScorer(
    #     #         batch_size=batch_size,
    #     #         num_beams=generation_config.num_beams,
    #     #         device=inputs_tensor.device,
    #     #         length_penalty=generation_config.length_penalty,
    #     #         do_early_stopping=generation_config.early_stopping,
    #     #         num_beam_hyps_to_keep=generation_config.num_return_sequences,
    #     #         num_beam_groups=generation_config.num_beam_groups,
    #     #         max_length=generation_config.max_length,
    #     #     )
    #     #     # 12. interleave input_ids with `num_beams` additional sequences per batch
    #     #     input_ids, model_kwargs = self._expand_inputs_for_generation(
    #     #         input_ids=input_ids,
    #     #         expand_size=generation_config.num_beams,
    #     #         is_encoder_decoder=self.config.is_encoder_decoder,
    #     #         **model_kwargs,
    #     #     )
    #     #     # 13. run beam search
    #     #     return self.group_beam_search(
    #     #         input_ids,
    #     #         beam_scorer,
    #     #         logits_processor=logits_processor,
    #     #         stopping_criteria=stopping_criteria,
    #     #         pad_token_id=generation_config.pad_token_id,
    #     #         eos_token_id=generation_config.eos_token_id,
    #     #         output_scores=generation_config.output_scores,
    #     #         return_dict_in_generate=generation_config.return_dict_in_generate,
    #     #         synced_gpus=synced_gpus,
    #     #         **model_kwargs,
    #     #     )

    #     # elif generation_mode == GenerationMode.CONSTRAINED_BEAM_SEARCH:
    #     #     final_constraints = []
    #     #     if generation_config.constraints is not None:
    #     #         final_constraints = generation_config.constraints

    #     #     if generation_config.force_words_ids is not None:

    #     #         def typeerror():
    #     #             raise ValueError(
    #     #                 "`force_words_ids` has to either be a `List[List[List[int]]]` or `List[List[int]]` "
    #     #                 f"of positive integers, but is {generation_config.force_words_ids}."
    #     #             )

    #     #         if (
    #     #             not isinstance(generation_config.force_words_ids, list)
    #     #             or len(generation_config.force_words_ids) == 0
    #     #         ):
    #     #             typeerror()

    #     #         for word_ids in generation_config.force_words_ids:
    #     #             if isinstance(word_ids[0], list):
    #     #                 if not isinstance(word_ids, list) or len(word_ids) == 0:
    #     #                     typeerror()
    #     #                 if any(not isinstance(token_ids, list) for token_ids in word_ids):
    #     #                     typeerror()
    #     #                 if any(
    #     #                     any((not isinstance(token_id, int) or token_id < 0) for token_id in token_ids)
    #     #                     for token_ids in word_ids
    #     #                 ):
    #     #                     typeerror()

    #     #                 constraint = DisjunctiveConstraint(word_ids)
    #     #             else:
    #     #                 if not isinstance(word_ids, list) or len(word_ids) == 0:
    #     #                     typeerror()
    #     #                 if any((not isinstance(token_id, int) or token_id < 0) for token_id in word_ids):
    #     #                     typeerror()

    #     #                 constraint = PhrasalConstraint(word_ids)
    #     #             final_constraints.append(constraint)

    #     #     # 11. prepare beam search scorer
    #     #     constrained_beam_scorer = ConstrainedBeamSearchScorer(
    #     #         constraints=final_constraints,
    #     #         batch_size=batch_size,
    #     #         num_beams=generation_config.num_beams,
    #     #         device=inputs_tensor.device,
    #     #         length_penalty=generation_config.length_penalty,
    #     #         do_early_stopping=generation_config.early_stopping,
    #     #         num_beam_hyps_to_keep=generation_config.num_return_sequences,
    #     #         max_length=generation_config.max_length,
    #     #     )
    #     #     # 12. interleave input_ids with `num_beams` additional sequences per batch
    #     #     input_ids, model_kwargs = self._expand_inputs_for_generation(
    #     #         input_ids=input_ids,
    #     #         expand_size=generation_config.num_beams,
    #     #         is_encoder_decoder=self.config.is_encoder_decoder,
    #     #         **model_kwargs,
    #     #     )
    #     #     # 13. run beam search
    #     #     return self.constrained_beam_search(
    #     #         input_ids,
    #     #         constrained_beam_scorer=constrained_beam_scorer,
    #     #         logits_processor=logits_processor,
    #     #         stopping_criteria=stopping_criteria,
    #     #         pad_token_id=generation_config.pad_token_id,
    #     #         eos_token_id=generation_config.eos_token_id,
    #     #         output_scores=generation_config.output_scores,
    #     #         return_dict_in_generate=generation_config.return_dict_in_generate,
    #     #         synced_gpus=synced_gpus,
    #     #         **model_kwargs,
    #     #     )

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
        active_adapters = []
        for i in range(len(active_adapters_idx)):
            active_adapters.append(
                [
                    self.config.adapter_names[active_adapter_idx[0]][active_adapter_idx[1]] 
                    for active_adapter_idx in active_adapters_idx[i]
                ]
            )
        batch_size, token_size, hidden_size = x.shape
        previous_dtype = x.dtype
        for layer_idx, layer_adapters in enumerate(self.adapter_names):
            layer_result = F.linear(x, transpose(self.weight, self.fan_in_fan_out), bias=self.bias)
            gate = self.gate[layer_idx]
            layer_lora_output = []
            layer_active_idx_list = []
            # find all active adapters in this layer adapters
            for active_adapter in active_adapters:
                active_adapter_dict = {adapter:i for i, adapter in enumerate(layer_adapters) if active_adapter in adapter}
                if len(active_adapter_dict)>0:
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
            bnb.nn.Linear8bitLt.__init__(
                self,
                in_features,
                out_features,
                bias=kwargs.get("bias", True),
                has_fp16_weights=kwargs.get("has_fp16_weights", True),
                memory_efficient_backward=kwargs.get("memory_efficient_backward", False),
                threshold=kwargs.get("threshold", 0.0),
                index=kwargs.get("index", None),
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
                    # result = (
                    #     self.lora_B[active_adapter](
                    #         self.lora_A[active_adapter](self.lora_dropout[active_adapter](x))
                    #     )
                    #     * self.scaling[active_adapter]
                    # )

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



            # list[list[str]] [batch_size, active_experts_num]
            previous_dtype = x.dtype
            batch_size, token_size, hidden_size = x.shape

            for layer_idx, layer_adapters in enumerate(self.adapter_names):
                layer_result = super().forward(x)
                gate = self.gate[layer_idx]
                for b in range(batch_size):
                    layer_lora_output = []
                    layer_active_idx_list = []
                    # find all active adapters in this layer adapters
                    for active_adapter in active_adapters[b]:
                        active_adapter_dict = {adapter:i for i, adapter in enumerate(layer_adapters) if active_adapter in adapter}
                        if len(active_adapter_dict)>0:
                            active_adapter_name, active_idx = next(iter(active_adapter_dict.items()))
                            
                            lora_output = self.single_expert_forward(x[b].unsqueeze(0), active_adapter_name) # [1, token_size, hidden_size]
                            layer_active_idx_list.append(active_idx)
                            layer_lora_output.append(lora_output.unsqueeze(2))
                        else:
                            continue
                    if len(layer_lora_output) > 0:
                        layer_lora_output = torch.cat(
                            layer_lora_output, dim=2
                        ).view(token_size, -1, hidden_size) # [token_size, layer_active_expert_num, hidden_size]
                        x = x.to(gate.weight.dtype)
                        layer_expert_weights = torch.softmax(
                            gate(x[b].unsqueeze(0)), dim=-1
                        ) # [1, token_size, layer_expert_num]
                        layer_expert_weights = layer_expert_weights.to(previous_dtype)
                        layer_expert_weights = torch.softmax(
                            torch.cat(
                                [
                                    layer_expert_weights[:,:,idx].unsqueeze(-1)
                                    for idx in layer_active_idx_list
                                ], dim=-1
                            ).view(token_size, 1, -1), dim=-1
                        ) # [token_size, 1, layer_active_expert_num]
                        
                        layer_result[b] += torch.bmm(
                            layer_expert_weights, layer_lora_output
                        ).squeeze(1).view(token_size, hidden_size)
                x = layer_result

            result = x.to(previous_dtype)
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
                active_adapters = []
                for i in range(len(active_adapters_idx)):
                    active_adapters.append(
                        [
                            self.config.adapter_names[active_adapter_idx[0]][active_adapter_idx[1]] 
                            for active_adapter_idx in active_adapters_idx[i]
                        ]
                    )
                previous_dtype = x.dtype
                batch_size, token_size, hidden_size = x.shape
                for layer_idx, layer_adapters in enumerate(self.adapter_names):
                    layer_result = super().forward(x)
                    gate = self.gate[layer_idx]
                    layer_lora_output = []
                    layer_active_idx_list = []
                    # find all active adapters in this layer adapters
                    for active_adapter in active_adapters:
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