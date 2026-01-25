"""
Training Configuration for Local LLM Fine-Tuning.

Provides configuration classes for LoRA/QLoRA fine-tuning,
model quantization, and training hyperparameters.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class QuantizationType(str, Enum):
    """Model quantization options."""
    NONE = "none"
    INT8 = "int8"
    INT4 = "int4"
    NF4 = "nf4"  # Normal Float 4-bit (QLoRA)


class TrainingMode(str, Enum):
    """Training mode options."""
    FULL = "full"  # Full fine-tuning
    LORA = "lora"  # LoRA adapters
    QLORA = "qlora"  # Quantized LoRA


@dataclass
class LoRAConfig:
    """Configuration for LoRA/QLoRA fine-tuning.
    
    Attributes:
        r: LoRA rank (typically 8-64).
        lora_alpha: LoRA alpha parameter (usually 2*r).
        lora_dropout: Dropout probability for LoRA layers.
        target_modules: Model modules to apply LoRA to.
        bias: Bias training mode ('none', 'all', 'lora_only').
        task_type: Task type for PEFT ('CAUSAL_LM', 'SEQ_CLS', etc.).
    """
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ])
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for PEFT."""
        return {
            "r": self.r,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": self.target_modules,
            "bias": self.bias,
            "task_type": self.task_type,
        }


@dataclass
class TrainingConfig:
    """Configuration for model training.
    
    Attributes:
        model_name: Base model name or path.
        output_dir: Directory for saving checkpoints and models.
        training_mode: Full, LoRA, or QLoRA training.
        quantization: Quantization type for model weights.
        lora_config: LoRA configuration (if using LoRA/QLoRA).
        
        # Training hyperparameters
        num_epochs: Number of training epochs.
        batch_size: Training batch size.
        gradient_accumulation_steps: Steps for gradient accumulation.
        learning_rate: Initial learning rate.
        weight_decay: Weight decay for regularization.
        warmup_ratio: Warmup ratio for learning rate scheduler.
        max_seq_length: Maximum sequence length.
        
        # Experiment tracking
        experiment_name: Name for experiment tracking.
        run_name: Name for this specific run.
        use_mlflow: Enable MLflow tracking.
        mlflow_tracking_uri: MLflow server URI.
    """
    # Model settings
    model_name: str = "meta-llama/Llama-3.1-8B"
    output_dir: str = "./training_output"
    training_mode: TrainingMode = TrainingMode.QLORA
    quantization: QuantizationType = QuantizationType.NF4
    lora_config: LoRAConfig = field(default_factory=LoRAConfig)
    
    # Training hyperparameters
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    max_seq_length: int = 2048
    
    # Optimization
    fp16: bool = False
    bf16: bool = True  # Use bfloat16 if available
    gradient_checkpointing: bool = True
    optim: str = "paged_adamw_32bit"
    
    # Logging and saving
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100
    save_total_limit: int = 3
    
    # Experiment tracking
    experiment_name: str = "document-intelligence-training"
    run_name: Optional[str] = None
    use_mlflow: bool = True
    mlflow_tracking_uri: str = "mlruns"
    
    def get_training_arguments(self) -> Dict[str, Any]:
        """Get Hugging Face TrainingArguments compatible dict.
        
        Returns:
            Dictionary for TrainingArguments initialization.
        """
        return {
            "output_dir": self.output_dir,
            "num_train_epochs": self.num_epochs,
            "per_device_train_batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "warmup_ratio": self.warmup_ratio,
            "fp16": self.fp16,
            "bf16": self.bf16,
            "gradient_checkpointing": self.gradient_checkpointing,
            "optim": self.optim,
            "logging_steps": self.logging_steps,
            "save_steps": self.save_steps,
            "eval_steps": self.eval_steps,
            "save_total_limit": self.save_total_limit,
            "run_name": self.run_name,
        }
    
    def get_quantization_config(self) -> Optional[Dict[str, Any]]:
        """Get BitsAndBytes quantization configuration.
        
        Returns:
            Quantization config dict or None.
        """
        if self.quantization == QuantizationType.NONE:
            return None
        
        if self.quantization == QuantizationType.INT8:
            return {
                "load_in_8bit": True,
            }
        
        if self.quantization in (QuantizationType.INT4, QuantizationType.NF4):
            return {
                "load_in_4bit": True,
                "bnb_4bit_compute_dtype": "bfloat16",
                "bnb_4bit_quant_type": self.quantization.value,
                "bnb_4bit_use_double_quant": True,
            }
        
        return None
    
    def validate(self) -> List[str]:
        """Validate configuration.
        
        Returns:
            List of validation errors (empty if valid).
        """
        errors = []
        
        if self.learning_rate <= 0:
            errors.append("learning_rate must be positive")
        
        if self.batch_size < 1:
            errors.append("batch_size must be at least 1")
        
        if self.num_epochs < 1:
            errors.append("num_epochs must be at least 1")
        
        if self.training_mode == TrainingMode.FULL and self.quantization != QuantizationType.NONE:
            errors.append("Full fine-tuning is incompatible with quantization")
        
        output_path = Path(self.output_dir)
        if output_path.exists() and not output_path.is_dir():
            errors.append(f"output_dir {self.output_dir} exists but is not a directory")
        
        return errors


@dataclass
class DatasetConfig:
    """Configuration for training datasets.
    
    Attributes:
        train_file: Path to training data file.
        eval_file: Optional path to evaluation data file.
        text_column: Column name for text data.
        max_samples: Maximum samples to use (None for all).
        train_split_ratio: Ratio for train/eval split if no eval file.
    """
    train_file: str = ""
    eval_file: Optional[str] = None
    text_column: str = "text"
    max_samples: Optional[int] = None
    train_split_ratio: float = 0.9
    
    # Document-specific settings
    include_metadata: bool = True
    document_separator: str = "\n\n---\n\n"
