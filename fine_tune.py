import json
import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    TrOCRProcessor, 
    VisionEncoderDecoderModel, 
    Trainer, 
    TrainingArguments,
    default_data_collator
)
from PIL import Image
import optuna
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from transformers import Trainer
import gc
import time
import shutil


class OCRDataset(Dataset):
    def __init__(self, data, processor, max_target_length=128):
        self.data = data
        self.processor = processor
        self.max_target_length = max_target_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Load and process image
        try:
            image = Image.open(item['image_path']).convert('RGB')
            
            # Resize large images to prevent GPU memory explosion
            max_size = 1024 
            if max(image.size) > max_size:
                ratio = max_size / max(image.size)
                new_size = tuple(int(dim * ratio) for dim in image.size)
                image = image.resize(new_size, Image.Resampling.LANCZOS)
            
            # Process image and text
            pixel_values = self.processor(image, return_tensors="pt").pixel_values
            labels = self.processor.tokenizer(
                item['text'],
                padding="max_length",
                max_length=self.max_target_length,
                truncation=True,
                return_tensors="pt"
            ).input_ids
            
            # Close image to free memory
            image.close()
            
            return {
                "pixel_values": pixel_values.squeeze(),
                "labels": labels.squeeze()
            }
        except Exception as e:
            print(f"Error processing item {idx}: {e}")
            # Return a dummy item to avoid crashes
            return {
                "pixel_values": torch.zeros((3, 224, 224)),
                "labels": torch.zeros(self.max_target_length, dtype=torch.long)
            }

def load_data(json_file):
    """Load and prepare data from JSON file"""
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Filter out low confidence samples if needed
    filtered_data = [item for item in data if item.get('confidence', 0) > 0.5]
    
    return filtered_data

def load_split_data(train_file, val_file, test_file=None):
    """Load pre-split data files"""
    train_data = load_data(train_file)
    val_data = load_data(val_file)
    test_data = load_data(test_file) if test_file else None
    
    return train_data, val_data, test_data

def compute_metrics(eval_pred, processor):
    """Compute CER (Character Error Rate)"""
    predictions, labels = eval_pred
    
    # Clear GPU cache during metric computation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Handle predictions format
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    # Move tensors to CPU immediately to free GPU memory
    if torch.is_tensor(predictions):
        predictions = predictions.detach().cpu().numpy()
    if torch.is_tensor(labels):
        labels = labels.detach().cpu().numpy()
    
    # Get predicted token IDs
    decoded_preds = []
    decoded_labels = []
    
    try:
        # Process in smaller batches to avoid GPU memory spikes
        batch_size = 4
        for i in range(0, len(predictions), batch_size):
            pred_batch = predictions[i:i+batch_size]
            label_batch = labels[i:i+batch_size]
            
            # Get argmax for predictions if logits
            if pred_batch.ndim == 3:
                pred_batch = np.argmax(pred_batch, axis=-1)
            
            # Replace -100 with pad token in labels
            label_batch = np.where(label_batch != -100, label_batch, processor.tokenizer.pad_token_id)
            
            pred_texts = processor.batch_decode(pred_batch, skip_special_tokens=True)
            label_texts = processor.batch_decode(label_batch, skip_special_tokens=True)
            
            decoded_preds.extend(pred_texts)
            decoded_labels.extend(label_texts)
            
            # Clear memory after each batch
            del pred_batch, label_batch, pred_texts, label_texts
            gc.collect()
    
    except Exception as e:
        print(f"Error in compute_metrics: {e}")
        return {"cer": 1.0}
    
    # DEBUG: Print first few examples to check what's happening
    if len(decoded_preds) > 0:
        print(f"\nDEBUG - Sample predictions vs labels:")
        for i in range(min(3, len(decoded_preds))):
            print(f"Pred {i}: '{decoded_preds[i]}'")
            print(f"Label {i}: '{decoded_labels[i]}'")
            print("---")
    
    # Compute CER using proper edit distance
    total_chars = 0
    total_errors = 0
    
    for pred, label in zip(decoded_preds, decoded_labels):
        if len(label.strip()) == 0:
            continue
            
        pred = pred.strip()
        label = label.strip()
        
        # Proper character-level edit distance calculation
        # Using dynamic programming for Levenshtein distance
        def edit_distance(s1, s2):
            if len(s1) == 0:
                return len(s2)
            if len(s2) == 0:
                return len(s1)
            
            # Create matrix
            matrix = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]
            
            # Initialize first row and column
            for i in range(len(s1) + 1):
                matrix[i][0] = i
            for j in range(len(s2) + 1):
                matrix[0][j] = j
            
            # Fill matrix
            for i in range(1, len(s1) + 1):
                for j in range(1, len(s2) + 1):
                    if s1[i-1] == s2[j-1]:
                        cost = 0
                    else:
                        cost = 1
                    
                    matrix[i][j] = min(
                        matrix[i-1][j] + 1,      # deletion
                        matrix[i][j-1] + 1,      # insertion
                        matrix[i-1][j-1] + cost  # substitution
                    )
            
            return matrix[len(s1)][len(s2)]
        
        errors = edit_distance(pred, label)
        total_errors += errors
        total_chars += len(label)
    
    cer = total_errors / max(total_chars, 1)
    
    print(f"DEBUG - Total errors: {total_errors}, Total chars: {total_chars}, CER: {cer}")
    
    # Final cleanup
    del decoded_preds, decoded_labels
    gc.collect()
    
    return {"cer": cer}

class TrOCRTrainer(Trainer):
    """Custom trainer with better memory management"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.writer = SummaryWriter(log_dir=self.args.logging_dir)
        # Track GPU memory usage
        self.memory_threshold = 0.9  # 90% of GPU memory

    def log(self, logs, start_time=None):
        # Call parent log method with correct signature
        super().log(logs, start_time=start_time)

        # Log metrics to TensorBoard
        if logs and hasattr(self, 'writer'):
            for key, value in logs.items():
                if isinstance(value, (int, float)) and hasattr(self.state, 'global_step'):
                    self.writer.add_scalar(key, value, self.state.global_step)
            
            # Log GPU memory usage
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                self.writer.add_scalar('gpu_memory_usage', memory_used, self.state.global_step)
                    
        # More aggressive cleanup based on memory usage
        if hasattr(self.state, 'global_step'):
            if torch.cuda.is_available():
                memory_used = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                if memory_used > self.memory_threshold or self.state.global_step % 50 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override training step for better memory management"""
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            if memory_used > self.memory_threshold:
                torch.cuda.empty_cache()
                gc.collect()
        
        # Call parent training_step method
        loss = super().training_step(model, inputs, num_items_in_batch)
        
        # More frequent cache clearing during training
        if torch.cuda.is_available() and self.state.global_step % 5 == 0:  # More frequent
            torch.cuda.empty_cache()
        
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Override prediction step for memory management during evaluation"""
        # Clear cache before prediction
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Call parent method
        with torch.no_grad():  # Ensure no gradients are computed
            outputs = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)
        
        # Clear cache after prediction
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return outputs

    def on_train_end(self):
        """Close writer and cleanup on training end"""
        super().on_train_end()
        if hasattr(self, "writer"):
            self.writer.close()
        
        # Force cleanup
        gc.collect()
        torch.cuda.empty_cache()


def cleanup_trial_files(trial_number):
    """Clean up files from previous trial"""
    trial_dir = f"./trocr_trial_{trial_number}"
    log_dir = f"./logs/trial_{trial_number}"
    
    for directory in [trial_dir, log_dir]:
        if os.path.exists(directory):
            try:
                shutil.rmtree(directory)
            except Exception as e:
                print(f"Warning: Could not remove {directory}: {e}")


def get_safe_batch_size():
    """Determine safe batch size based on available GPU memory"""
    if not torch.cuda.is_available():
        return 2
    
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9  # GB
    
    # Conservative batch size calculation
    if total_memory < 6:  # Less than 6GB
        return 1
    elif total_memory < 12:  # 6-12GB
        return 2
    elif total_memory < 24:  # 12-24GB
        return 4
    else:  # 24GB+
        return 8


def objective(trial):
    """Optuna objective function with better memory management"""
    
    # Clean up from previous trial
    if trial.number > 0:
        cleanup_trial_files(trial.number - 1)
    
    # Initial cleanup
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
    
    try:
        # Use safe batch size based on GPU memory
        max_batch_size = get_safe_batch_size()
        
        # Hyperparameters to optimize
        learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)  # Lower max
        batch_size = trial.suggest_int("batch_size", 1, max_batch_size)
        num_epochs = trial.suggest_int("num_epochs", 15, 25)  # Much longer!
        warmup_steps = trial.suggest_int("warmup_steps", 200, 800)  # More warmup
        
        # Load processor and model
        processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
        model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
        
        # Set special tokens
        model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
        model.config.pad_token_id = processor.tokenizer.pad_token_id
        model.config.vocab_size = model.config.decoder.vocab_size
        
        model.config.max_length = 128
        model.config.early_stopping = True
        model.config.no_repeat_ngram_size = 3
        model.config.length_penalty = 2.0
        model.config.num_beams = 4
        
        # Enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable()
        
        # Load pre-split data
        train_data, val_data, _ = load_split_data("output2/train.json", "output2/validation.json")
        
        # Limit dataset size based on GPU memory
        # total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 4
        if len(train_data) > 5000:
            max_train_samples = min(5000, len(train_data))  # Use up to 5000
            max_val_samples = min(1000, len(val_data))      # Use up to 1000
        else:
            max_train_samples = len(train_data)  # Use all available data
            max_val_samples = len(val_data)
            
        train_data = train_data[:max_train_samples]
        val_data = val_data[:max_val_samples]
        
        # Create datasets
        train_dataset = OCRDataset(train_data, processor)
        val_dataset = OCRDataset(val_data, processor)
        
        # Calculate gradient accumulation to maintain effective batch size
        effective_batch_size = 8  # Target effective batch size
        gradient_accumulation_steps = max(1, effective_batch_size // batch_size)
        
        # Training arguments with memory optimizations
        training_args = TrainingArguments(
            output_dir=f"./trocr_trial_{trial.number}",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=max(1, batch_size // 2),
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            
            # Less frequent evaluation (faster training)
            logging_steps=50,
            eval_steps=500,  # Less frequent evaluation
            eval_strategy="steps",
            
            # Save less frequently (faster training)
            save_steps=1000,
            save_total_limit=1,
            
            # Learning rate scheduling
            lr_scheduler_type="cosine",  # Better than default linear
            
            remove_unused_columns=False,
            push_to_hub=False,
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            load_best_model_at_end=True,
            metric_for_best_model="cer",
            greater_is_better=False,
            report_to="tensorboard",
            logging_dir=f"./logs/trial_{trial.number}",
            fp16=True,
            gradient_checkpointing=True,
            dataloader_drop_last=True,
            optim="adamw_torch",
            adam_epsilon=1e-8,
            max_grad_norm=1.0,
            eval_accumulation_steps=1,
            past_index=-1,
            
            # Add weight decay for better generalization
            weight_decay=0.01,
        )
        
        # Create trainer
        trainer = TrOCRTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=lambda eval_pred: compute_metrics(eval_pred, processor),  # Pass processor here
            data_collator=default_data_collator,
        )
        
        # Log hyperparameters
        trainer.writer.add_hparams(
            {
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "num_epochs": num_epochs,
                "warmup_steps": warmup_steps,
                "gradient_accumulation_steps": gradient_accumulation_steps,
            },
            {"trial_number": trial.number}
        )
        
        # Train model
        trainer.train()
        
        # Evaluate
        eval_results = trainer.evaluate()
        cer_score = eval_results["eval_cer"]
        
        # Cleanup trainer
        if hasattr(trainer, 'writer'):
            trainer.writer.close()
        
        del trainer
        del model
        del train_dataset
        del val_dataset
        del processor
        del training_args
        
        # Force garbage collection
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()
        
        # Wait a bit for cleanup
        time.sleep(3)
        
        return cer_score
        
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"Trial {trial.number} failed due to GPU OOM: {e}")
            # Handle OOM gracefully
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                torch.cuda.reset_peak_memory_stats()
            time.sleep(5)  # Wait longer after OOM
            return 1.0  # Return high error to discourage similar configurations
        else:
            raise e
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        
        # Cleanup on failure
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        
        # Return a high error rate to mark this trial as failed
        return 1.0


def run_optimization():
    """Run Optuna optimization with better error handling"""
    
    # Create study
    study = optuna.create_study(
        direction="minimize",
        study_name="trocr_optimization",
        storage="sqlite:///trocr_optuna2.db",
        load_if_exists=True
    )
    
    # Reduce number of trials for memory-constrained environments
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 4
    n_trials = 3 if total_memory < 8 else 5
    
    successful_trials = 0
    
    for i in range(n_trials):
        try:
            print(f"Starting trial {i+1}/{n_trials}")
            
            # Memory check before each trial
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                available_memory = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()
                print(f"Available GPU memory: {available_memory / 1e9:.1f} GB")
            
            study.optimize(objective, n_trials=1, gc_after_trial=True)
            successful_trials += 1
            print(f"Trial {i+1} completed successfully")
            
            # Force cleanup between trials
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(2)
            
        except Exception as e:
            print(f"Trial {i+1} failed: {e}")
            # Continue with next trial
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            time.sleep(5)  # Wait longer after failure
    
    print(f"Completed {successful_trials}/{n_trials} trials successfully")
    
    if successful_trials > 0:
        # Print best parameters
        print("Best trial:")
        print(f"  Value: {study.best_value}")
        print("  Params: ")
        for key, value in study.best_params.items():
            print(f"    {key}: {value}")
        
        return study.best_params
    else:
        print("No successful trials completed")
        return None

def train_final_model(best_params):
    """Train final model with best parameters"""
    
    if best_params is None:
        print("No best parameters available, using defaults")
        # Conservative default parameters
        max_batch_size = get_safe_batch_size()
        best_params = {
            "learning_rate": 5e-5,
            "batch_size": max_batch_size,
            "num_epochs": 3,
            "warmup_steps": 200
        }
    
    # Force cleanup before final training
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    # Load processor and model
    processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-printed")
    model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-printed")
    
    # Set special tokens
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    
    model.config.max_length = 128
    model.config.early_stopping = True
    model.config.no_repeat_ngram_size = 3
    model.config.length_penalty = 2.0
    model.config.num_beams = 4
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # Load pre-split data
    train_data, val_data, _ = load_split_data("output2/train.json", "output2/validation.json")
    
    # Create datasets
    train_dataset = OCRDataset(train_data, processor)
    val_dataset = OCRDataset(val_data, processor)
    
    # Calculate gradient accumulation
    effective_batch_size = 8
    gradient_accumulation_steps = max(1, effective_batch_size // best_params["batch_size"])
    
    training_args = TrainingArguments(
        output_dir="./trocr_final",
        per_device_train_batch_size=best_params["batch_size"],
        per_device_eval_batch_size=max(1, best_params["batch_size"] // 2),
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=best_params["num_epochs"],
        learning_rate=best_params["learning_rate"],
        warmup_steps=best_params["warmup_steps"],
        logging_steps=25,
        eval_steps=100,
        eval_strategy="steps",
        save_steps=600,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=False,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        report_to="tensorboard",
        logging_dir="./logs/final_model",
        fp16=True,
        gradient_checkpointing=True,
        dataloader_drop_last=True,
        optim="adamw_torch",
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        eval_accumulation_steps=1,
        past_index=-1,
    )
    
    trainer = TrOCRTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=lambda eval_pred: compute_metrics(eval_pred, processor),  # Pass processor here
        data_collator=default_data_collator,
    )
    
    # Log final model hyperparameters
    trainer.writer.add_hparams(best_params, {"final_model": 1})
    
    # Train model
    trainer.train()
    
    # Save final model
    trainer.save_model("./trocr_final_model")
    processor.save_pretrained("./trocr_final_model")
    
    print("Final model saved to ./trocr_final_model")


if __name__ == "__main__":
    # Ensure CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"GPU Memory Available: {(torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9:.1f} GB")
        
        # Set memory fraction to prevent OOM
        torch.cuda.set_per_process_memory_fraction(0.95)  # Use 95% of available memory
    
    # Run optimization
    print("Starting hyperparameter optimization...")
    best_params = run_optimization()
    
    # Clean up before final training
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    time.sleep(3)
    
    # Train final model
    print("Training final model with best parameters...")
    train_final_model(best_params)
    
    print("Training complete!")
    print("View TensorBoard logs with: tensorboard --logdir=./logs")