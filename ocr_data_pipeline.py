import os
import json
import numpy as np
import pandas as pd
import torch
import logging
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from pathlib import Path
from typing import List, Tuple, Dict, Optional

from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.model_selection import train_test_split

import easyocr
import pytesseract

import albumentations as A

from transformers import TrOCRProcessor, VisionEncoderDecoderModel

class OCRDataPipeline:
    """
    Comprehensive pipeline for preparing OCR training data with automated labeling
    """

    def __init__(self,
                 input_dir: str,
                 output_dir: str,
                 model_name: str = "microsoft/trocr-base-printed",
                 use_easyocr: bool = True,
                 use_tesseract: bool = True,
                 languages: List[str] = ['en']):
        """
        Initialize the OCR data preparation pipeline

        Args:
            input_dir: Directory containing raw images
            output_dir: Directory to save processed data
            model_name: HuggingFace model name for TrOCR
            use_easyocr: Whether to use EasyOCR for labeling
            use_tesseract: Whether to use Tesseract for labeling
            languages: List of language codes for OCR engines
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.model_name = model_name
        self.languages = languages

        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "images").mkdir(exist_ok=True)
        (self.output_dir / "processed").mkdir(exist_ok=True)

        # Initialize OCR engines
        self.ocr_engines = {}
        if use_easyocr:
            self.ocr_engines['easyocr'] = easyocr.Reader(languages)
        if use_tesseract:
            self.ocr_engines['tesseract'] = pytesseract

        # Initialize TrOCR for validation/ensemble
        self.processor = TrOCRProcessor.from_pretrained(model_name, use_fast=False)
        self.trocr_model = VisionEncoderDecoderModel.from_pretrained(model_name)

        # Data storage
        self.dataset = []

    def validate_image(self, image_path: Path) -> bool:
        """Validate if image is suitable for OCR processing"""
        try:
            img = Image.open(image_path)
            # Check image size
            if img.size[0] < 32 or img.size[1] < 32:
                return False
            # Check if image has text-like content (basic heuristic)
            img_array = np.array(img.convert('L'))
            if np.std(img_array) < 10:  # Too uniform, likely no text
                return False
            return True
        except Exception as e:
            logger.warning(f"Invalid image {image_path}: {e}")
            return False

    def preprocess_image(self, image: Image.Image) -> Image.Image:
        """Apply preprocessing to improve OCR accuracy"""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Apply preprocessing steps
        # 1. Resize if too large (maintaining aspect ratio)
        max_size = 1024
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        # 2. Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)

        # 3. Enhance sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)

        return image

    def extract_text_easyocr(self, image: Image.Image) -> Tuple[str, float]:
        """Extract text using EasyOCR"""
        try:
            img_array = np.array(image)
            results = self.ocr_engines['easyocr'].readtext(img_array)

            if not results:
                return "", 0.0

            # Combine all detected text
            text_parts = []
            confidences = []

            for (bbox, text, confidence) in results:
                text_parts.append(text)
                confidences.append(confidence)

            combined_text = " ".join(text_parts)
            avg_confidence = np.mean(confidences) if confidences else 0.0

            return combined_text, avg_confidence
        except Exception as e:
            logger.error(f"EasyOCR error: {e}")
            return "", 0.0

    def extract_text_tesseract(self, image: Image.Image) -> Tuple[str, float]:
        """Extract text using Tesseract"""
        try:
            # Configure Tesseract
            config = '--oem 3 --psm 6'  # LSTM OCR Engine, uniform text block

            # Get text and confidence
            data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)

            # Filter out low confidence detections
            confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
            texts = [data['text'][i] for i, conf in enumerate(data['conf']) if int(conf) > 30]

            combined_text = " ".join(texts).strip()
            avg_confidence = np.mean(confidences) / 100.0 if confidences else 0.0

            return combined_text, avg_confidence
        except Exception as e:
            logger.error(f"Tesseract error: {e}")
            return "", 0.0

    def extract_text_trocr(self, image: Image.Image) -> Tuple[str, float]:
        """Extract text using TrOCR (for validation/ensemble)"""
        try:
            pixel_values = self.processor(image, return_tensors="pt").pixel_values

            with torch.no_grad():
                generated_ids = self.trocr_model.generate(pixel_values)
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # TrOCR doesn't provide confidence scores directly
            # We'll use a heuristic based on text length and content
            confidence = min(len(generated_text.split()) / 10, 1.0)

            return generated_text, confidence
        except Exception as e:
            logger.error(f"TrOCR error: {e}")
            return "", 0.0

    def ensemble_ocr(self, image: Image.Image) -> Tuple[str, float, Dict]:
        """Combine results from multiple OCR engines"""
        results = {}

        # Get results from all available engines
        if 'easyocr' in self.ocr_engines:
            text, conf = self.extract_text_easyocr(image)
            results['easyocr'] = {'text': text, 'confidence': conf}

        if 'tesseract' in self.ocr_engines:
            text, conf = self.extract_text_tesseract(image)
            results['tesseract'] = {'text': text, 'confidence': conf}

        # Add TrOCR for validation
        text, conf = self.extract_text_trocr(image)
        results['trocr'] = {'text': text, 'confidence': conf}

        # Select best result based on confidence
        best_engine = max(results.keys(), key=lambda k: results[k]['confidence'])
        best_text = results[best_engine]['text']
        best_confidence = results[best_engine]['confidence']

        return best_text, best_confidence, results

    def apply_augmentations(self, image: Image.Image) -> List[Image.Image]:
        """Apply data augmentation to increase dataset diversity"""
        augmented_images = [image]  # Include original

        # Convert to numpy for albumentations
        img_array = np.array(image)

        # Define augmentation pipeline
        transform = A.Compose([
            A.OneOf([
                A.GaussNoise(var_limit=(10, 50), p=0.5),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.5),
            ], p=0.3),
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=0.5),
                A.GaussianBlur(blur_limit=3, p=0.5),
            ], p=0.2),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.2),
            A.Rotate(limit=2, p=0.3),
            A.OneOf([
                A.ElasticTransform(alpha=50, sigma=10, p=0.5),
                A.GridDistortion(num_steps=5, distort_limit=0.1, p=0.5),
            ], p=0.1)
        ])

        # Generate augmented versions
        for i in range(2):  # Create 2 augmented versions
            try:
                augmented = transform(image=img_array)
                aug_image = Image.fromarray(augmented['image'])
                augmented_images.append(aug_image)
            except Exception as e:
                logger.warning(f"Augmentation failed: {e}")

        return augmented_images

    def process_single_image(self, image_path: Path, apply_augmentation: bool = True) -> List[Dict]:
        """Process a single image and return dataset entries"""
        if not self.validate_image(image_path):
            return []

        try:
            # Load and preprocess image
            image = Image.open(image_path)
            image = self.preprocess_image(image)

            # Apply augmentation if requested
            images_to_process = self.apply_augmentations(image) if apply_augmentation else [image]

            entries = []
            for idx, img in enumerate(images_to_process):
                # Extract text using ensemble method
                text, confidence, all_results = self.ensemble_ocr(img)

                # Skip if confidence is too low or text is empty
                if confidence < 0.3 or len(text.strip()) < 2:
                    continue

                # Generate unique filename
                suffix = f"_aug_{idx}" if idx > 0 else ""
                image_filename = f"{image_path.stem}{suffix}.jpg"
                processed_image_path = self.output_dir / "images" / image_filename

                # Save processed image
                img.save(processed_image_path, "JPEG", quality=95)

                # Create dataset entry
                entry = {
                    'image_path': str(processed_image_path),
                    'text': text.strip(),
                    'confidence': confidence,
                    'original_path': str(image_path),
                    'ocr_results': all_results,
                    'image_size': img.size,
                    'augmented': idx > 0
                }
                entries.append(entry)

            return entries

        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return []

    def process_dataset(self, apply_augmentation: bool = True, max_samples: Optional[int] = None):
        """Process entire dataset"""
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = [
            f for f in self.input_dir.rglob('*')
            if f.suffix.lower() in image_extensions
        ]

        if max_samples:
            image_files = image_files[:max_samples]

        logger.info(f"Processing {len(image_files)} images...")

        for image_path in tqdm(image_files, desc="Processing images"):
            entries = self.process_single_image(image_path, apply_augmentation)
            self.dataset.extend(entries)

        logger.info(f"Generated {len(self.dataset)} training samples")

    def create_train_val_split(self, val_ratio: float = 0.2, test_ratio: float = 0.1):
        """Split dataset into train/validation/test sets"""
        if len(self.dataset) == 0:
            raise ValueError("No data to split. Run process_dataset first.")

        # First split: train+val / test
        train_val, test = train_test_split(
            self.dataset,
            test_size=test_ratio,
            random_state=42,
            stratify=None 
        )

        # Second split: train / val
        train, val = train_test_split(
            train_val,
            test_size=val_ratio/(1-test_ratio),
            random_state=42
        )

        return train, val, test

    def save_dataset(self, format_type: str = "json"):
        """Save processed dataset in specified format"""
        train, val, test = self.create_train_val_split()

        splits = {
            'train': train,
            'validation': val,
            'test': test
        }

        for split_name, split_data in splits.items():
            if format_type == "json":
                output_file = self.output_dir / f"{split_name}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(split_data, f, indent=2, ensure_ascii=False)

            elif format_type == "csv":
                output_file = self.output_dir / f"{split_name}.csv"
                df = pd.DataFrame(split_data)
                df.to_csv(output_file, index=False)

            elif format_type == "huggingface":
                # Create HuggingFace dataset format
                hf_data = []
                for item in split_data:
                    hf_data.append({
                        'image': item['image_path'],
                        'text': item['text']
                    })

                output_file = self.output_dir / f"{split_name}_hf.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(hf_data, f, indent=2, ensure_ascii=False)

        # Save metadata
        metadata = {
            'total_samples': len(self.dataset),
            'train_samples': len(train),
            'val_samples': len(val),
            'test_samples': len(test),
            'model_used': self.model_name,
            'languages': self.languages,
            'ocr_engines': list(self.ocr_engines.keys())
        }

        with open(self.output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Dataset saved in {format_type} format")
        logger.info(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    def visualize_samples(self, num_samples: int = 5):
        """Visualize sample images with their extracted text"""
        if len(self.dataset) == 0:
            logger.warning("No data to visualize")
            return

        samples = np.random.choice(self.dataset, min(num_samples, len(self.dataset)), replace=False)

        fig, axes = plt.subplots(num_samples, 1, figsize=(12, 4*num_samples))
        if num_samples == 1:
            axes = [axes]

        for idx, sample in enumerate(samples):
            img = Image.open(sample['image_path'])
            axes[idx].imshow(img)
            axes[idx].set_title(f"Text: {sample['text'][:100]}...\nConfidence: {sample['confidence']:.2f}")
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig(self.output_dir / "sample_visualization.png", dpi=150, bbox_inches='tight')
        plt.show()