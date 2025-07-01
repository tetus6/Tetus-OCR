from ocr_data_pipeline import OCRDataPipeline


def main():

    # Initialize pipeline
    pipeline = OCRDataPipeline(
        input_dir="tobacco3482/Tobacco3482-jpg/Letter",
        output_dir="output2",
        model_name="microsoft/trocr-base-printed",
        languages=['en']  # Add target languages
    )

    # Process dataset
    pipeline.process_dataset(
        apply_augmentation=False,
        max_samples=1000  # Limit for testing
    )

    # Save in multiple formats
    pipeline.save_dataset(format_type="json")
    pipeline.save_dataset(format_type="huggingface")

    # Visualize samples
    pipeline.visualize_samples(num_samples=3)

    print("Data preparation complete!")

if __name__ == "__main__":
    main()


        
