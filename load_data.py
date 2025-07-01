import kagglehub

# Download latest version
path = kagglehub.dataset_download("patrickaudriaz/tobacco3482jpg")

print("Path to dataset files:", path)