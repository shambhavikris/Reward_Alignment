import gdown

# Define the URL of the zip file on Google Drive
url = 'https://drive.google.com/uc?id=10F1SMXpriqA1FVLieUjihb94Jaj8nwHt'

# Define the output file name for the downloaded zip file
output = 'sft.zip'

# Download the zip file using gdown
gdown.download(url, output, quiet=False)