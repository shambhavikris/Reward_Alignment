import gdown

# Define the URL of the zip file on Google Drive
url = 'https://drive.google.com/file/d/10F1SMXpriqA1FVLieUjihb94Jaj8nwHt/view?usp=drive_link'

# Define the output file name for the downloaded zip file
output = 'sft_sft_0_202404201349_step20000.pt'

# Download the zip file using gdown
gdown.download(url, output, quiet=False)