import os
import gdown

def DownloadFromRemote(share_url, output_path, overwrite=False, verbose=False):
    """
    Downloads a file from Google Drive using a shareable link.

    Parameters:
        share_url (str): URL to the shared file on Google Drive
        output_path (str): Path where the file should be saved
        overwrite (bool): If True, overwrites the file if it already exists.
                          If False, skips download if file exists. Default is False.
    """
    # Check if the file exists and handle based on overwrite flag
    if os.path.exists(output_path) and not overwrite:
        print(f"File already exists at {output_path}. Set overwrite=True to replace it.")
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True) # Create file's directory if it doesn't exist
    gdown.download(share_url, output_path, quiet=not verbose, fuzzy=True) # Download the file