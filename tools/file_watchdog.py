#%%
import paramiko
from paramiko import SSHClient
from scp import SCPClient
import time
import re
import json
import os
import requests
from tqdm import tqdm
import sys


with open(sys.argv[1]) as f:
    config = json.load(f)
    
print('Starting the Watchdog...')

file_pattern = '*.pickle'

# Create an SSH client instance
ssh = SSHClient()
ssh.load_system_host_keys()
ssh.connect(config['server'], port=config['port'], username=config['user'], password=config['password'])


# Function to check for new files
def check_for_new_files():
    # stdin, stdout, stderr = ssh.exec_command('ls -1 ' + config['remote_path'])
    command = f"find {config['remote_path']} -type f -name '{file_pattern}'"
    _, stdout, _ = ssh.exec_command(command)
    remote_files = stdout.read().decode().splitlines()
    
    return [f for f in remote_files if os.path.basename(f) not in os.listdir(config['local_path'])]


def get_remote_file_size(ssh, remote_file_path):
    stdin, stdout, stderr = ssh.exec_command(f"stat -c %s {remote_file_path}")
    exit_status = stdout.channel.recv_exit_status()
    if exit_status == 0:
        return int(stdout.read().decode().strip())
    else:
        print(f"Error getting file size for {remote_file_path}")
        return None


def progress(filename, size, sent):
    percent_complete = float(sent) / float(size) * 100.
    print(f"Downloading \"{os.path.basename(filename)}\": {percent_complete:.1f}% \r", end='', flush=True)


def transfer_files(remote_files, ssh, config):
    with SCPClient(ssh.get_transport(), progress=progress) as scp:
        for file in remote_files:
            scp.get(file, f"{config['local_path']}/{os.path.basename(file)}")


# Main monitoring loop
try:
    while True:
        new_files = check_for_new_files()
        if new_files:
            print("\nNew file(s) detected:")
            for x in new_files: print('--- ', x)
            print("\nTransferring file(s)...\n")
            transfer_files(new_files, ssh, config)
            print("Transfer complete.", flush=True)
        time.sleep(60)
        
except KeyboardInterrupt:
    print("Watchdog is interrupted by user.")
    
finally:
    ssh.close()
