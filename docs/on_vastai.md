# Start an RTX4090 instance on vast.ai

1. Sign up for [vast.ai](https://vast.ai/), and use credit card to buy credits ($10 will be enough)
1. Search for an europe-based instance with 256 GB of Disk space, with 1x RTX4090, using search bar in vast.ai (eu, because lower latency)

![alt text](vastai.png)

1. "Edit Image" -> Select pytorch:latest docker image. 

![alt text](pytorch.png)

# SSH into the instance

1. First, grab the ssh connection info:

![alt text](ssh.png)

1. Paste it into terminal, and say "yes". Check you have "nvidia-smi" available. This means GPU is online.

1. Open VSCode, and install "remote development extension pack", from extensions.

1. Press ctrl+p, and type "add new ssh host", and paste in the full "ssh connection info", like "ssh -p 51167 root@85.167.26.137"

1. Connect.

You can now jump back to the [README.md](../README.md) and follow the quickstart guide there.