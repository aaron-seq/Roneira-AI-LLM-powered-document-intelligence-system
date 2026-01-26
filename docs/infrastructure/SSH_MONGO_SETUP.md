# SSH Port Forwarding & MongoDB Setup Guide

**Goal**: Connect to a remote MongoDB instance securely via SSH tunneling, allowing you to use local tools (Compass, mongosh) as if the DB were on `localhost`.

## 1. Generate SSH Key (Local Machine)
If you don't have an SSH key yet:
```bash
ssh-keygen -t rsa -b 4096 -C "your_email@example.com"
# Press Enter to save to default location (~/.ssh/id_rsa)
```

## 2. Add Public Key to Remote VM
1.  Copy your public key:
    ```bash
    cat ~/.ssh/id_rsa.pub
    ```
    *(Windows Powershell: `Get-Content ~/.ssh/id_rsa.pub`)*
2.  Log into the remote VM (or ask an admin).
3.  Append the key to `~/.ssh/authorized_keys`:
    ```bash
    echo "ssh-rsa AAAA..." >> ~/.ssh/authorized_keys
    chmod 600 ~/.ssh/authorized_keys
    ```

## 3. Establish SSH Tunnel
Forward remote port `27024` (MongoDB) to local port `27033`.

```bash
# Syntax: ssh -L [LocalPort]:[RemoteHost]:[RemotePort] [User]@[VM_IP] -i [PrivateKey]
ssh -L 27033:localhost:27024 user@remote-vm-ip -i ~/.ssh/id_rsa -N -f
```

*   `-L 27033:localhost:27024`: Forwards local 27033 to remote's localhost:27024.
*   `-N`: Do not execute a remote command (just forward ports).
*   `-f`: Go to background.

## 4. Connect via MongoDB Compass
1.  Open MongoDB Compass.
2.  New Connection string:
    ```
    mongodb://localhost:27033
    ```
3.  Connect! You are now accessing the remote DB securely.
