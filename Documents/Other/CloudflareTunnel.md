# Label Studio Remote Deployment with Cloudflare Tunnel

## Goal

Expose a local Label Studio instance to students through:

```text
https://labelstudio.aflucas.com
```

Label Studio runs locally on:

```text
http://localhost:8080
```

Cloudflare Tunnel forwards public HTTPS traffic to that local port. Students do not need VPN access or your IP address.

---

## Architecture

```text
Student browser
      |
      v
https://labelstudio.aflucas.com
      |
      v
Cloudflare Tunnel
      |
      v
Host machine / DGX
Label Studio at http://localhost:8080
```

The tunnel must run on the same machine that is running Label Studio.

---

## Important Files

Cloudflare creates two local files:

```text
cert.pem
<TUNNEL_ID>.json
```

These are private credentials. Do not commit them to Git.

Recommended location:

```text
~/.cloudflared/
```

On Windows this usually means:

```text
%USERPROFILE%\.cloudflared\
```

For this repository, keep Cloudflare credentials outside Git. The documentation is enough to recreate the setup.

---

## Prerequisites

You need:

- access to the Cloudflare account that manages `aflucas.com`;
- `cloudflared` installed on the host machine;
- Label Studio running locally on port `8080`;
- a Label Studio admin account/password for users to log in.

Cloudflare Tunnel credentials are separate from Label Studio login credentials.

---

## 1. Install `cloudflared`

### Windows

Install using one of these methods.

With Winget:

```powershell
winget install Cloudflare.cloudflared
```

Or download it manually from:

```text
https://github.com/cloudflare/cloudflared/releases/latest
```

After installation, confirm PowerShell can see it:

```powershell
cloudflared --version
```

### macOS

```bash
brew install cloudflared
cloudflared --version
```

### Linux / DGX

```bash
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x cloudflared-linux-amd64
sudo mv cloudflared-linux-amd64 /usr/local/bin/cloudflared
cloudflared --version
```

---

## 2. Start Label Studio Locally

From the repository root:

```powershell
.\Scripts\MTSD-Scripts\LabelStudio\start_labelstudio.ps1
```

Confirm it works locally:

```text
http://localhost:8080
```

If this local URL does not work, fix Label Studio before working on the tunnel.

---

## 3. Authenticate Cloudflare

Run:

```powershell
cloudflared tunnel login
```

A browser window opens.

Select:

```text
aflucas.com
```

After login, Cloudflare writes a certificate file locally.

On Windows it is usually:

```text
%USERPROFILE%\.cloudflared\cert.pem
```

Check it exists:

```powershell
Get-ChildItem "$env:USERPROFILE\.cloudflared"
```

---

## 4. Create the Tunnel

Create a named tunnel:

```powershell
cloudflared tunnel create labelstudio
```

This creates a tunnel and prints a tunnel ID.

It also creates a credentials JSON file like:

```text
%USERPROFILE%\.cloudflared\<TUNNEL_ID>.json
```

List tunnels:

```powershell
cloudflared tunnel list
```

You should see a tunnel named:

```text
labelstudio
```

Copy the tunnel ID. You need it in the config file.

---

## 5. Route the DNS Name

Connect the tunnel to the public hostname:

```powershell
cloudflared tunnel route dns labelstudio labelstudio.aflucas.com
```

This creates the Cloudflare DNS route automatically.

---

## 6. Create the Cloudflare Config File

Create or edit:

```text
%USERPROFILE%\.cloudflared\config.yml
```

PowerShell:

```powershell
notepad "$env:USERPROFILE\.cloudflared\config.yml"
```

Use this content, replacing `<TUNNEL_ID>` with the real tunnel ID:

```yaml
tunnel: labelstudio
credentials-file: <absolute path to your .cloudflared>\<TUNNEL_ID>.json

ingress:
  - hostname: labelstudio.aflucas.com
    service: http://localhost:8080
  - service: http_status:404
```

Example Windows path shape:

```yaml
credentials-file: %USERPROFILE%\.cloudflared\f40eb9d6-4a86-4756-8eae-df45718d6865.json
```

Important:

- `service` must be `http://localhost:8080`.
- The JSON file path must point to the real `<TUNNEL_ID>.json`.
- Keep the config in `.cloudflared`, not in Git.

---

## 7. Run the Tunnel

Keep Label Studio running in one terminal.

In a second terminal, run:

```powershell
cloudflared tunnel run labelstudio
```

If the config is not in the default location, pass it explicitly:

```powershell
cloudflared --config "$env:USERPROFILE\.cloudflared\config.yml" tunnel run labelstudio
```

When the tunnel is running, open:

```text
https://labelstudio.aflucas.com
```

---

## 8. Daily Usage

Each time you want the public Label Studio link to work:

1. Start Label Studio:

```powershell
.\Scripts\MTSD-Scripts\LabelStudio\start_labelstudio.ps1
```

2. In another terminal, start the tunnel:

```powershell
cloudflared tunnel run labelstudio
```

3. Share this URL:

```text
https://labelstudio.aflucas.com
```

If either Label Studio or `cloudflared` stops, the public URL stops working.

---

## Quick Temporary Test

Before setting up DNS, you can test with a temporary Cloudflare URL:

```powershell
cloudflared tunnel --url http://localhost:8080
```

Cloudflare prints a temporary URL like:

```text
https://random-name.trycloudflare.com
```

This is useful for testing, but it is not the permanent `labelstudio.aflucas.com` setup.

---

## User Access

Cloudflare Tunnel only exposes the web app.

Label Studio controls who can log in and what they can do.

Before sharing the public URL:

- use a strong Label Studio admin password;
- create student accounts if needed;
- avoid sharing admin credentials with annotators;
- assign students only the permissions they need.

Anyone with valid Label Studio credentials can access it from a different IP address through the tunnel.

---

## Troubleshooting

### `config.yml` not found

Error:

```text
open %USERPROFILE%/.cloudflared/config.yml: The system cannot find the path specified.
```

Cause:

You are pointing `cloudflared` to a config file that does not exist.

Fix:

Use the default config path:

```powershell
cloudflared tunnel run labelstudio
```

Or create/pass the real config file:

```powershell
notepad "$env:USERPROFILE\.cloudflared\config.yml"
cloudflared --config "$env:USERPROFILE\.cloudflared\config.yml" tunnel run labelstudio
```

### Label Studio not reachable

Check:

```powershell
curl http://localhost:8080
```

If this fails, start Label Studio first.

### Tunnel exists but domain does not work

Re-run:

```powershell
cloudflared tunnel route dns labelstudio labelstudio.aflucas.com
```

### Wrong port

This setup uses:

```text
8080
```

So the Cloudflare config must contain:

```yaml
service: http://localhost:8080
```

### Remove local Cloudflare credentials from this machine

This removes local Cloudflare tunnel files from the current Windows user profile:

```powershell
Remove-Item "$env:USERPROFILE\.cloudflared" -Recurse -Force
```

This does not delete the tunnel from the Cloudflare account. To delete the tunnel from Cloudflare:

```powershell
cloudflared tunnel delete labelstudio
```

Only delete the Cloudflare tunnel if you are sure you no longer need `labelstudio.aflucas.com`.

### Uninstall `cloudflared`

If installed with Winget:

```powershell
winget uninstall Cloudflare.cloudflared
```

If installed manually, delete the `cloudflared.exe` file from wherever you placed it.

---

## Summary

The permanent setup is:

```text
Label Studio: http://localhost:8080
Tunnel name: labelstudio
Public URL: https://labelstudio.aflucas.com
Config path: ~/.cloudflared/config.yml
Credential path: ~/.cloudflared/<TUNNEL_ID>.json
```
