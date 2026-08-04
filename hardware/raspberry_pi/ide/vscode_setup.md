# VS Code Setup on Raspberry Pi

> **File:** `ide/vscode_setup.md`
> **Brief:** Step-by-step guide to install and configure VS Code on Raspberry Pi 4/5 for development of the Autonomous Vehicle Control System.
> **Author:** Autonomous Vehicle Team
> **Date:** 2025-01-30
> **License:** MIT

## 1. Why VS Code on Pi?

VS Code is the lightest fully-featured editor that runs natively on the
Raspberry Pi ARM64 architecture. With the **Remote-SSH** extension you can
develop on the Pi from your laptop while running the heavy ROS 2 build
natively on the Pi's ARM cores.

Two workflows are supported:

| Workflow             | Editor runs on | Build runs on | Best for                  |
|----------------------|----------------|---------------|---------------------------|
| Local on Pi          | Pi (native)    | Pi            | Field debugging           |
| Remote-SSH from host | Host x86       | Pi            | Day-to-day development    |

---

## 2. Install on the Pi (native)

### 2.1 Via apt (recommended on Ubuntu 22.04)

```bash
sudo apt update
sudo apt install -y code
```

### 2.2 Via Microsoft's ARM64 .deb

```bash
wget -O /tmp/code.deb "https://code.visualstudio.com/sha/download?build=stable&os=linux-arm64"
sudo apt install -y /tmp/code.deb
```

### 2.3 Via snap

```bash
sudo snap install code --classic
```

### 2.4 Launch

```bash
code --user-data-dir=$HOME/.vscode  # avoid root-write issues
```

Pin to the desktop with `Menu → Accessories → Code-OSS`.

---

## 3. Remote-SSH workflow (recommended)

On your workstation (x86_64 Linux/macOS/Windows):

```bash
# 1. Install VS Code on your host
#    https://code.visualstudio.com/

# 2. Install the "Remote - SSH" extension (ms-vscode-remote.remote-ssh)

# 3. Add to ~/.ssh/config:
cat <<EOF >> ~/.ssh/config
Host autovehicle-pi
    HostName autovehicle-pi.local
    User ubuntu
    IdentityFile ~/.ssh/id_ed25519
    ForwardAgent yes
    ServerAliveInterval 60
EOF

# 4. Connect from VS Code
#    Ctrl+Shift+P → "Remote-SSH: Connect to Host" → autovehicle-pi
```

VS Code will install its server component on the Pi (~150 MB) on first
connect. Subsequent connects take <5 seconds.

---

## 4. Recommended extensions

Install these on the **remote side** (Pi):

| Extension ID                            | Purpose                          |
|-----------------------------------------|----------------------------------|
| `ms-python.python`                      | Python language server           |
| `ms-python.vscode-pylance`              | Type checking                    |
| `ms-python.black-formatter`             | Code formatting                  |
| `charliermarsh.ruff`                    | Linting                          |
| `ms-iot.vscode-ros`                     | ROS 2 launch/debug               |
| `ms-vscode.cpptools`                    | C/C++ IntelliSense               |
| `ms-vscode.cpptools-extension-pack`     | CMake + clangd                   |
| `ms-azuretools.vscode-docker`           | Docker Compose UI                |
| `redhat.vscode-yaml`                    | YAML for compose/launch          |
| `platformio.platformio-ide`             | Cross-compile to STM32/ESP32     |
| `VisualStudioExptTeam.vscodeintellicode| AI completion                    |
| `eamodio.gitlens`                       | Git history                      |
| `ms-vscode.remote-explorer`             | Manage SSH targets               |

### 4.1 Bulk install (in VS Code terminal)

```bash
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension ms-python.black-formatter
code --install-extension charliermarsh.ruff
code --install-extension ms-iot.vscode-ros
code --install-extension ms-vscode.cpptools
code --install-extension ms-azuretools.vscode-docker
code --install-extension redhat.vscode-yaml
code --install-extension platformio.platformio-ide
code --install-extension eamodio.gitlens
```

---

## 5. Workspace configuration

Create `.vscode/settings.json` in the repo root:

```json
{
  "python.defaultInterpreterPath": "${workspaceFolder}/venv/bin/python",
  "python.analysis.extraPaths": [
    "/opt/ros/humble/lib/python3.10/site-packages",
    "${workspaceFolder}/python",
    "${workspaceFolder}/ros2"
  ],
  "python.analysis.typeCheckingMode": "basic",
  "python.formatting.provider": "black",
  "[python]": {
    "editor.defaultFormatter": "ms-python.black-formatter",
    "editor.formatOnSave": true,
    "editor.rulers": [100]
  },
  "files.associations": {
    "*.launch": "xml",
    "*.urdf": "xml"
  },
  "ros.distro": "humble",
  "C_Cpp.default.includePath": [
    "/opt/ros/humble/include/**",
    "/usr/include/**",
    "${workspaceFolder}/cpp"
  ],
  "C_Cpp.default.cppStandard": "c++17",
  "C_Cpp.default.cStandard": "c11",
  "C_Cpp.default.intelliSenseMode": "linux-gcc-arm64"
}
```

---

## 6. Tasks (`tasks.json`)

Create `.vscode/tasks.json`:

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "ROS2: build",
      "type": "shell",
      "command": "cd ~/ros2_ws && colcon build --symlink-install --parallel-workers 4",
      "group": { "kind": "build", "isDefault": true },
      "problemMatcher": "$gcc"
    },
    {
      "label": "ROS2: source",
      "type": "shell",
      "command": "source /opt/ros/humble/setup.bash && source ~/ros2_ws/install/setup.bash"
    },
    {
      "label": "Python: lint",
      "type": "shell",
      "command": "ruff check python/ && black --check python/"
    },
    {
      "label": "Docker: up",
      "type": "shell",
      "command": "docker compose up -d",
      "options": { "cwd": "${workspaceFolder}" }
    },
    {
      "label": "Docker: logs",
      "type": "shell",
      "command": "docker compose logs -f ros2",
      "options": { "cwd": "${workspaceFolder}" }
    }
  ]
}
```

---

## 7. Debugging configurations (`launch.json`)

### 7.1 Python — standalone module

```json
{
  "name": "Python: main.py",
  "type": "debugpy",
  "request": "launch",
  "program": "${workspaceFolder}/python/main.py",
  "args": ["--rate", "30", "--config", "config/av-01.yaml"],
  "console": "integratedTerminal",
  "justMyCode": false,
  "env": {
    "PYTHONPATH": "${workspaceFolder}/python:${workspaceFolder}/ros2"
  }
}
```

### 7.2 ROS 2 node (via vscode-ros)

```json
{
  "name": "ROS: launch perception",
  "type": "ros",
  "request": "launch",
  "target": "${workspaceFolder}/ros2/launch.py",
  "arguments": ["vehicle_id:=av-01"]
}
```

### 7.3 Attach to running Python process

```json
{
  "name": "Python: attach remote",
  "type": "debugpy",
  "request": "attach",
  "connect": { "host": "autovehicle-pi.local", "port": 5678 },
  "pathMappings": [
    { "localRoot": "${workspaceFolder}", "remoteRoot": "/home/ubuntu/av_ws" }
  ]
}
```

To enable remote attach, launch your node with:

```bash
python3 -m debugpy --listen 0.0.0.0:5678 --wait-for-client python/main.py
```

### 7.4 C/C++ (gdbserver attach)

```json
{
  "name": "C++: attach sensor_manager",
  "type": "cppdbg",
  "request": "attach",
  "program": "${workspaceFolder}/cpp/build/sensor_manager",
  "processId": "${command:pickProcess}",
  "MIMode": "gdb",
  "setupCommands": [
    { "text": "set pagination off" }
  ]
}
```

---

## 8. PlatformIO integration

Use PlatformIO to cross-compile the STM32/ESP32 companion boards *from* the
Pi:

```bash
code --install-extension platformio.platformio-ide
mkdir -p ~/pio_ws && cd ~/pio_ws
platformio init -b nucleo_f446re   # STM32 target
platformio run -t upload
```

---

## 9. Tips & gotchas

* **Memory:** Disable GitLens "current line blame" (`gitlens.currentLine.enabled: false`)
  to save ~150 MB RAM on a 4 GB Pi.
* **Linting speed:** Set `"python.analysis.indexing": false` on Pi 4 with 2 GB.
* **Folder exclusions:** Add `build/`, `install/`, `log/` to
  `files.watcherExclude` to avoid CPU spikes.
* **Keyboard:** Use `Ctrl+Shift+P` → "Preferences: Open Keyboard Shortcuts" to
  bind ROS 2 commands.
* **GPU accel:** The Pi 5 supports hardware video decode; install the
  `v4l2-requests` codecs and enable `"terminal.integrated.gpuAcceleration":
  "canvas"` if you see UI lag.

---

## 10. Useful keyboard shortcuts

| Shortcut                 | Action                          |
|--------------------------|---------------------------------|
| `Ctrl+Shift+P`           | Command palette                 |
| `Ctrl+P`                 | Quick open file                 |
| `F5`                     | Start debug                     |
| `Ctrl+Shift+B`           | Build task                      |
| `Ctrl+Shift+D`           | Open Run/Debug panel            |
| `Ctrl+\`                 | Split editor                    |
