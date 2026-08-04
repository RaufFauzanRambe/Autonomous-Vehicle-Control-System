# PyCharm Setup on Raspberry Pi

> **File:** `ide/pycharm_setup.md`
> **Brief:** Installing and configuring JetBrains PyCharm on Raspberry Pi 4/5 for developing the Autonomous Vehicle Control System, including remote interpreters and ROS 2 debugging.
> **Author:** Autonomous Vehicle Team
> **Date:** 2025-01-30
> **License:** MIT

## 1. When to choose PyCharm

PyCharm offers best-in-class refactoring, type-aware navigation, and an
excellent remote debugger. It is heavier than VS Code — use it on:

* Pi 5 with 8 GB RAM (Professional Edition runs natively).
* Any workstation (x86_64) connecting to the Pi via SSH/Remote Development.

| Edition                | Cost   | Remote debug | ROS 2 plugin | Recommended |
|------------------------|--------|--------------|--------------|-------------|
| Community (local)      | Free   | No           | Limited      | Pi 4 / 5 4 GB |
| Professional (local)   | Paid   | Yes          | Yes          | Pi 5 8 GB   |
| Professional (Gateway) | Paid   | Yes (thin)   | Yes          | Workstation → Pi |

---

## 2. Install PyCharm on the Pi (native)

### 2.1 Via JetBrains Toolbox (recommended)

```bash
# 1. Install prerequisites
sudo apt install -y libxext6 libxrender1 libxtst6 libxi6 libfreetype6

# 2. Download Toolbox for ARM64
wget -O /tmp/toolbox.tar.gz \
  https://download.jetbrains.com/toolbox/jetbrains-toolbox-2.2.3.20089-arm64.tar.gz
mkdir -p ~/.local/share/JetBrains/Toolbox
tar -xzf /tmp/toolbox.tar.gz -C ~/.local/share/JetBrains/Toolbox --strip-components=1
~/.local/share/JetBrains/Toolbox/jetbrains-toolbox
```

The Toolbox app will appear in your system tray. Click **PyCharm → Install**.
Toolbox handles updates automatically.

### 2.2 Manual install

```bash
cd ~/Downloads
wget https://download.jetbrains.com/python/pycharm-professional-2024.1-aarch64.tar.gz
sudo mkdir -p /opt/pycharm
sudo tar -xzf pycharm-professional-2024.1-aarch64.tar.gz -C /opt/pycharm --strip-components=1
sudo ln -s /opt/pycharm/bin/pycharm.sh /usr/local/bin/pycharm
pycharm &
```

### 2.3 Desktop entry

```bash
cat <<EOF | sudo tee /usr/share/applications/pycharm.desktop
[Desktop Entry]
Version=1.0
Type=Application
Name=PyCharm
Icon=/opt/pycharm/bin/pycharm.png
Exec="/opt/pycharm/bin/pycharm.sh" %f
Comment=Python IDE
Categories=Development;IDE;
Terminal=false
StartupWMClass=jetbrains-pycharm
EOF
```

---

## 3. First-launch configuration

1. **Accept license.**
2. **UI theme:** Dark (recommended — reduces Pi display brightness).
3. **Plugins to enable:**
   * `IdeaVIM` (optional)
   * `Docker`
   * `ROS Support` (Professional only)
   * `Markdown`
   * `.ignore`
4. **Disable plugins you don't need** to save RAM:
   * `Android Development`
   * `Jupyter Notebook` (unless used)
   * `Database Tools & SQL` (unless used)

---

## 4. Open the project

```bash
git clone https://github.com/your-org/autonomous_vehicle_hardware.git
cd autonomous_vehicle_hardware/raspberry_pi
pycharm .
```

Or via the welcome screen → **Open** → navigate to the folder.

---

## 5. Configure Python interpreter

### 5.1 Local venv on the Pi

```bash
python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install -r requirements.txt
```

In PyCharm:

1. **File → Settings → Project → Python Interpreter**
2. Click **Add Interpreter → Add Local Interpreter → Virtualenv Environment**
3. Select **Existing** → `~/autonomous_vehicle_hardware/raspberry_pi/venv/bin/python`
4. Apply → OK.

### 5.2 Add ROS 2 Python paths

In **Settings → Project → Python Interpreter → Show paths for the selected
interpreter → +**, add:

```
/opt/ros/humble/lib/python3.10/site-packages
/opt/ros/humble/local/lib/python3.10/dist-packages
~/ros2_ws/install/av_perception/lib/python3.10/site-packages
~/ros2_ws/install/av_control/lib/python3.10/site-packages
~/ros2_ws/install/av_localization/lib/python3.10/site-packages
~/ros2_ws/install/av_navigation/lib/python3.10/site-packages
```

### 5.3 Remote interpreter (Professional)

If your workstation runs PyCharm and the Pi runs the code:

1. **File → Settings → Project → Python Interpreter → Add → SSH Interpreter**
2. Host: `autovehicle-pi.local`, User: `ubuntu`, port `22`, key file
   `~/.ssh/id_ed25519`.
3. Interpreter: `/home/ubuntu/av_ws/venv/bin/python`.
4. Sync folders: local `python/` ↔ remote `/home/ubuntu/av_ws/python/`.
5. Save → PyCharm uploads the project on every save.

---

## 6. Configure deployment (RSync)

**Settings → Build, Execution, Deployment → Deployment → +**

* **Type:** SFTP
* **SFTP host:** `autovehicle-pi.local`
* **Port:** 22
* **Root path:** `/home/ubuntu/av_ws`
* **User name:** `ubuntu`
* **Auth type:** Key pair
* **Mappings:** Local `…/raspberry_pi/` → Deployment `/av_ws/`

Enable **Tools → Deployment → Automatic Upload (always)**.

---

## 7. Configure run/debug configurations

### 7.1 Run `main.py`

1. **Run → Edit Configurations → + → Python**
2. Script path: `$ProjectFileDir$/python/main.py`
3. Parameters: `--rate 30 --config config/av-01.yaml`
4. Working directory: `$ProjectFileDir$`
5. Environment variables:
   ```
   PYTHONPATH=$ProjectFileDir$/python:$ProjectFileDir$/ros2
   ROS_DOMAIN_ID=42
   ```
6. Python interpreter: your venv.

### 7.2 Debug a ROS 2 node

1. **Run → Edit Configurations → + → Python**
2. Module name: `rclpy` (or use script `ros2_ws/install/.../perception_node.py`).
3. Parameters: `--ros-args -p vehicle_id:=av-01`
4. Before launch:
   * Add **External tool** → `source /opt/ros/humble/setup.bash`
     (configure an External Tool that runs `bash -lc`).
5. Set breakpoints in the node.
6. Click the **Debug** icon (🐛).

### 7.3 Attach to running process

1. **Run → Attach to Process** → pick `python3 perception_node.py`.
2. PyCharm injects the debug agent via `pydevd-pycharm`.

For remote attach:

```python
# add to your node entrypoint
import pydevd_pycharm
pydevd_pycharm.settrace('host.docker.internal', port=12345,
                        stdoutToServer=True, stderrToServer=True,
                        suspend=False)
```

Then **Run → Edit Configurations → + → Python Debug Server**, port `12345`,
start it, and run the node on the Pi.

---

## 8. ROS 2 plugin specifics

The ROS plugin (Professional only) provides:

* Syntax highlighting for `.launch.py` / `.urdf` / `.msg`
* Jump-to-message-definition
* Topic list view (`ROS → Topics`)
* Service list view (`ROS → Services`)
* RViz-like 2D plot viewer (limited)

**Note:** On 32-bit `rclpy` installs the plugin's indexer may segfault — use
the 64-bit Raspberry Pi OS / Ubuntu Server 22.04 only.

---

## 9. Code style & inspection

1. **Settings → Editor → Code Style → Python** → set hard wrap at `100`.
2. **Settings → Tools → Black** → enable "On save".
3. **Settings → Editor → Inspections → Python**:
   * Enable `PEP 8 coding style violation`.
   * Enable `Type checker`.
   * Disable `PyPep8Naming` if you use ROS naming conventions.

---

## 10. Memory tuning on Pi 4

Edit **Help → Edit Custom VM Options…** and paste:

```
-Xms256m
-Xmx1024m
-XX:ReservedCodeCacheSize=240m
-XX:+UseG1GC
-XX:SoftRefLRUPolicyMSPerMB=50
-Dsun.io.useCanonCaches=false
-Djava.net.preferIPv4Stack=true
-Djdk.attach.allowAttachSelf=true
-Dkotlin.daemon.jvm.options="-Xmx256m"
```

Restart PyCharm. This keeps memory under 1.4 GB on a 4 GB Pi.

---

## 11. Common issues

| Symptom                                  | Fix                                                |
|------------------------------------------|----------------------------------------------------|
| Slow typing / laggy UI                   | Disable VIM plugin; reduce inspections             |
| `ImportError: rclpy` not found           | Add ROS 2 paths to interpreter (section 5.2)       |
| "Cannot connect to display" over SSH     | `ssh -X` or use X11 forwarding with `xauth`        |
| Remote sync stuck                        | Increase `Settings → Deployment → Advanced → Timeout` to 60 s |
| Run config can't find module             | Set working dir explicitly; check `PYTHONPATH`     |
| `Java heap space` on indexing            | Increase `-Xmx` (section 10) or exclude `build/`, `install/` |
| PyCharm does not show in menu            | Re-run `jetbrains-toolbox` once                    |

---

## 12. Useful shortcuts

| Shortcut                 | Action                          |
|--------------------------|---------------------------------|
| `Shift Shift`            | Search everywhere               |
| `Ctrl Shift F`           | Find in project                 |
| `Ctrl Shift R`           | Replace in project              |
| `F9`                     | Resume debug                    |
| `F8`                     | Step over                       |
| `F7`                     | Step into                       |
| `Alt F8`                 | Evaluate expression             |
| `Ctrl Alt L`             | Reformat code                   |
| `Ctrl /`                 | Toggle comment                  |
