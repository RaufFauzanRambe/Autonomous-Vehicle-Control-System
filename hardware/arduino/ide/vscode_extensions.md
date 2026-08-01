# Recommended VS Code Extensions

The AVCS firmware is best developed in **Visual Studio Code** with the
following extensions installed.  Each extension below has been chosen to
improve productivity, code quality, or debugging for embedded C++ work.

---

## 1. PlatformIO IDE

| Field        | Value                                    |
|--------------|------------------------------------------|
| Publisher    | PlatformIO                               |
| Marketplace ID | `platformio.platformio-ide`            |
| Install      | `code --install-extension platformio.platformio-ide` |

**Why you need it:** Builds, uploads, monitors, and debugs embedded firmware
from a single IDE.  Manages per-project dependencies without polluting your
global Arduino libraries folder.

**Key features**

- One-click build / upload / serial-monitor.
- Per-environment IntelliSense (the same code can be analysed for both AVR
  and ARM targets).
- Library dependency manager (LDF) with deep mode.
- Native unit-testing framework.
- Integrates with JTAG/SWD debuggers (e.g. Atmel-ICE).

**After install**: open the PIO Home tab and install `Atmel AVR` and
`Atmel SAM` platforms.

---

## 2. Arduino (Community Edition)

| Field        | Value                                    |
|--------------|------------------------------------------|
| Publisher    | Microsoft (originally) / Microsoft vsciot-vscode |
| Marketplace ID | `vsciot-vscode.vscode-arduino`         |
| Install      | `code --install-extension vsciot-vscode.vscode-arduino` |

**Why you need it:** Lets you build & upload `.ino` sketches directly from
VS Code without the Arduino IDE GUI — useful for quick one-off examples like
`examples/blink_example.ino`.

**Key features**

- Select board & port from the status bar.
- Verifies and uploads using the Arduino CLI.
- Snippet library for common Arduino idioms.
- Code-completion for the Arduino API.

> **Note**: if you have PlatformIO installed, this extension is *optional*.
> Both can coexist; PlatformIO is generally the better choice for AVCS.

---

## 3. C/C++ (Microsoft)

| Field        | Value                                    |
|--------------|------------------------------------------|
| Publisher    | Microsoft                                |
| Marketplace ID | `ms-vscode.cpptools`                    |
| Install      | `code --install-extension ms-vscode.cpptools` |

**Why you need it:** The de-facto IntelliSense, formatting, and debugging
engine for C/C++ in VS Code.

**Key features**

- Go-to-definition, find-all-references, rename-symbol across the project.
- Debugging with GDB (for SAM targets) and simavr (AVR simulation).
- clang-format integration — set `"C_Cpp.formatting": "clangFormat"`.
- Configurable include paths via `c_cpp_properties.json`.

**Recommended settings** (add to `.vscode/settings.json`):

```json
{
  "C_Cpp.default.cStandard":   "c11",
  "C_Cpp.default.cppStandard": "c++17",
  "C_Cpp.default.intelliSenseMode": "gcc-arm",
  "C_Cpp.formatting": "clangFormat",
  "C_Cpp.codeAnalysis.clangTidy.enabled": true,
  "C_Cpp.codeAnalysis.clangTidy.checks": [
    "bugprone-*", "modernize-*", "performance-*", "readability-*"
  ]
}
```

---

## 4. CMake Tools

| Field        | Value                                    |
|--------------|------------------------------------------|
| Publisher    | Microsoft                                |
| Marketplace ID | `ms-vscode.cmake-tools`                 |
| Install      | `code --install-extension ms-vscode.cmake-tools` |

**Why you need it:** The `native` PlatformIO environment (used for PC-side
unit tests) builds with a CMake-like flow; this extension provides
configuration selection, build, and test targets in a side panel.

**Key features**

- Configure / build / install from the status bar.
- Test explorer integration for `ctest`.
- Multi-variant build presets.
- Debug-launch integration for executables built by CMake.

---

## 5. Code Spell Checker

| Field        | Value                                    |
|--------------|------------------------------------------|
| Publisher    | Street Side Software                     |
| Marketplace ID | `streetsidesoftware.code-spell-checker` |
| Install      | `code --install-extension streetsidesoftware.code-spell-checker` |

**Why you need it:** Catch typos in comments, strings, identifiers, and
documentation — important for code that ends up in Doxygen-generated docs.

**Key features**

- Multi-language dictionaries (en-US, en-GB, …).
- Per-workspace ignore lists via `cspell.json`.
- Inline squiggles and quick-fix suggestions.
- Integrates with platformio.ini / Doxygen comments.

**Project-level config** — create `cspell.json` in the project root:

```json
{
  "version": "0.2",
  "language": "en",
  "words": [
    "Arduino", "PlatformIO", "BNO", "TinyGPSPlus", "Ackermann",
    "Haversine", "NMEA", "PWM", "I2C", "SPI", "UART", "EEPROM",
    "AVR", "SAMD", "ATmega", "ATSAM"
  ],
  "ignorePaths": [".pio/**", "node_modules/**", "*.hex"]
}
```

---

## 6. GitLens — Git Supercharged

| Field        | Value                                    |
|--------------|------------------------------------------|
| Publisher    | GitKraken                                |
| Marketplace ID | `eamodio.gitlens`                       |
| Install      | `code --install-extension eamodio.gitlens` |

**Why you need it:** Inline blame annotations, file-history explorer, and a
powerful commit-graph view.  Essential for understanding *why* a particular
PID gain or pin macro was changed.

**Key features**

- Hover-blame on every line shows the last commit that touched it.
- Side-bar view: commits, branches, stashes, remotes, contributors.
- Compare any two commits in a diff editor.
- One-click "Open on GitHub/GitLab".

**Recommended settings**:

```json
{
  "gitblame.inline_message.enabled": true,
  "gitblame.statusBar.messageFormat": "${author}, ${commit.summary}"
}
```

---

## 7. Bonus Extensions (Optional but Worth It)

### Clang-Format

`xaver.clang-format` — applies `.clang-format` on save. Use Google style
with `IndentWidth: 4` and `ColumnLimit: 100`.

### Doxygen Documentation Generator

`cschlosser.doxdocgen` — type `/**` + Enter above a function and it
auto-generates `@brief @param @return` skeletons.

### Better Comments

`aaron-bond.better-comments` — colour-codes `!`, `?`, `// TODO:`, `*`
comments for at-a-glance scanning.

### Hex Editor

`ms-vscode.hexeditor` — inspect compiled `.hex` / `.bin` files directly in
VS Code.

### Serial Monitor

`ms-vscode.vscode-serial-monitor` — Microsoft's modern serial monitor
(replaces the PIO one for non-PIO projects).

---

## 8. Quick Install (One-Liner)

```bash
code --install-extension platformio.platformio-ide \
     --install-extension vsciot-vscode.vscode-arduino \
     --install-extension ms-vscode.cpptools \
     --install-extension ms-vscode.cmake-tools \
     --install-extension streetsidesoftware.code-spell-checker \
     --install-extension eamodio.gitlens \
     --install-extension cschlosser.doxdocgen \
     --install-extension xaver.clang-format
```

---

## 9. Recommended `settings.json` for AVCS

```json
{
  "editor.formatOnSave": true,
  "editor.rulers": [100],
  "files.trimTrailingWhitespace": true,
  "files.insertFinalNewline": true,
  "[cpp]":  { "editor.defaultFormatter": "xaver.clang-format" },
  "[c]":    { "editor.defaultFormatter": "xaver.clang-format" },
  "[ini]":  { "editor.defaultFormatter": "redhat.vscode-yaml" },
  "C_Cpp.default.cppStandard": "c++17",
  "C_Cpp.default.intelliSenseMode": "gcc-arm",
  "platformio-ide.useBuiltinPython": true,
  "cSpell.enabled": true
}
```

---

## 10. Summary Table

| # | Extension                       | Purpose                            |
|---|---------------------------------|------------------------------------|
| 1 | PlatformIO IDE                  | Build / upload / monitor firmware  |
| 2 | Arduino                          | Quick `.ino` sketch editing        |
| 3 | C/C++ (Microsoft)                | IntelliSense & debugging           |
| 4 | CMake Tools                      | Native unit-test builds            |
| 5 | Code Spell Checker               | Catch comment typos                |
| 6 | GitLens                          | Git blame / history                |
| 7 | Doxygen Documentation Generator  | Auto-generate docstring skeletons  |
| 8 | Clang-Format                     | Consistent code style              |
| 9 | Hex Editor                       | Inspect compiled binaries          |
| 10| Serial Monitor (MS)              | Stand-alone serial monitor         |
