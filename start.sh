#!/bin/bash

# 1. Kikerüljük a rendszer Qt fájljait, és kizárólag a .venv saját fájljait használjuk
export LD_LIBRARY_PATH=$(find .venv -type d -path "*/PyQt5/Qt5/lib" | head -n 1):$LD_LIBRARY_PATH

# 2. XCB platform kényszerítése, ha a GNOME Wayland-en fut
export QT_QPA_PLATFORM=xcb

# 3. A program indítása
.venv/bin/python prism_gui.py
