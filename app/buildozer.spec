[app]

# ============================================================
# Application
# ============================================================

title = Ugaritic OCR Studio Pro

package.name = ugariticocr

package.domain = org.ugaritic

source.dir = .

version = 1.0.0


# ============================================================
# Source files
# ============================================================

source.include_exts = py,png,jpg,jpeg,kv,atlas,ttf,tflite,json,txt,xml

source.include_patterns = fonts/*.ttf,*.tflite

source.exclude_exts = pyc,pyo

source.exclude_dirs = .git,.github,__pycache__,bin,tests,.venv,venv


# ============================================================
# Python / Kivy requirements
# ============================================================

requirements = python3,kivy==2.3.0,kivymd==1.2.0,pillow,numpy,plyer,opencv,tflite-runtime


# ============================================================
# Application UI
# ============================================================

orientation = portrait

fullscreen = 0


# ============================================================
# Android permissions
# ============================================================

android.permissions = CAMERA,READ_MEDIA_IMAGES


# ============================================================
# Android SDK / NDK
# ============================================================

# Target Android API
android.api = 35

# Minimum Android API
android.minapi = 21

# Recommended NDK
android.ndk = 28c

# NDK API should normally match minapi
android.ndk_api = 21


# ============================================================
# Architecture
# ============================================================

android.archs = arm64-v8a


# ============================================================
# Android storage / packaging
# ============================================================

android.private_storage = True

android.copy_libs = 1

android.accept_sdk_license = True


# ============================================================
# Python-for-Android
# ============================================================

p4a.branch = master


# ============================================================
# Android logging
# ============================================================

android.logcat_filters = *:S python:D


# ============================================================
# Android backup
# ============================================================

android.allow_backup = True


# ============================================================
# Buildozer
# ============================================================

[buildozer]

log_level = 2

warn_root = 1
