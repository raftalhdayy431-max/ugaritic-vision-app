[app]

# (str) Title of your application

title = Ugaritic Vision AI

# (str) Application version

version = 0.1

# (str) Package name

package.name = ugariticvision

# (str) Package domain

package.domain = org.ugaritic

# ============================================================

# SOURCE

# ============================================================

# (str) Source files directory

source.dir = .

# (list) Source extensions to include

source.include_exts = py,png,jpg,kv,ttf,tflite

# (list) Additional source patterns

source.include_patterns = fonts/*.ttf,*.tflite

# ============================================================

# PYTHON REQUIREMENTS

# ============================================================

requirements = python3,kivy,kivymd,pillow,numpy,opencv,tflite-runtime

# ============================================================

# DISPLAY

# ============================================================

# (str) Supported orientation

orientation = portrait

# (bool) Fullscreen mode

fullscreen = 0

# ============================================================

# ANDROID PERMISSIONS

# ============================================================

android.permissions = CAMERA,WRITE_EXTERNAL_STORAGE,READ_EXTERNAL_STORAGE

# ============================================================

# ANDROID SDK CONFIGURATION

# ============================================================

# Target Android API

android.api = 33

# Minimum Android API

android.minapi = 24
android.ndk_api = 24
# Android SDK Platform

android.sdk = 33

# Force specific Build Tools version

android.build_tools_version = 33.0.2

# ============================================================

# ANDROID NDK

# ============================================================

# Stable NDK version

android.ndk = 25b

# ============================================================

# ANDROID ARCHITECTURES

# ============================================================

# ARM 64-bit and ARM 32-bit support

android.archs = arm64-v8a, armeabi-v7a

# ============================================================

# GITHUB ACTIONS SDK OVERRIDES

# ============================================================

# Force Buildozer to use the pre-installed SDK

android.sdk_path = /home/runner/.buildozer/android/platform/android-sdk

# Prevent Buildozer from automatically updating SDK components

android.skip_update = True

# SDK licenses are accepted automatically in CI

android.accept_sdk_license = True

# ============================================================

# BUILDOZER

# ============================================================

[buildozer]

# Log level

log_level = 2

# Display warning if buildozer is run as root

warn_root = 1
