[app]

title = Ugaritic OCR Studio Pro
package.name = ugariticocr
package.domain = org.ugaritic

source.dir = .
source.include_exts = py,png,jpg,jpeg,kv,atlas,ttf,tflite,json,txt,xml
source.exclude_exts = pyc,pyo
source.exclude_dirs = .git,.github,__pycache__,bin,tests,.venv,venv

version = 1.0.0

requirements = python3,kivy==2.3.0,kivymd==1.2.0,pillow,numpy,plyer,opencv,tflite-runtime

orientation = portrait
fullscreen = 0

# Android
android.api = 35
android.minapi = 21
android.ndk_api = 21
android.archs = arm64-v8a

android.accept_sdk_license = True
android.private_storage = True
android.copy_libs = 1

# python-for-android
p4a.branch = master

# Logging
android.logcat_filters = *:S python:D
