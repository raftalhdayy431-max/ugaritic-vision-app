[app]

# (str) Title of your application
title = Ugaritic Vision AI
version = 0.1
# (str) Package name
package.name = ugariticvision

# (str) Package domain (needed for android packaging)
package.domain = org.ugaritic

# (str) Source files where the *.py files live
source.dir = .

# (list) Source files to include
source.include_exts = py,png,jpg,kv,ttf,tflite

# (list) List of inclusions
source.include_patterns = fonts/*.ttf,*.tflite

# (list) Application requirements
# استبدل السطر القديم بهذا السطر الصحيح هندسياً:
requirements = python3,kivy,kivymd,pillow,numpy,opencv,tflite-runtime
# (str) Supported orientations
orientation = portrait

# (list) Permissions
android.permissions = CAMERA,WRITE_EXTERNAL_STORAGE,READ_EXTERNAL_STORAGE

# (int) Target Android API (تم رفعه إلى 34 ليتوافق مع أحدث متطلبات متجر جوجل بلاي والإصدارات الحديثة)
android.api = 33

# (int) Minimum API your APK will support (تم ضبطه على 21 ليدعم الهواتف القديمة بدءاً من Android 5.0 فصاعداً)
android.minapi = 21

# (int) Android NDK version to use (الإصدار الأقر والأكثر استقراراً مع بايثون وتنسرفلو)
android.ndk = 25b

# (str) Android SDK version to use
android.sdk = 33
android.build_tools_version = 33.0.2
# (bool) Indicate whether the application should be fullscreen or not
fullscreen = 0

# (str) Supported architectures (دعم المعمارية القديمة والحديثة 32-bit و 64-bit ليعمل التطبيق على كافة الأجهزة بلا استثناء)
android.archs = arm64-v8a, armeabi-v7a

[buildozer]

# (int) Log level (0 = error, 1 = info, 2 = debug)
log_level = 2

# (int) Display warning if buildozer is run as root
warn_root = 1
