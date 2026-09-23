# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['labelme/__main__.py'],
    pathex=['D:/Code/labelme"     --collect-submodules labelme     --specpath=build     --add-data D:/Code/labelme/.pixi/envs/default/lib/site-packages/osam/_models/yoloworld/clip/bpe_simple_vocab_16e6.txt.gz', 'osam/_models/yoloworld/clip     --add-data D:/Code/labelme/labelme/_config/default_config.yaml', 'labelme/_config     --add-data D:/Code/labelme/labelme/icons/*', 'labelme/icons     --add-data D:/Code/labelme/labelme/translate/*', 'translate     --icon D:/Code/labelme/labelme/icons/icon-256.png     --onedir'],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='Labelme',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='Labelme',
)
