# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files
from PyInstaller.utils.hooks import collect_dynamic_libs
from PyInstaller.utils.hooks import collect_submodules
from PyInstaller.utils.hooks import copy_metadata

datas = []
binaries = []
hiddenimports = ['accelerator', 'hmog_consecutive_rejects', 'hmog_data', 'hmog_metrics', 'hmog_token_auth_inference', 'hmog_token_transformer', 'hmog_tokenizer', 'hmog_vqgan_experiment', 'hmog_vqgan_token_transformer_experiment', 'platformdirs', 'runtime_paths', 'vqgan']
datas += collect_data_files('torch_npu')
datas += copy_metadata('torch')
datas += copy_metadata('pandas')
datas += copy_metadata('scikit-learn')
binaries += collect_dynamic_libs('torch_npu')
hiddenimports += collect_submodules('src')
hiddenimports += collect_submodules('grpc_health')
hiddenimports += collect_submodules('torch_npu')


a = Analysis(
    ['src/cli.py'],
    pathex=['/data/code/backup/ca-server', '/data/code/backup/ca-server/ca_train'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
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
    name='ca-server',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
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
    name='ca-server',
)
