# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for QuantAI.

打包命令：pyinstaller quantai.spec --noconfirm
说明：weasyprint 作为可选依赖未列入，PDF 导出将使用 reportlab 降级方案。
"""
import sys
from pathlib import Path

block_cipher = None
root = Path(SPECPATH).resolve()

a = Analysis(
    ['run.py'],
    pathex=[str(root / 'src')],
    binaries=[],
    datas=[
        (str(root / 'config'), 'config'),
    ],
    hiddenimports=[
        # Streamlit 运行时需要
        'streamlit.web.cli',
        'streamlit.runtime.scriptrunner',
        'streamlit.web.server.server',
        # 项目包（显式列出避免动态导入遗漏）
        'quantai',
        'quantai.config',
        'quantai.paths',
        'quantai.logger',
        'quantai.data',
        'quantai.data.fetcher',
        'quantai.data.cache',
        'quantai.data.indicators',
        'quantai.ml',
        'quantai.ml.features',
        'quantai.ml.models',
        'quantai.ml.predictor',
        'quantai.ai',
        'quantai.ai.client',
        'quantai.ai.prompts',
        'quantai.report',
        'quantai.report.chart',
        'quantai.report.generator',
        'quantai.report.batch',
        'quantai.report.compare',
        'quantai.report.export',
        'quantai.web',
        'quantai.web.app',
        # 数据源
        'baostock',
        'akshare',
        # ML
        'sklearn',
        'xgboost',
        'lightgbm',
        # 可视化
        'plotly',
        # PDF 导出降级方案
        'reportlab',
        'reportlab.pdfbase',
        'reportlab.pdfbase.ttfonts',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'unittest',
        'pydoc',
        'doctest',
        'IPython',
        'jupyter',
        'notebook',
        'matplotlib',
        'weasyprint',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='QuantAI',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
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
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='QuantAI',
)
