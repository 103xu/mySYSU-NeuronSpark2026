@echo off
cd /d "%~dp0"
echo ============================================
echo NS-2026-09 观测之环 - 预测脚本
echo ============================================

REM 设置 Python 路径 (如有需要)
REM set PYTHON=python

echo.
echo [1/3] 检查格式...
python tools/check_format.py results.json --tasks test.jsonl 2>nul
IF %ERRORLEVEL% NEQ 0 (
    echo 格式检查失败，但继续...
)

echo.
echo [2/3] 运行预测...
python main.py --tasks test.jsonl --out results.json

echo.
echo [3/3] 验证输出格式...
python tools/check_format.py results.json --tasks test.jsonl

echo.
echo 完成! 输出文件: results.json
echo 请将 results.json 压缩为 NS-2026-09-answer.zip 后提交
pause
