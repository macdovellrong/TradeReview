@echo off
:: ==============================
:: 1. 解决乱码问题
:: ==============================
:: 将代码页切换为 UTF-8 (65001)，这样中文就不会乱码了
chcp 65001 >nul

title Codex Start Script
echo.

:: ==============================
:: 2. 配置参数 (根据你的日志已填好)
:: ==============================
set PROXY_HOST=10.0.0.25
set PROXY_PORT=1082

:: 这里填你要运行的程序名称
set RUN_COMMAND=codex

:: ==============================
:: 3. 设置代理并运行
:: ==============================

echo [INFO] 正在配置代理环境...
echo [INFO] 目标服务器: %PROXY_HOST%:%PROXY_PORT%

:: 设置环境变量
set HTTP_PROXY=http://%PROXY_HOST%:%PROXY_PORT%
set HTTPS_PROXY=http://%PROXY_HOST%:%PROXY_PORT%
set ALL_PROXY=http://%PROXY_HOST%:%PROXY_PORT%

echo.
echo ---------------------------------------------------
echo [INFO] 环境变量配置完毕，准备启动 %RUN_COMMAND%
echo ---------------------------------------------------
echo.

:: 使用 call 命令运行，防止某些脚本运行完直接关闭窗口
call %RUN_COMMAND%

:: ==============================
:: 4. 错误排查 (如果 gemini 没启动)
:: ==============================
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] 运行出错！
    echo 请检查：
    echo 1. 你的电脑上是否已经安装了 gemini-cli？
    echo 2. 在 cmd 里直接输入 gemini 能不能运行？
)

echo.
echo [INFO] 程序已结束，按任意键关闭窗口...
pause