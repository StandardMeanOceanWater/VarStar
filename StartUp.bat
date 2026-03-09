@echo off
cd /d D:\VarStar\pipeline
call .venv\Scripts\activate
echo (.venv) 已啟動，開啟 Jupyter Notebook...
D:
jupyter notebook
