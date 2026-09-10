# Fish YOLO 捕鱼图片识别辅助工具
> ⚠️ 仅用于计算机视觉学习，静态图片上传识别鱼类，无自动点击、自动游戏控制，禁止用作游戏外挂

## ✨功能
✅ 本地上传捕鱼截图
✅ YOLOv8识别图片内鱼种，自动画检测框保存图片
✅ OCR识别截图金币数字
✅ 自动计算打鱼策略、BOSS倒计时、盈亏统计

## 📦环境安装（电脑运行）
1. 安装 Python3.10 + Git + Tesseract-OCR
2. 拉取项目
```bash
git clone https://github.com/CHCH2828/fish-yolo-assistant.git
cd fish-yolo-assistant
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
