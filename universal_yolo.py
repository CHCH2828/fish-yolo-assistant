# 跨平台通用 YOLO识别代码：Windows / Mac / Android / iPhone 全部支持
from ultralytics import YOLO
import cv2
import numpy as np
import easyocr
import os
from pathlib import Path

# ====================== 【全局配置，所有设备统一】 ======================
# 模型路径，pathlib自动适配不同操作系统斜杠，无需手动改斜杠
MODEL_RELATIVE_PATH = Path("fish_train_result") / "fish_yolov8_model" / "weights" / "best.pt"
RESULT_IMAGE_NAME = "detect_result.jpg"
# OCR语言：简体中文 + 英文
OCR_LANG = ["ch_sim", "en"]
# ======================================================================

def get_absolute_path(relative_path: Path) -> Path:
    """自动获取绝对路径，适配Windows/Mac/安卓/苹果手机"""
    base_dir = Path(__file__).resolve().parent
    return base_dir / relative_path

def detect_image(image_input_path: str):
    # 转为跨平台路径对象
    img_path = Path(image_input_path)
    model_full_path = get_absolute_path(MODEL_RELATIVE_PATH)

    # 校验模型文件是否存在
    if not model_full_path.exists():
        print(f"\n❌ 错误：没有找到模型 best.pt")
        print(f"👉 预期模型完整路径：{model_full_path}")
        print("👉 请检查文件夹层级：fish_train_result/fish_yolov8_model/weights/best.pt")
        return

    # 校验图片是否存在
    if not img_path.exists():
        print(f"\n❌ 错误：图片文件不存在 -> {img_path}")
        return

    print(f"\n✅ 正在加载YOLO模型：{model_full_path}")
    model = YOLO(str(model_full_path))
    print("✅ YOLO模型加载完成！")

    # 加载图片
    img = cv2.imread(str(img_path))
    if img is None:
        print("❌ 读取图片失败，图片损坏或者格式不支持！")
        return

    # YOLO目标检测推理
    print("🔍 开始识别图片...")
    results = model(img)
    result_draw_img = results[0].plot()

    # 保存带标注的结果图片
    save_full_path = get_absolute_path(RESULT_IMAGE_NAME)
    cv2.imwrite(str(save_full_path), result_draw_img)
    print(f"✅ 检测图片已保存到：{save_full_path}")
    print(f"👉 检测目标总数量：{len(results[0].boxes)}")

    # EasyOCR文字识别（全平台通用，不需要安装tesseract）
    print("\n🔤 开始OCR文字识别...")
    ocr_reader = easyocr.Reader(OCR_LANG, verbose=False)
    ocr_result = ocr_reader.readtext(str(img_path))

    print("==== OCR识别文字结果 ====")
    if len(ocr_result) == 0:
        print("没有识别到任何文字")
    else:
        for box, text, score in ocr_result:
            print(f"文字：{text}，置信度：{score:.2f}")
    print("\n=========================")


if __name__ == "__main__":
    print("==== 🐟 跨平台通用 YOLO鱼类截图识别程序 ====")
    print("✅ 支持设备：Windows电脑 / Mac电脑 / 安卓手机 / iPhone苹果手机")
    print("👉 输入图片的完整路径，例如：")
    print("    Windows示例：C:\\test\\screenshot.jpg")
    print("    Mac示例：/Users/xxx/Desktop/test.jpg")
    print("    安卓示例：/storage/emulated/0/Download/pic.jpg")
    print("    iPhone示例：/private/var/mobile/Containers/xxx/test.jpg")
    print("============================================")
    img_input = input("请输入图片完整路径：")
    detect_image(img_input.strip())
