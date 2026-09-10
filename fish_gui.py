import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO

# ========= 这里修改为你电脑上 tesseract 的路径，电脑安装完再修改 =========
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class FishDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Fish YOLO 识别工具")
        self.root.geometry("600x400")

        self.model_path = r"fish_train_result/fish_yolov8_model/weights/best.pt"
        self.model = None

        tk.Label(root, text="YOLO鱼类截图识别程序", font=("SimHei",16)).pack(pady=10)
        tk.Button(root, text="加载YOLO模型", command=self.load_model, width=20).pack(pady=5)
        tk.Button(root, text="选择截图识别", command=self.select_image, width=20).pack(pady=5)
        self.info_text = tk.Text(root, height=12)
        self.info_text.pack(padx=10,pady=10)

    def log(self, msg):
        self.info_text.insert(tk.END, msg + "\n")
        self.info_text.see(tk.END)
        self.root.update_idletasks()

    def load_model(self):
        try:
            self.log(f"正在加载模型：{self.model_path}")
            self.model = YOLO(self.model_path)
            self.log("✅ YOLO模型加载成功！")
        except Exception as e:
            self.log(f"❌ 模型加载失败：{str(e)}")
            messagebox.showerror("错误", f"模型加载失败：{e}\n检查best.pt路径是否正确")

    def select_image(self):
        if self.model is None:
            messagebox.showwarning("警告","请先加载YOLO模型！")
            return
        file_path = filedialog.askopenfilename(filetypes=[("图片文件","*.jpg;*.png;*.jpeg")])
        if not file_path:
            return
        self.detect_image(file_path)

    def detect_image(self, img_path):
        try:
            img = cv2.imread(img_path)
            results = self.model(img)
            res_img = results[0].plot()
            cv2.imwrite("result_output.jpg", res_img)
            # OCR识别文字
            ocr_text = pytesseract.image_to_string(img)
            self.log(f"====识别图片：{img_path}====")
            self.log(f"检测到目标数量：{len(results[0].boxes)}")
            self.log(f"OCR识别文字：{ocr_text.strip()}")
            self.log("✅ 识别完成，结果保存 result_output.jpg\n")
            cv2.imshow("识别结果", res_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            self.log(f"❌识别出错：{str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = FishDetectorGUI(root)
    root.mainloop()
