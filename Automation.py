import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# 定義生成圖像的函數
def generate_images():
    # 這裡添加生成圖像程式碼
    print("Generating images...")
    # Example function call
    # gan.generate()  # 確保函數來生成圖像

# 自動提交並推送到 GitHub
def generate_and_upload():
    while True:
        generate_images()  # 生成圖像
        os.system('git add .')
        os.system('git commit -m "Add new"')
        os.system('git pull --rebase --autostash')# 使用 rebase 和自動存儲
        os.system('git push')
        time.sleep(3600)  # 每小時執行一次

class MyHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.is_directory:
            return None
        else:
            os.system('git add .')
            os.system('git commit -m "Auto commit new"')
            os.system('git pull --rebase --autostash')# 使用 rebase 和自動存儲
            os.system('git push')

if __name__ == "__main__":
    path = "C:/Users/user/Desktop/GAN 20241117/result"  # 修改為正確的路徑格式
    event_handler = MyHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    
    try:
        generate_and_upload()  # 開始生成圖像並上傳
    except KeyboardInterrupt:
        observer.stop()
    observer.join()
