install dataset di https://www.kaggle.com/datasets/juanthomaswijaya/indonesian
license-plate-dataset		

python mainuas.py #untuk membuat ground_thruth

# menjalankan lmstudio
lms server start       
lms load Qwen2-VL-2B-Instruct-GGUF   

python UAS.py # menjalankan OCR    
