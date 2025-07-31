import os
import json
import csv
import base64
import requests
from typing import List, Dict, Tuple
from dataclasses import dataclass
from pathlib import Path
import difflib
from PIL import Image
import time

@dataclass
class OCRResult:
    """Class untuk menyimpan hasil OCR"""
    image_path: str
    ground_truth: str
    prediction: str
    cer_score: float

class LMStudioClient:
    """Client untuk berinteraksi dengan LMStudio API"""
    
    def __init__(self, base_url: str = "http://localhost:1234"):
        self.base_url = base_url
        self.api_endpoint = f"{base_url}/v1/chat/completions"
        
    def encode_image_to_base64(self, image_path: str) -> str:
        """Encode gambar ke base64"""
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
                return encoded_string
        except Exception as e:
            print(f"Error encoding image {image_path}: {e}")
            return None
    
    def predict_license_plate(self, image_path: str, model_name: str = "llava") -> str:
        """Prediksi plat nomor menggunakan VLM"""
        try:
            image_base64 = self.encode_image_to_base64(image_path)
            if not image_base64:
                return ""
            
            payload = {
                "model": model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "What is the license plate number shown in this image? Respond only with the plate number without any additional text or explanation."
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 100,
                "temperature": 0.1
            }
            
            response = requests.post(
                self.api_endpoint,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=50000
            )
            
            if response.status_code == 200:
                result = response.json()
                prediction = result['choices'][0]['message']['content'].strip()
                prediction = self.clean_prediction(prediction)
                return prediction
            else:
                print(f"Error calling LMStudio API: {response.status_code}")
                print(f"Response: {response.text}")
                return ""
                
        except Exception as e:
            print(f"Error predicting license plate for {image_path}: {e}")
            return ""
    
    def clean_prediction(self, prediction: str) -> str:
        cleaned = prediction.replace('"', '').replace("'", '').strip()
        import re
        plate_pattern = r'[A-Z]{1,2}\s?\d{1,4}\s?[A-Z]{1,3}'
        match = re.search(plate_pattern, cleaned.upper())
        if match:
            return match.group().replace(' ', '')
        return ''.join(c for c in cleaned.upper() if c.isalnum())

class CERCalculator:
    """Class untuk menghitung Character Error Rate (CER)"""
    
    @staticmethod
    def calculate_cer(ground_truth: str, prediction: str) -> float:
        if not ground_truth:
            return 1.0 if prediction else 0.0
        
        operations = difflib.SequenceMatcher(None, ground_truth, prediction)
        
        substitutions = 0
        deletions = 0
        insertions = 0
        
        for operation, i1, i2, j1, j2 in operations.get_opcodes():
            if operation == 'replace':
                substitutions += max(i2 - i1, j2 - j1)
            elif operation == 'delete':
                deletions += i2 - i1
            elif operation == 'insert':
                insertions += j2 - j1
        
        total_errors = substitutions + deletions + insertions
        cer = total_errors / len(ground_truth) if len(ground_truth) > 0 else 0.0
        return cer

    @staticmethod
    def calculate_detailed_cer(ground_truth: str, prediction: str) -> Dict:
        if not ground_truth:
            return {
                'cer': 1.0 if prediction else 0.0,
                'substitutions': 0,
                'deletions': 0,
                'insertions': len(prediction) if prediction else 0,
                'total_errors': len(prediction) if prediction else 0,
                'ground_truth_length': 0
            }
        
        operations = difflib.SequenceMatcher(None, ground_truth, prediction)
        
        substitutions = 0
        deletions = 0
        insertions = 0
        
        for operation, i1, i2, j1, j2 in operations.get_opcodes():
            if operation == 'replace':
                substitutions += max(i2 - i1, j2 - j1)
            elif operation == 'delete':
                deletions += i2 - i1
            elif operation == 'insert':
                insertions += j2 - j1
        
        total_errors = substitutions + deletions + insertions
        cer = total_errors / len(ground_truth)
        
        return {
            'cer': cer,
            'substitutions': substitutions,
            'deletions': deletions,
            'insertions': insertions,
            'total_errors': total_errors,
            'ground_truth_length': len(ground_truth)
        }

class LicensePlateOCR:
    """Main class untuk OCR plat nomor"""
    
    def __init__(self, lmstudio_url: str = "http://localhost:1234", model_name: str = "llava"):
        self.client = LMStudioClient(lmstudio_url)
        self.model_name = model_name
        self.cer_calculator = CERCalculator()
        self.results: List[OCRResult] = []
    
    def load_dataset(self, dataset_path: str, ground_truth_file: str = None) -> Dict[str, str]:
        ground_truth_dict = {}
        if ground_truth_file and os.path.exists(ground_truth_file):
            try:
                with open(ground_truth_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        ground_truth_dict[row['image']] = row['ground_truth']
            except Exception as e:
                print(f"Error reading ground truth file: {e}")
        return ground_truth_dict
    
    def process_single_image(self, image_path: str, ground_truth: str = "") -> OCRResult:
        print(f"Processing: {image_path}")
        prediction = self.client.predict_license_plate(image_path, self.model_name)
        cer_score = self.cer_calculator.calculate_cer(ground_truth, prediction)
        result = OCRResult(image_path=image_path, ground_truth=ground_truth, prediction=prediction, cer_score=cer_score)
        print(f"Ground Truth: {ground_truth}")
        print(f"Prediction: {prediction}")
        print(f"CER Score: {cer_score:.4f}")
        print("-" * 50)
        return result
    
    def process_dataset(self, dataset_path: str, ground_truth_file: str = None) -> List[OCRResult]:
        if not os.path.exists(dataset_path):
            print(f"Error: Dataset path '{dataset_path}' does not exist!")
            return []

        ground_truth_dict = self.load_dataset(dataset_path, ground_truth_file)
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        image_files = [f for f in Path(dataset_path).iterdir() if f.is_file() and f.suffix.lower() in image_extensions]

        if not image_files:
            print(f"No image files found in '{dataset_path}'")
            return []

        results = []
        for image_path in image_files:
            image_name = image_path.name
            ground_truth = ground_truth_dict.get(image_name, "")
            try:
                result = self.process_single_image(str(image_path), ground_truth)
                results.append(result)
                time.sleep(0.5)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")
                continue

        self.results = results
        return results

    def save_results_to_csv(self, output_file: str = "ocr_results.csv"):
        if not self.results:
            print("No results to save!")
            return
        try:
            with open(output_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['image', 'ground_truth', 'prediction', 'CER_score'])
                for result in self.results:
                    writer.writerow([
                        os.path.basename(result.image_path),
                        result.ground_truth,
                        result.prediction,
                        f"{result.cer_score:.4f}"
                    ])
            print(f"Results saved to {output_file}")
        except Exception as e:
            print(f"Error saving results: {e}")
    
    def calculate_overall_metrics(self) -> Dict:
        if not self.results:
            return {
                'total_images': 0,
                'average_cer': 0.0,
                'accuracy': 0.0,
                'total_substitutions': 0,
                'total_deletions': 0,
                'total_insertions': 0,
                'total_ground_truth_length': 0,
                'correct_predictions': 0
            }

        total_cer = sum(result.cer_score for result in self.results)
        avg_cer = total_cer / len(self.results)
        correct_predictions = sum(1 for result in self.results if result.ground_truth == result.prediction and result.ground_truth != "")
        images_with_gt = sum(1 for result in self.results if result.ground_truth != "")
        accuracy = correct_predictions / images_with_gt if images_with_gt > 0 else 0.0

        total_substitutions = 0
        total_deletions = 0
        total_insertions = 0
        total_ground_truth_length = 0
        
        for result in self.results:
            if result.ground_truth:
                detailed = self.cer_calculator.calculate_detailed_cer(result.ground_truth, result.prediction)
                total_substitutions += detailed['substitutions']
                total_deletions += detailed['deletions']
                total_insertions += detailed['insertions']
                total_ground_truth_length += detailed['ground_truth_length']
        
        return {
            'total_images': len(self.results),
            'average_cer': avg_cer,
            'accuracy': accuracy,
            'total_substitutions': total_substitutions,
            'total_deletions': total_deletions,
            'total_insertions': total_insertions,
            'total_ground_truth_length': total_ground_truth_length,
            'correct_predictions': correct_predictions,
            'images_with_ground_truth': images_with_gt
        }

    def print_summary(self):
        metrics = self.calculate_overall_metrics()
        print("\n" + "="*60)
        print("SUMMARY RESULTS")
        print("="*60)
        print(f"Total Images Processed: {metrics['total_images']}")
        print(f"Images with Ground Truth: {metrics['images_with_ground_truth']}")
        print(f"Average CER: {metrics['average_cer']:.4f}")
        print(f"Accuracy (Exact Match): {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"Correct Predictions: {metrics['correct_predictions']}/{metrics['images_with_ground_truth']}")
        print(f"Total Substitutions: {metrics['total_substitutions']}")
        print(f"Total Deletions: {metrics['total_deletions']}")
        print(f"Total Insertions: {metrics['total_insertions']}")
        print("="*60)

# === Main Entry Point ===
def main():
    DATASET_PATH = r"Indonesian License Plate Recognition Dataset\images\test"
    GROUND_TRUTH_FILE = r"Indonesian License Plate Recognition Dataset\labels\ground_truth.csv"
    OUTPUT_FILE = "ocr_results.csv"
    LMSTUDIO_URL = "http://localhost:1234"
    MODEL_NAME = "C:/Users/454ken warden/.lmstudio/models/lmstudio-community/Qwen2-VL-2B-Instruct-GGUF/Qwen2-VL-2B-Instruct-Q4_K_M.gguf"

    ocr = LicensePlateOCR(LMSTUDIO_URL, MODEL_NAME)
    print("Starting License Plate OCR with VLM...")
    print(f"Dataset Path: {DATASET_PATH}")
    print(f"Ground Truth File: {GROUND_TRUTH_FILE}")
    print(f"Model: {MODEL_NAME}")
    print(f"LMStudio URL: {LMSTUDIO_URL}")
    print("-" * 60)
    
    try:
        results = ocr.process_dataset(DATASET_PATH, GROUND_TRUTH_FILE)
        if results:
            ocr.save_results_to_csv(OUTPUT_FILE)
            ocr.print_summary()
        else:
            print("No images were processed successfully!")
    except Exception as e:
        print(f"Error processing dataset: {e}")
        print("Please make sure:")
        print("1. LMStudio is running on the specified URL")
        print("2. Model is loaded in LMStudio")
        print("3. Dataset path exists and contains images")
        print("4. Ground truth file format is correct")

if __name__ == "__main__":
    main()
