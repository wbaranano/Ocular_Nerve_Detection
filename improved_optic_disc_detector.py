import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class OpticDiscDetector:
    def __init__(self):
        self.debug = True
        self.image_type = None  # Will be set to fundus or bscan
        
    def load_image(self, image_path):
        """Load and return grayscale image"""
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Cannot load image: {image_path}")
        return image
    
    def detect_image_type(self, image):
        """Automatically detect if image is fundus or B-scan based on characteristics"""
        h, w = image.shape
        aspect_ratio = w / h
        
        # B-scans are typically wider than tall (aspect ratio > 2)
        # and have characteristic horizontal layering
        if aspect_ratio > 2.0:
        
            horizontal_gradient = np.diff(image, axis=0)
            horizontal_variation = np.std(horizontal_gradient)
            
            vertical_gradient = np.diff(image, axis=1)
            vertical_variation = np.std(vertical_gradient)
            
        
            if horizontal_variation > vertical_variation * 1.5:
                self.image_type = 'bscan'
                if self.debug:
                    print(f"Detected B-scan image (aspect: {aspect_ratio:.2f})")
                return 'bscan'
        
        self.image_type = 'fundus'
        if self.debug:
            print(f"Detected fundus image (aspect: {aspect_ratio:.2f})")
        return 'fundus'
    
    def preprocess_fundus_image(self, image):
        """Preprocessing for fundus images"""
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        return enhanced
    
    def preprocess_bscan_image(self, image):
        """Preprocessing specifically for B-scan images"""
        # B-scans often have speckle noise, use different denoising
        denoised = cv2.bilateralFilter(image, 9, 75, 75)
        

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 4))  
        enhanced = clahe.apply(denoised)
   
        kernel_horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 1))
        enhanced = cv2.morphologyEx(enhanced, cv2.MORPH_TOPHAT, kernel_horizontal)
        enhanced = cv2.add(image, enhanced)
        
        return enhanced
    
    def preprocess_image(self, image):
        """Adaptive preprocessing based on image type"""
        image_type = self.detect_image_type(image)
        
        if image_type == 'bscan':
            return self.preprocess_bscan_image(image)
        else:
            return self.preprocess_fundus_image(image)
    
    def find_bright_regions_fundus(self, image):
        """Find bright regions in fundus images"""
        mean_val = np.mean(image)
        std_val = np.std(image)
        
        methods = []
        
        thresh1 = min(mean_val + 2 * std_val, 255)
        _, binary1 = cv2.threshold(image, thresh1, 255, cv2.THRESH_BINARY)
        methods.append(("Statistical", binary1, thresh1))
        
        thresh2 = np.percentile(image, 98)
        _, binary2 = cv2.threshold(image, thresh2, 255, cv2.THRESH_BINARY)
        methods.append(("98th Percentile", binary2, thresh2))
        
        thresh3, binary3 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        methods.append(("Otsu", binary3, thresh3))
        
        binary4 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
        methods.append(("Adaptive", binary4, 0))
        
        return methods
    
    def find_bright_regions_bscan(self, image):
        """Find bright regions in B-scan images (optic nerve head appears as bright vertical structure)"""
        mean_val = np.mean(image)
        std_val = np.std(image)
        
        methods = []
        
    
        thresh1 = min(mean_val + 1.5 * std_val, 255)
        _, binary1 = cv2.threshold(image, thresh1, 255, cv2.THRESH_BINARY)
        methods.append(("B-scan Statistical", binary1, thresh1))
        
     
        thresh2 = np.percentile(image, 95)
        _, binary2 = cv2.threshold(image, thresh2, 255, cv2.THRESH_BINARY)
        methods.append(("95th Percentile", binary2, thresh2))
        
        thresh3, binary3 = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        methods.append(("Otsu B-scan", binary3, thresh3))
        
      
        binary4 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 21, 2)
        methods.append(("Adaptive B-scan", binary4, 0))
        
  
        thresh5 = mean_val + 0.8 * std_val
        _, binary5 = cv2.threshold(image, thresh5, 255, cv2.THRESH_BINARY)
        methods.append(("B-scan Custom", binary5, thresh5))
        
        if self.debug:
            print(f"B-scan image stats: mean={mean_val:.1f}, std={std_val:.1f}")
            for name, _, thresh in methods:
                if thresh > 0:
                    print(f"{name} threshold: {thresh:.1f}")
        
        return methods
    
    def find_bright_regions(self, image):
        """Adaptive bright region detection based on image type"""
        if self.image_type == 'bscan':
            return self.find_bright_regions_bscan(image)
        else:
            return self.find_bright_regions_fundus(image)
    
    def analyze_contour_fundus(self, contour, original_image):
        """Analyze contour for fundus images (circular optic disc)"""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if area < 100 or perimeter == 0:
            return None
        
        circularity = 4 * np.pi * area / (perimeter * perimeter)
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h
        
        (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
        center = (int(center_x), int(center_y))
        
        mask = np.zeros(original_image.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [contour], 255)
        mean_intensity = cv2.mean(original_image, mask=mask)[0]
        
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        return {
            'contour': contour,
            'center': center,
            'radius': int(radius),
            'area': area,
            'perimeter': perimeter,
            'circularity': circularity,
            'aspect_ratio': aspect_ratio,
            'intensity': mean_intensity,
            'solidity': solidity,
            'bbox': (x, y, w, h),
            'height': h,
            'width': w
        }
    
    def analyze_contour_bscan(self, contour, original_image):
        """Analyze contour for B-scan images (vertical nerve head structure)"""
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if area < 50 or perimeter == 0:  # Lower threshold for B-scans
            return None
        
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(h) / w  # Height/width for vertical structures
        
      
        if aspect_ratio < 0.8:  # Should be taller than wide
            return None
        
        center = (int(x + w/2), int(y + h/2))
        
        mask = np.zeros(original_image.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [contour], 255)
        mean_intensity = cv2.mean(original_image, mask=mask)[0]
        
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0
        
        
        nerve_height = h
        nerve_width = w
        
        return {
            'contour': contour,
            'center': center,
            'area': area,
            'perimeter': perimeter,
            'aspect_ratio': aspect_ratio,
            'intensity': mean_intensity,
            'solidity': solidity,
            'bbox': (x, y, w, h),
            'height': nerve_height,
            'width': nerve_width,
            'nerve_head_height': nerve_height  # measurments for b scan
        }
    
    def analyze_contour(self, contour, original_image):
        """Adaptive contour analysis based on image type"""
        if self.image_type == 'bscan':
            return self.analyze_contour_bscan(contour, original_image)
        else:
            return self.analyze_contour_fundus(contour, original_image)
    
    def filter_candidates_fundus(self, candidates, image_shape):
        """Filter candidates for fundus images"""
        if not candidates:
            return []
        
        h, w = image_shape
        filtered = []
        
        for candidate in candidates:
            if 300 < candidate['area'] < 10000:
                if candidate['circularity'] > 0.2:
                    if 0.5 < candidate['aspect_ratio'] < 2.0:
                        if candidate['solidity'] > 0.7:
                            x, y = candidate['center']
                            margin = 50
                            if (margin < x < w - margin and margin < y < h - margin):
                                filtered.append(candidate)
        
        def score_candidate(cand):
            brightness_score = cand['intensity'] / 255.0
            shape_score = cand['circularity']
            size_score = min(cand['area'] / 3000.0, 1.0)
            return 0.5 * brightness_score + 0.3 * shape_score + 0.2 * size_score
        
        filtered.sort(key=score_candidate, reverse=True)
        return filtered
    
    def filter_candidates_bscan(self, candidates, image_shape):
        """Filter candidates for B-scan images"""
        if not candidates:
            return []
        
        h, w = image_shape
        filtered = []
        
        for candidate in candidates:
          
            if 200 < candidate['area'] < 8000:  
                if candidate['aspect_ratio'] > 1.2:
                    if candidate['solidity'] > 0.6:  
                        if candidate['height'] > 20:  
                            x, y = candidate['center']
                            margin = 30  # Smaller margin
                            if (margin < x < w - margin and margin < y < h - margin):
                                filtered.append(candidate)
        
        def score_bscan_candidate(cand):
            brightness_score = cand['intensity'] / 255.0
            vertical_score = min(cand['aspect_ratio'] / 3.0, 1.0)  # Favor vertical structures
            size_score = min(cand['area'] / 2000.0, 1.0)
            height_score = min(cand['height'] / 100.0, 1.0)  # Favor taller structures
            return 0.4 * brightness_score + 0.2 * vertical_score + 0.2 * size_score + 0.2 * height_score
        
        filtered.sort(key=score_bscan_candidate, reverse=True)
        return filtered
    
    def filter_candidates(self, candidates, image_shape):
        """Adaptive candidate filtering based on image type"""
        if self.image_type == 'bscan':
            return self.filter_candidates_bscan(candidates, image_shape)
        else:
            return self.filter_candidates_fundus(candidates, image_shape)
    
    def detect_optic_disc(self, image_path):
        """Main detection function with B-scan support"""
        image = self.load_image(image_path)
        processed = self.preprocess_image(image)  #auto ditects image type fingers crossed
        
        if self.debug:
            print(f"\nProcessing: {os.path.basename(image_path)}")
            print(f"Image type: {self.image_type}")
            print(f"Image shape: {image.shape}")
            print(f"Intensity range: {image.min()} - {image.max()}")
        
        threshold_methods = self.find_bright_regions(processed)
        all_candidates = []
        
        for method_name, binary_mask, threshold_val in threshold_methods:
            # changes operation based on image type
            if self.image_type == 'bscan':
                # make sure it does not lose the kernal structure
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5))  # Vertical kernal
            else:
                # Circular structures for fundus
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            
            cleaned = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if self.debug:
                print(f"{method_name}: {len(contours)} contours found")
            
            for contour in contours:
                candidate = self.analyze_contour(contour, image)
                if candidate:
                    candidate['method'] = method_name
                    all_candidates.append(candidate)
        
        good_candidates = self.filter_candidates(all_candidates, image.shape)
        
        if self.debug:
            print(f"Found {len(good_candidates)} good candidates after filtering")
            for i, cand in enumerate(good_candidates[:3]):
                if self.image_type == 'bscan':
                    print(f"  {i+1}. Center: {cand['center']}, Height: {cand['height']}, "
                          f"Width: {cand['width']}, Intensity: {cand['intensity']:.1f}")
                else:
                    print(f"  {i+1}. Center: {cand['center']}, Radius: {cand.get('radius', 'N/A')}, "
                          f"Intensity: {cand['intensity']:.1f}, Circularity: {cand.get('circularity', 'N/A'):.2f}")
        
        return good_candidates, image, processed
    
    def visualize_results(self, candidates, original_image, processed_image, save_path=None):
        """Visualize detection results for both fundus and B-scan images"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image
        axes[0, 0].imshow(original_image, cmap='gray')
        axes[0, 0].set_title(f'Original Image ({self.image_type.upper()})')
        axes[0, 0].axis('off')
        

        axes[0, 1].imshow(processed_image, cmap='gray')
        axes[0, 1].set_title('Preprocessed Image')
        axes[0, 1].axis('off')
        
    
        result = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
        
        if candidates:
            best_candidate = candidates[0]
            center = best_candidate['center']
            
            if self.image_type == 'bscan':
                # boundry box
                x, y, w, h = best_candidate['bbox']
                cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(result, center, 3, (0, 0, 255), -1)
                
                # label with height 
                height = best_candidate['height']
                cv2.putText(result, "Nerve Head", 
                           (center[0] - 40, center[1] - h//2 - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(result, f"Height: {height}px", 
                           (center[0] - 40, center[1] - h//2 - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
            else:
                #draw the circle 
                radius = best_candidate.get('radius', 20)
                cv2.circle(result, center, radius, (0, 255, 0), 3)
                cv2.circle(result, center, 3, (0, 0, 255), -1)
                cv2.drawContours(result, [best_candidate['contour']], -1, (0, 255, 0), 2)
                
                cv2.putText(result, "Optic Disc", 
                           (center[0] - 40, center[1] - radius - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        title = f'Best Detection ({self.image_type.upper()})'
        axes[1, 0].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        axes[1, 0].set_title(title)
        axes[1, 0].axis('off')
        
        # visulize the image :)
        if candidates:
            if self.image_type == 'bscan':
                thresh_val = np.percentile(processed_image, 95)
            else:
                thresh_val = np.percentile(processed_image, 98)
            _, thresh_img = cv2.threshold(processed_image, thresh_val, 255, cv2.THRESH_BINARY)
            
            axes[1, 1].imshow(thresh_img, cmap='gray')
            axes[1, 1].set_title('Threshold Visualization')
            axes[1, 1].axis('off')
        else:
            axes[1, 1].text(0.5, 0.5, 'No Detection', transform=axes[1, 1].transAxes,
                           ha='center', va='center', fontsize=16)
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            cv2.imwrite(save_path.replace('.png', '_result.png'), result)
        
        plt.show()
        
        return result

def process_all_images():
    """Process all images with B-scan support"""
    detector = OpticDiscDetector()
    
    # lets it use mutliple points of inpt
    input_dir = "/Users/williambaranano/Downloads/osv images/DR1-additional-marked-images"
    
    if not os.path.exists(input_dir):
        print(f"Input directory not found: {input_dir}")
        return
    
    # Look for multiple formats, :/
    image_files = []
    for ext in ['.pgm', '.tif', '.tiff', '.jpg', '.jpeg', '.png', '.bmp']:
        image_files.extend([f for f in os.listdir(input_dir) if f.lower().endswith(ext)])
    
    if not image_files:
        print("No image files found in the directory")
        return
    
    print(f"Found {len(image_files)} images to process")
    print(f"Processing all images in: {input_dir}")
    print("-" * 60)
    
    successful_detections = 0
    failed_detections = 0
    results_summary = []
    fundus_count = 0
    bscan_count = 0
    
    for i, filename in enumerate(image_files, 1):
        image_path = os.path.join(input_dir, filename)
        
        print(f"\n[{i}/{len(image_files)}] Processing: {filename}")
        
        try:
            candidates, original, processed = detector.detect_optic_disc(image_path)
            
            output_dir = "/Users/williambaranano/Desktop/python/optic_disc_results"
            os.makedirs(output_dir, exist_ok=True)
            
            base_filename = os.path.splitext(filename)[0]
            save_path = os.path.join(output_dir, f"{base_filename}_detection.png")
            
           
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            axes[0, 0].imshow(original, cmap='gray')
            axes[0, 0].set_title(f'Original ({detector.image_type.upper()})')
            axes[0, 0].axis('off')
            
            axes[0, 1].imshow(processed, cmap='gray')
            axes[0, 1].set_title('Preprocessed Image')
            axes[0, 1].axis('off')
            
            result = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
            
            if candidates:
                best_candidate = candidates[0]
                center = best_candidate['center']
                
                if detector.image_type == 'bscan':
                    bscan_count += 1
                    x, y, w, h = best_candidate['bbox']
                    cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.circle(result, center, 3, (0, 0, 255), -1)
                    
                    height = best_candidate['height']
                    cv2.putText(result, "Nerve Head", 
                               (center[0] - 40, center[1] - h//2 - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(result, f"Height: {height}px", 
                               (center[0] - 40, center[1] - h//2 - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    
                    measurement_type = "nerve_head_height"
                    measurement_value = height
                else:
                    fundus_count += 1
                    radius = best_candidate.get('radius', 20)
                    cv2.circle(result, center, radius, (0, 255, 0), 3)
                    cv2.circle(result, center, 3, (0, 0, 255), -1)
                    cv2.drawContours(result, [best_candidate['contour']], -1, (0, 255, 0), 2)
                    
                    cv2.putText(result, "Optic Disc", 
                               (center[0] - 40, center[1] - radius - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    measurement_type = "optic_disc_radius"
                    measurement_value = radius
                
                axes[1, 0].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
                axes[1, 0].set_title(f'Best Detection ({detector.image_type.upper()})')
                axes[1, 0].axis('off')
                
                thresh_val = np.percentile(processed, 95 if detector.image_type == 'bscan' else 98)
                _, thresh_img = cv2.threshold(processed, thresh_val, 255, cv2.THRESH_BINARY)
                axes[1, 1].imshow(thresh_img, cmap='gray')
                axes[1, 1].set_title('Threshold Visualization')
                axes[1, 1].axis('off')
                
                successful_detections += 1
                
                results_summary.append({
                    'filename': filename,
                    'status': 'SUCCESS',
                    'image_type': detector.image_type,
                    'center': best_candidate['center'],
                    'intensity': best_candidate['intensity'],
                    'area': best_candidate['area'],
                    measurement_type: measurement_value,
                    'height': best_candidate.get('height', 'N/A'),
                    'width': best_candidate.get('width', 'N/A')
                })
                
                print(f"✓ DETECTED ({detector.image_type}) - Center: {center}, {measurement_type}: {measurement_value}")
                
            else:
                axes[1, 0].text(0.5, 0.5, 'No Detection', transform=axes[1, 0].transAxes,
                               ha='center', va='center', fontsize=16, color='red')
                axes[1, 0].axis('off')
                axes[1, 1].text(0.5, 0.5, 'No Detection', transform=axes[1, 1].transAxes,
                               ha='center', va='center', fontsize=16, color='red')
                axes[1, 1].axis('off')
                
                failed_detections += 1
                results_summary.append({
                    'filename': filename,
                    'status': 'FAILED',
                    'image_type': detector.image_type
                })
                
                print(f"✗ NO DETECTION ({detector.image_type})")
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            cv2.imwrite(save_path.replace('.png', '_result.png'), result)
            plt.close()
            
        except Exception as e:
            print(f"✗ ERROR: {str(e)}")
            failed_detections += 1
            results_summary.append({
                'filename': filename,
                'status': 'ERROR',
                'error': str(e)
            })
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE - SUMMARY")
    print("="*60)
    print(f"Total images processed: {len(image_files)}")
    print(f"Fundus images: {fundus_count}")
    print(f"B-scan images: {bscan_count}")
    print(f"Successful detections: {successful_detections}")
    print(f"Failed detections: {failed_detections}")
    print(f"Success rate: {(successful_detections/len(image_files)*100):.1f}%")
    

    output_dir = "/Users/williambaranano/Desktop/python/optic_disc_results"
    csv_path = os.path.join(output_dir, "detection_results_with_bscans.csv")
    
    with open(csv_path, 'w') as f:
        f.write("Filename,Status,Image_Type,Center_X,Center_Y,Intensity,Area,Height,Width,Measurement_Value\n")
        for result in results_summary:
            if result['status'] == 'SUCCESS':
                measurement = result.get('nerve_head_height', result.get('optic_disc_radius', 'N/A'))
                f.write(f"{result['filename']},SUCCESS,{result['image_type']},"
                       f"{result['center'][0]},{result['center'][1]},"
                       f"{result['intensity']:.1f},{result['area']:.0f},"
                       f"{result['height']},{result['width']},{measurement}\n")
            else:
                f.write(f"{result['filename']},{result['status']},{result.get('image_type', 'unknown')},,,,,,\n")
    
    print(f"\nResults saved to: {csv_path}")
    
    return results_summary

def test_detection():
    """Test with a single image (supports both fundus and B-scans)"""
    detector = OpticDiscDetector()

    input_dir = "/Users/williambaranano/Downloads/osv images/DR1-additional-marked-images"
    
    if not os.path.exists(input_dir):
        print(f"Input directory not found: {input_dir}")
        return
    

    image_files = []
    for ext in ['.pgm', '.tif', '.tiff', '.jpg', '.jpeg', '.png', '.bmp']:
        image_files.extend([f for f in os.listdir(input_dir) if f.lower().endswith(ext)])
    
    if not image_files:
        print("No image files found in the directory")
        return
    
  
    test_image = os.path.join(input_dir, image_files[0])
    print(f"Testing with: {image_files[0]}")
    
    try:
        candidates, original, processed = detector.detect_optic_disc(test_image)
        
        if candidates:
            best = candidates[0]
            print(f"✓ Detection successful!")
            print(f"Image type: {detector.image_type}")
            print(f"Center: {best['center']}")
            print(f"Intensity: {best['intensity']:.1f}")
            
            if detector.image_type == 'bscan':
                print(f"Nerve head height: {best['height']} pixels")
            else:
                print(f"Optic disc radius: {best.get('radius', 'N/A')} pixels")
            
           
            detector.visualize_results(candidates, original, processed)
            return best
        else:
            print("✗ No detection found")
            return None
    except Exception as e:
        print(f"✗ Error: {e}")
        return None

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        print("Processing ALL images (fundus and B-scans)...")
        process_all_images()
    else:
        print("Processing SINGLE image (use '--all' flag to process all images)")
        test_detection()

#python3 improved_optic_disc_detector.py --all
