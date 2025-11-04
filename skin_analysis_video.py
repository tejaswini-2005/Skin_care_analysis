#!/usr/bin/env python3
"""
Real-time Skin Type Analysis using OpenCV and Deep Learning

This script uses a trained EfficientNetV2 model to analyze skin type from video feed.
It detects faces in real-time and classifies them as dry, normal, or oily skin.

Features:
- Real-time video capture and processing
- Face detection with bounding boxes
- Skin type classification with confidence scores
- Save frame analysis functionality
- Batch image processing

Author: AI Assistant
Date: October 2025
"""

import cv2
import numpy as np
import os
import pickle
import tensorflow as tf
from tensorflow.keras.models import load_model
import tensorflow_hub as hub
import matplotlib.pyplot as plt

class SkinAnalyzer:
    def __init__(self, model_path=None, classes_path=None):
        """
        Initialize the skin analyzer
        
        Args:
            model_path: Path to the trained model file
            classes_path: Path to the class labels pickle file
        """
        self.model = None
        self.class_labels = None
        self.face_cascade = None
        
        # Set default paths
        if model_path is None:
            model_path = os.path.join('model', 'skin_type_classification_model.h5')
        if classes_path is None:
            classes_path = os.path.join('model', 'class_labels.pkl')
            
        self.model_path = model_path
        self.classes_path = classes_path
        
        # Load components
        self.load_model()
        self.load_class_labels()
        self.setup_face_detection()
    
    def load_model(self):
        """Load the trained model"""
        try:
            print("Loading trained model...")
            self.model = load_model(self.model_path, custom_objects={'KerasLayer': hub.KerasLayer})
            print("✅ Model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
        return True
    
    def load_class_labels(self):
        """Load class labels"""
        try:
            with open(self.classes_path, 'rb') as f:
                self.class_labels = pickle.load(f)
            print(f"✅ Class labels loaded: {self.class_labels}")
        except Exception as e:
            print(f"❌ Error loading class labels: {e}")
            self.class_labels = ['dry', 'normal', 'oily']  # Default labels
        return True
    
    def setup_face_detection(self):
        """Setup face detection"""
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            print("✅ Face detection setup complete!")
        except Exception as e:
            print(f"❌ Error setting up face detection: {e}")
            return False
        return True
    
    def preprocess_image(self, img, target_size=(224, 224)):
        """
        Preprocess image for model prediction
        
        Args:
            img: Input image (BGR format)
            target_size: Target size for resizing
            
        Returns:
            Preprocessed image batch
        """
        # Resize image
        img_resized = cv2.resize(img, target_size)
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        
        # Normalize pixel values to [0, 1]
        img_normalized = img_rgb.astype(np.float32) / 255.0
        
        # Add batch dimension
        img_batch = np.expand_dims(img_normalized, axis=0)
        
        return img_batch
    
    def predict_skin_type(self, img):
        """
        Predict skin type from image
        
        Args:
            img: Input image
            
        Returns:
            tuple: (predicted_class, confidence, all_predictions)
        """
        if self.model is None or self.class_labels is None:
            return None, 0.0, None
        
        # Preprocess image
        processed_img = self.preprocess_image(img)
        
        # Make prediction
        predictions = self.model.predict(processed_img, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        predicted_class = self.class_labels[predicted_class_idx]
        
        return predicted_class, confidence, predictions[0]
    
    def detect_faces(self, img):
        """
        Detect faces in image
        
        Args:
            img: Input image
            
        Returns:
            Array of face coordinates
        """
        if self.face_cascade is None:
            return []
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))
        return faces
    
    def extract_face_region(self, img, face_coords, padding=20):
        """
        Extract face region with padding
        
        Args:
            img: Input image
            face_coords: Face coordinates (x, y, w, h)
            padding: Padding around face
            
        Returns:
            tuple: (face_image, adjusted_coordinates)
        """
        x, y, w, h = face_coords
        
        # Add padding
        x_start = max(0, x - padding)
        y_start = max(0, y - padding)
        x_end = min(img.shape[1], x + w + padding)
        y_end = min(img.shape[0], y + h + padding)
        
        face_img = img[y_start:y_end, x_start:x_end]
        return face_img, (x_start, y_start, x_end - x_start, y_end - y_start)
    
    def analyze_video_realtime(self, camera_id=0, save_dir="analyzed_frames"):
        """
        Real-time video analysis
        
        Args:
            camera_id: Camera device ID (0 for default camera)
            save_dir: Directory to save analyzed frames
        """
        # Create save directory
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # Initialize video capture
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print("❌ Error: Could not open camera")
            return
        
        print("🎥 Starting real-time skin analysis...")
        print("📋 Controls:")
        print("   - Press 'q' to quit")
        print("   - Press 's' to save current frame analysis")
        print("   - Press 'r' to reset analysis")
        print("   - Make sure your face is well-lit and centered")
        
        frame_count = 0
        analysis_results = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Error: Could not read frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect faces
            faces = self.detect_faces(frame)
            
            # Process each detected face
            for i, (x, y, w, h) in enumerate(faces):
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                
                # Extract face region for analysis
                face_img, face_coords = self.extract_face_region(frame, (x, y, w, h))
                
                if face_img.size > 0:
                    # Predict skin type
                    predicted_class, confidence, all_predictions = self.predict_skin_type(face_img)
                    
                    if predicted_class is not None:
                        # Display results on frame
                        result_text = f"Skin Type: {predicted_class}"
                        confidence_text = f"Confidence: {confidence:.2f}"
                        
                        # Position text above the face rectangle
                        cv2.putText(frame, result_text, (x, y-30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, confidence_text, (x, y-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        
                        # Show all class probabilities
                        for j, (class_name, prob) in enumerate(zip(self.class_labels, all_predictions)):
                            prob_text = f"{class_name}: {prob:.3f}"
                            cv2.putText(frame, prob_text, (10, 30 + j*20), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add instructions
            cv2.putText(frame, "Press 'q' to quit, 's' to save analysis", (10, frame.shape[0]-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow('Real-time Skin Type Analysis', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and len(faces) > 0:
                # Save current analysis
                timestamp = cv2.getTickCount()
                for i, (x, y, w, h) in enumerate(faces):
                    face_img, _ = self.extract_face_region(frame, (x, y, w, h))
                    if face_img.size > 0:
                        predicted_class, confidence, all_predictions = self.predict_skin_type(face_img)
                        
                        if predicted_class is not None:
                            # Save face image
                            filename = os.path.join(save_dir, f"analyzed_face_{timestamp}_{i}.jpg")
                            cv2.imwrite(filename, face_img)
                            
                            # Save full frame
                            frame_filename = os.path.join(save_dir, f"analyzed_frame_{timestamp}.jpg")
                            cv2.imwrite(frame_filename, frame)
                            
                            # Store analysis result
                            result = {
                                'timestamp': timestamp,
                                'predicted_class': predicted_class,
                                'confidence': float(confidence),
                                'all_predictions': {name: float(prob) for name, prob in zip(self.class_labels, all_predictions)},
                                'face_filename': filename,
                                'frame_filename': frame_filename
                            }
                            analysis_results.append(result)
                            print(f"✅ Analysis saved: {predicted_class} (confidence: {confidence:.3f})")
            
            frame_count += 1
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"🏁 Analysis complete! Processed {frame_count} frames")
        print(f"💾 Saved {len(analysis_results)} analyses to {save_dir}")
        
        return analysis_results
    
    def analyze_image(self, image_path, display_result=True):
        """
        Analyze a single image
        
        Args:
            image_path: Path to the image file
            display_result: Whether to display the result
            
        Returns:
            Analysis results
        """
        # Read image
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Error: Could not load image from {image_path}")
            return None
        
        # Detect faces
        faces = self.detect_faces(img)
        
        if len(faces) == 0:
            print("❌ No faces detected in the image")
            return None
        
        print(f"✅ Detected {len(faces)} face(s)")
        
        results = []
        
        # Analyze each face
        for i, (x, y, w, h) in enumerate(faces):
            # Draw rectangle around face
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Extract face region
            face_img, _ = self.extract_face_region(img, (x, y, w, h))
            
            if face_img.size > 0:
                # Predict skin type
                predicted_class, confidence, all_predictions = self.predict_skin_type(face_img)
                
                if predicted_class is not None:
                    result = {
                        'face_id': i,
                        'predicted_class': predicted_class,
                        'confidence': float(confidence),
                        'all_predictions': {name: float(prob) for name, prob in zip(self.class_labels, all_predictions)}
                    }
                    results.append(result)
                    
                    print(f"Face {i+1}:")
                    print(f"  Predicted skin type: {predicted_class}")
                    print(f"  Confidence: {confidence:.3f}")
                    print("  All predictions:")
                    for class_name, prob in zip(self.class_labels, all_predictions):
                        print(f"    {class_name}: {prob:.3f}")
                    print()
                    
                    # Add text to image
                    result_text = f"Face {i+1}: {predicted_class} ({confidence:.2f})"
                    cv2.putText(img, result_text, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Display result if requested
        if display_result and results:
            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            plt.title('Skin Type Analysis Results')
            plt.axis('off')
            plt.show()
        
        return results

def main():
    """Main function for command line usage"""
    print("🚀 Initializing Skin Type Analyzer...")
    
    # Initialize analyzer
    analyzer = SkinAnalyzer()
    
    # Check if all components loaded successfully
    if analyzer.model is None or analyzer.class_labels is None or analyzer.face_cascade is None:
        print("❌ Failed to initialize analyzer. Please check model files.")
        return
    
    print("✅ Analyzer initialized successfully!")
    print("\n" + "="*50)
    print("SKIN TYPE ANALYSIS SYSTEM")
    print("="*50)
    print("Choose an option:")
    print("1. Real-time video analysis")
    print("2. Analyze single image")
    print("3. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            print("\n🎥 Starting real-time analysis...")
            results = analyzer.analyze_video_realtime()
            print(f"\n📊 Analysis complete! {len(results)} frames analyzed.")
            
        elif choice == '2':
            image_path = input("Enter path to image file: ").strip()
            if os.path.exists(image_path):
                print(f"\n🖼️ Analyzing image: {image_path}")
                results = analyzer.analyze_image(image_path)
                if results:
                    print(f"✅ Analysis complete! Found {len(results)} face(s).")
            else:
                print("❌ Image file not found!")
                
        elif choice == '3':
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please select 1, 2, or 3.")

if __name__ == "__main__":
    main()