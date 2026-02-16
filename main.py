import cv2
import numpy as np
import sqlite3
import pickle
from datetime import datetime, date
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify, Response
import os
import tempfile
import time
import logging
import threading
import pandas as pd

# Import enhanced YOLO face recognizer
try:
    from yolo_face_recognizer_enhanced import YOLOFaceRecognizer
    ENHANCED_YOLO_AVAILABLE = True
    print("✓ Enhanced YOLO Face Recognizer loaded")
except ImportError as e:
    ENHANCED_YOLO_AVAILABLE = False
    print(f"⚠ Enhanced YOLO not available: {e}")

# Required packages
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Setup comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('face_recognition.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Constants for availability
TENSORFLOW_AVAILABLE = False  # TensorFlow not compatible with Python 3.13
SKLEARN_AVAILABLE = True
DLIB_AVAILABLE = True
FACE_RECOGNITION_AVAILABLE = True

print("Face recognition system loaded successfully!")

# Fast native Python functions (no Numba compilation delays)
def fast_histogram(image_array, bins=32):
    """Fast histogram calculation using OpenCV (optimized C++ implementation)"""
    hist = cv2.calcHist([image_array], [0], None, [bins], [0, 256])
    return hist.flatten() / np.sum(hist) if np.sum(hist) > 0 else hist.flatten()

def fast_lbp_extract(image, radius=1):
    """Fast Local Binary Pattern extraction using vectorized operations"""
    # Use a simplified and faster approach
    rows, cols = image.shape
    if rows < 3 or cols < 3:
        return image
    
    # Simple edge detection that's much faster than full LBP
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
    edges = cv2.filter2D(image, -1, kernel)
    return np.abs(edges).astype(np.uint8)

class YOLOFaceRecognizer:
    """YOLO-style face recognition system using YuNet model for ultra-fast detection"""
    
    def __init__(self):
        self.yunet_model = None
        self.face_recognizer = None
        self.setup_models()
        logger.info("YOLOFaceRecognizer initialized")
    
    def setup_models(self):
        """Initialize YuNet face detection model"""
        start_time = time.time()
        try:
            # Load YuNet face detection model (YOLO-style)
            model_path = 'face_detection_yunet_2023mar.onnx'
            if os.path.exists(model_path):
                self.yunet_model = cv2.FaceDetectorYN.create(
                    model_path,
                    "",
                    (320, 240),  # Input size for speed
                    0.3,         # Lower score threshold for better detection
                    0.3,         # NMS threshold
                    5000         # Top K
                )
                logger.info("YuNet face detector loaded successfully")
                print("Using YuNet YOLO-style face detector")
            else:
                # Fallback to OpenCV DNN
                self.setup_opencv_dnn()
                
        except Exception as e:
            logger.error(f"Error setting up YuNet model: {str(e)}")
            self.setup_opencv_dnn()
    
    def setup_opencv_dnn(self):
        """Fallback to OpenCV DNN face detector"""
        try:
            # Try to load OpenCV DNN face detector
            net_path = 'opencv_face_detector_uint8.pb'
            config_path = 'opencv_face_detector.pbtxt'
            
            if os.path.exists(net_path) and os.path.exists(config_path):
                self.face_detector = cv2.dnn.readNetFromTensorflow(net_path, config_path)
                logger.info("Using OpenCV DNN face detector")
                print("Using OpenCV DNN face detector")
            else:
                # Final fallback to Haar cascades
                self.face_detector = cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                )
                logger.info("Using Haar cascade face detector (fallback)")
                print("Using Haar cascade face detector (fallback)")
        except Exception as e:
            logger.error(f"Error setting up fallback detector: {str(e)}")
            # Use Haar cascade as final fallback
            self.face_detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            print(f"Using Haar cascade face detector (final fallback): {e}")
    
    def detect_faces(self, image):
        """Detect faces using YuNet (YOLO-style) or fallback methods"""
        if self.yunet_model is not None:
            return self.detect_faces_yunet(image)
        elif hasattr(self, 'face_detector') and isinstance(self.face_detector, cv2.dnn.Net):
            return self.detect_faces_dnn(image)
        else:
            return self.detect_faces_haar(image)
    
    def detect_faces_yunet(self, image):
        """YuNet YOLO-style face detection - ultra fast"""
        try:
            height, width = image.shape[:2]
            
            # Set input size for the model
            self.yunet_model.setInputSize((width, height))
            
            # Detect faces
            _, faces = self.yunet_model.detect(image)
            
            if faces is None:
                return []
            
            # Convert YuNet format to our format (top, right, bottom, left)
            face_locations = []
            for face in faces:
                x, y, w, h = face[:4].astype(int)
                # Convert to (top, right, bottom, left)
                face_locations.append((y, x + w, y + h, x))
            
            return face_locations
            
        except Exception as e:
            logger.error(f"YuNet detection error: {str(e)}")
            return []
    
    def detect_faces_dnn(self, image):
        """OpenCV DNN face detection"""
        try:
            h, w = image.shape[:2]
            blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123])
            self.face_detector.setInput(blob)
            detections = self.face_detector.forward()
            
            faces = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    x1 = int(detections[0, 0, i, 3] * w)
                    y1 = int(detections[0, 0, i, 4] * h)
                    x2 = int(detections[0, 0, i, 5] * w)
                    y2 = int(detections[0, 0, i, 6] * h)
                    faces.append((y1, x2, y2, x1))
            return faces
        except Exception as e:
            logger.error(f"DNN detection error: {str(e)}")
            return []
    
    def detect_faces_haar(self, image):
        """Haar cascade face detection"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Ultra-fast detection with aggressive parameters
            height, width = gray.shape
            scale = 1.0
            if width > 400:
                scale = 400.0 / width
                new_width = 400
                new_height = int(height * scale)
                gray = cv2.resize(gray, (new_width, new_height))
            
            faces = self.face_detector.detectMultiScale(
                gray, 
                scaleFactor=1.3,
                minNeighbors=2,
                minSize=(20, 20),
                maxSize=(200, 200),
                flags=cv2.CASCADE_DO_CANNY_PRUNING
            )
            
            # Scale back and convert format
            if scale != 1.0:
                faces = [(int(x/scale), int(y/scale), int(w/scale), int(h/scale)) for (x, y, w, h) in faces]
            
            return [(y, x+w, y+h, x) for (x, y, w, h) in faces]
            
        except Exception as e:
            logger.error(f"Haar detection error: {str(e)}")
            return []
    
    def extract_face_encoding(self, image, face_location):
        """Extract face encoding using ultra-fast feature extraction optimized for registration speed"""
        start_time = time.time()
        top, right, bottom, left = face_location
        
        # Extract face region
        face_image = image[top:bottom, left:right]
        
        if face_image.size == 0:
            logger.warning("Empty face image extracted")
            return None
        
        # Convert to grayscale for feature extraction
        gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        
        # Resize to even smaller size for maximum speed (16x16)
        resized_face = cv2.resize(gray_face, (16, 16))
        
        # Ultra-fast feature extraction (minimal processing)
        try:
            # 1. Basic pixel-based features (fastest possible)
            features = []
            
            # Flatten the 16x16 image and take every 4th pixel for speed
            pixels = resized_face.flatten()[::4]  # Take every 4th pixel
            features.extend(pixels.astype(np.float32))
            
            # 2. Only essential statistical features
            features.extend([
                np.mean(resized_face),
                np.std(resized_face),
                np.max(resized_face),
                np.min(resized_face)
            ])
            
            # 3. Simple edge detection (much faster)
            dx = cv2.Sobel(resized_face, cv2.CV_64F, 1, 0, ksize=3)
            dy = cv2.Sobel(resized_face, cv2.CV_64F, 0, 1, ksize=3)
            features.extend([np.mean(dx), np.mean(dy)])
            
            encoding_time = time.time() - start_time
            logger.info(f"Ultra-fast face encoding extracted in {encoding_time:.4f} seconds")
            
            # Normalize features for better comparison
            feature_array = np.array(features, dtype=np.float32)
            if np.std(feature_array) > 0:
                feature_array = (feature_array - np.mean(feature_array)) / np.std(feature_array)
            
            return feature_array
            
        except Exception as e:
            logger.error(f"Error extracting face encoding: {str(e)}")
            # Ultra-simple fallback
            features = resized_face.flatten()[::8].astype(np.float32)  # Even more sparse sampling
            return np.array(features)
    
    def compare_faces(self, known_encoding, face_encoding, tolerance=0.6):
        """Compare two face encodings"""
        if known_encoding is None or face_encoding is None:
            return False, 1.0
        
        # Ensure both encodings have the same length
        min_len = min(len(known_encoding), len(face_encoding))
        known_encoding = known_encoding[:min_len]
        face_encoding = face_encoding[:min_len]
        
        # Calculate cosine similarity
        similarity = cosine_similarity([face_encoding], [known_encoding])[0][0]
        distance = 1 - similarity
        
        return distance <= tolerance, distance
    
    def find_face_match(self, face_encoding, known_encodings, known_names, tolerance=0.6):
        """Find the best match for a face encoding"""
        if not known_encodings or face_encoding is None:
            return "Unknown", 1.0
        
        best_match_distance = float('inf')
        best_match_name = "Unknown"
        
        for i, known_encoding in enumerate(known_encodings):
            is_match, distance = self.compare_faces(known_encoding, face_encoding, tolerance)
            
            if distance < best_match_distance:
                best_match_distance = distance
                if is_match:
                    best_match_name = known_names[i]
        
        return best_match_name, best_match_distance

# Initialize the enhanced face recognizer
if ENHANCED_YOLO_AVAILABLE:
    yolo_face_recognizer = YOLOFaceRecognizer() if FACE_RECOGNITION_AVAILABLE else None
    print("✓ Using Enhanced YOLO Face Recognition System")
else:
    # Fallback to original implementation
    yolo_face_recognizer = YOLOFaceRecognizer() if FACE_RECOGNITION_AVAILABLE else None
    print("⚠ Using original face recognition system")

class AttendanceSystem:
    def __init__(self):
        self.db_connection = None
        self.known_face_encodings = []
        self.known_face_names = []
        self.attendance_records = {}
        # Initialize database
        self.init_database()
        # Load existing face encodings
        self.load_face_encodings()
        
    def init_database(self):
        """Initialize SQLite database connection and create tables with optimizations"""
        start_time = time.time()
        try:
            self.db_connection = sqlite3.connect('attendance_system.db', check_same_thread=False)
            logger.info("Connected to SQLite database")
            
            # Enable WAL mode for better performance
            self.db_connection.execute("PRAGMA journal_mode = WAL")
            # Optimize SQLite for faster operations
            self.db_connection.execute("PRAGMA synchronous = NORMAL")
            self.db_connection.execute("PRAGMA cache_size = 10000")
            self.db_connection.execute("PRAGMA temp_store = MEMORY")
            
            cursor = self.db_connection.cursor()
            
            # Create students table with indexes
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS students (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT UNIQUE NOT NULL,
                    name TEXT NOT NULL,
                    email TEXT,
                    face_encoding BLOB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create attendance table with indexes
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT,
                    date DATE,
                    time TIME,
                    status TEXT DEFAULT 'Present',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (student_id) REFERENCES students(student_id),
                    UNIQUE(student_id, date)
                )
            """)
            
            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_students_student_id ON students(student_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_attendance_student_id ON attendance(student_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_attendance_date ON attendance(date)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_attendance_student_date ON attendance(student_id, date)")
            
            self.db_connection.commit()
            db_time = time.time() - start_time
            logger.info(f"Database initialized successfully in {db_time:.4f} seconds")
            print("Database initialized successfully!")
            
        except sqlite3.Error as err:
            logger.error(f"Database error: {err}")
            print(f"Database error: {err}")
            self.db_connection = None
            print(f"Database Error: Could not connect to database: {err}")
    
    # Flask will handle the web interface, so GUI code is removed.
        
    # Button creation is now handled by HTML templates in Flask.
    
    # Registration form will be handled by Flask route and HTML form.
    
    def register_student_with_photo_fast(self, student_id, name, email, image_path):
        """Ultra-fast student registration optimized for speed"""
        total_start_time = time.time()
        logger.info(f"Starting FAST registration for student {student_id}: {name}")
        
        try:
            if not FACE_RECOGNITION_AVAILABLE or yolo_face_recognizer is None:
                logger.error("Face recognition not available")
                return False
                
            # Load image with minimal processing
            image_start_time = time.time()
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not load image file: {image_path}")
                return False
            
            image_load_time = time.time() - image_start_time
            logger.info(f"Image loaded in {image_load_time:.4f} seconds")
                
            # Ultra-fast face detection with aggressive parameters
            detection_start_time = time.time()
            
            # Use Haar cascade directly for maximum speed
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Super aggressive detection parameters for speed
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=1.1,  # More sensitive
                minNeighbors=1,   # Very minimal for speed
                minSize=(20, 20),
                flags=cv2.CASCADE_DO_CANNY_PRUNING | cv2.CASCADE_SCALE_IMAGE
            )
            
            detection_time = time.time() - detection_start_time
            logger.info(f"Ultra-fast detection completed in {detection_time:.4f} seconds, found {len(faces)} faces")
            
            if len(faces) == 0:
                logger.warning("No face detected in the image")
                return False
            
            # Use the first detected face and convert to our format
            x, y, w, h = faces[0]
            face_location = (y, x+w, y+h, x)  # Convert to (top, right, bottom, left)
            logger.info(f"Using face location: {face_location}")
            
            # Ultra-fast encoding extraction
            encoding_start_time = time.time()
            top, right, bottom, left = face_location
            face_image = image[top:bottom, left:right]
            
            if face_image.size == 0:
                logger.error("Empty face region")
                return False
            
            # Minimal feature extraction for maximum speed
            gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            tiny_face = cv2.resize(gray_face, (8, 8))  # Extremely small for speed
            
            # Super simple encoding - just flattened pixels
            face_encoding = tiny_face.flatten().astype(np.float32)
            
            encoding_time = time.time() - encoding_start_time
            logger.info(f"Ultra-fast encoding extracted in {encoding_time:.4f} seconds")
            
            # Store in database with minimal processing
            db_start_time = time.time()
            cursor = self.db_connection.cursor()
            
            # Serialize face encoding
            face_encoding_blob = pickle.dumps(face_encoding)
            
            query = """INSERT INTO students (student_id, name, email, face_encoding) 
                      VALUES (?, ?, ?, ?)"""
            cursor.execute(query, (student_id, name, email, face_encoding_blob))
            self.db_connection.commit()
            db_time = time.time() - db_start_time
            logger.info(f"Database insertion completed in {db_time:.4f} seconds")
            
            # Add to memory for immediate use
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(name)
            
            total_time = time.time() - total_start_time
            logger.info(f"Student {name} registered successfully in {total_time:.4f} seconds total (FAST MODE)")
            return True
            
        except sqlite3.IntegrityError:
            logger.error("Student ID already exists!")
            return False
        except Exception as e:
            logger.error(f"Fast registration error: {e}")
            return False
    
    def register_student_with_photo(self, student_id, name, email, image_path):
        """Register a student with their photo for face recognition with performance logging"""
        total_start_time = time.time()
        logger.info(f"Starting registration for student {student_id}: {name}")
        
        try:
            if not FACE_RECOGNITION_AVAILABLE or yolo_face_recognizer is None:
                logger.error("Face recognition not available")
                print("Error: Face recognition not available. Please install required packages.")
                return False
                
            # Load and process the image
            image_start_time = time.time()
            image = cv2.imread(image_path)
            if image is None:
                logger.error(f"Could not load image file: {image_path}")
                print("Error: Could not load image file.")
                return False
            
            image_load_time = time.time() - image_start_time
            logger.info(f"Image loaded in {image_load_time:.4f} seconds")
                
            # Detect faces
            detection_start_time = time.time()
            face_locations = yolo_face_recognizer.detect_faces(image)
            detection_time = time.time() - detection_start_time
            logger.info(f"Face detection completed in {detection_time:.4f} seconds, found {len(face_locations)} faces")
            
            if len(face_locations) == 0:
                logger.warning("No face detected in the image")
                print("Error: No face detected in the image. Please use a clear photo.")
                return False
            
            # Use the first detected face
            face_location = face_locations[0]
            logger.info(f"Using face location: {face_location}")
            
            # Extract face encoding
            encoding_start_time = time.time()
            face_encoding = yolo_face_recognizer.extract_face_encoding(image, face_location)
            encoding_time = time.time() - encoding_start_time
            logger.info(f"Face encoding extraction completed in {encoding_time:.4f} seconds")
            
            if face_encoding is None:
                logger.error("Could not extract face features")
                print("Error: Could not extract face features.")
                return False
            
            # Store in database
            db_start_time = time.time()
            cursor = self.db_connection.cursor()
            
            # Serialize face encoding
            face_encoding_blob = pickle.dumps(face_encoding)
            
            query = """INSERT INTO students (student_id, name, email, face_encoding) 
                      VALUES (?, ?, ?, ?)"""
            cursor.execute(query, (student_id, name, email, face_encoding_blob))
            self.db_connection.commit()
            db_time = time.time() - db_start_time
            logger.info(f"Database insertion completed in {db_time:.4f} seconds")
            
            # Add to memory for immediate use
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(name)
            
            total_time = time.time() - total_start_time
            logger.info(f"Student {name} registered successfully in {total_time:.4f} seconds total")
            print(f"Student {name} registered successfully with face encoding")
            return True
            
        except sqlite3.IntegrityError:
            print("Error: Student ID already exists!")
            return False
        except Exception as e:
            print(f"Registration error: {e}")
            return False
    
    def load_face_encodings(self):
        """Load face encodings from database"""
        try:
            if not FACE_RECOGNITION_AVAILABLE:
                print("Warning: Face recognition not available. Skipping face encoding loading.")
                return
                
            cursor = self.db_connection.cursor()
            cursor.execute("SELECT name, face_encoding FROM students")
            results = cursor.fetchall()
            
            self.known_face_encodings = []
            self.known_face_names = []
            
            for name, face_encoding_blob in results:
                if face_encoding_blob:
                    face_encoding = pickle.loads(face_encoding_blob)
                    self.known_face_encodings.append(face_encoding)
                    self.known_face_names.append(name)
            
            print(f"Loaded {len(self.known_face_names)} student records")
            
        except Exception as e:
            print(f"Error loading face encodings: {e}")
    
    def initialize_camera_robust(self):
        """Robust camera initialization with multiple fallback options for macOS"""
        # Based on the camera test, we know camera 0 works with AVFoundation and Default
        configs_to_try = [
            (0, cv2.CAP_AVFOUNDATION, "AVFoundation"),
            (0, cv2.CAP_ANY, "Default"),
            (0, None, "Basic")  # No backend specified
        ]
        
        for idx, backend, backend_name in configs_to_try:
            try:
                logger.info(f"Attempting camera {idx} with {backend_name} backend")
                
                if backend is not None:
                    video_capture = cv2.VideoCapture(idx, backend)
                else:
                    video_capture = cv2.VideoCapture(idx)
                
                if video_capture.isOpened():
                    # Set camera properties for better compatibility
                    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    video_capture.set(cv2.CAP_PROP_FPS, 30)
                    
                    # Test if we can actually read a frame
                    ret, test_frame = video_capture.read()
                    if ret and test_frame is not None:
                        logger.info(f"Camera {idx} with {backend_name} backend working successfully")
                        return video_capture
                    else:
                        video_capture.release()
                        logger.warning(f"Camera {idx} {backend_name}: opened but cannot read frames")
                else:
                    logger.warning(f"Could not open camera {idx} with {backend_name} backend")
                        
            except Exception as e:
                logger.error(f"Error with camera {idx} {backend_name} backend: {str(e)}")
                if 'video_capture' in locals() and video_capture:
                    video_capture.release()
                continue
        
        logger.error("No working camera found with any configuration")
        return None
        
        logger.error("No working camera found with any configuration")
        return None

    def start_attendance(self):
        """Start the face recognition attendance system (for Flask)"""
        try:
            if not FACE_RECOGNITION_AVAILABLE:
                return "Face recognition not available. Please install face_recognition package."
                
            if not self.known_face_encodings:
                logger.warning("No students registered yet!")
                return "No students registered yet!"
            
            # Test camera initialization
            video_capture = self.initialize_camera_robust()
            if video_capture is None:
                error_msg = "Could not access camera. Please check camera permissions and ensure no other app is using the camera."
                logger.error(error_msg)
                return error_msg
            
            # Camera test successful, release it for now
            video_capture.release()
            logger.info("Camera test successful, ready for attendance")
            
            # Run attendance system in a separate thread to avoid blocking the web interface
            import threading
            attendance_thread = threading.Thread(target=self.run_attendance_system_threaded)
            attendance_thread.daemon = True
            attendance_thread.start()
            
            return "Attendance system started successfully. Camera window should open shortly."
            
        except Exception as e:
            error_msg = f"Error starting attendance system: {str(e)}"
            logger.error(error_msg)
            return error_msg

    def run_attendance_system_threaded(self):
        """Threaded version of attendance system"""
        try:
            self.run_attendance_system()
        except Exception as e:
            logger.error(f"Error in threaded attendance system: {str(e)}")

    def run_attendance_system(self):
        """Start the face recognition attendance system (for Flask)"""
        if not FACE_RECOGNITION_AVAILABLE:
            return "Face recognition not available. Please install face_recognition package."
            
        if not self.known_face_encodings:
            print("Warning: No students registered yet!")
            return "No students registered yet!"
        # Run attendance system directly (could be threaded if needed)
        self.run_attendance_system()
        return "Attendance process started. Check console for details."
    
    def run_attendance_system(self):
        """Run the main attendance recognition loop (console only)"""
        if not FACE_RECOGNITION_AVAILABLE:
            print("Error: Face recognition not available!")
            return
        
        # Use robust camera initialization
        video_capture = self.initialize_camera_robust()
        
        if video_capture is None:
            error_msg = "Error: Could not access any camera! Please check camera permissions and ensure no other app is using the camera."
            logger.error(error_msg)
            print(error_msg)
            print("\nTroubleshooting tips:")
            print("1. Check System Preferences > Security & Privacy > Camera")
            print("2. Make sure no other app (Zoom, Teams, etc.) is using the camera")
            print("3. Try restarting your computer")
            print("4. Check if camera works in other apps like Photo Booth")
            return

        print("Attendance system running - Processing frames for face recognition...")
        logger.info("Attendance system started successfully")
        today = date.today()
        daily_attendance = set()
        process_this_frame = True
        face_names = []
        frame_count = 0
        
        try:
            while frame_count < 300:  # Limit to 300 frames (about 10 seconds at 30fps)
                ret, frame = video_capture.read()
                if not ret:
                    logger.warning("Failed to read frame from camera")
                    break
                
                frame_count += 1
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = small_frame[:, :, ::-1]
                
                if process_this_frame and FACE_RECOGNITION_AVAILABLE and yolo_face_recognizer:
                    face_locations = yolo_face_recognizer.detect_faces(rgb_small_frame)
                    face_names = []
                    
                    for face_location in face_locations:
                        # Extract face encoding
                        face_encoding = yolo_face_recognizer.extract_face_encoding(rgb_small_frame, face_location)
                        
                        if face_encoding is not None and len(self.known_face_encodings) > 0:
                            name, distance = yolo_face_recognizer.find_face_match(
                                face_encoding, self.known_face_encodings, self.known_face_names
                            )
                            
                            if name != "Unknown" and distance < 0.6:
                                # Mark attendance if not already marked today
                                if name not in daily_attendance:
                                    self.mark_attendance(name)
                                    daily_attendance.add(name)
                                    print(f"✓ Attendance marked for {name}")
                                    logger.info(f"Attendance marked for {name}")
                            
                            face_names.append(name)
                        else:
                            face_names.append("Unknown")
                
                process_this_frame = not process_this_frame
                
                # Log face detection results every 30 frames
                if frame_count % 30 == 0:
                    if face_names:
                        logger.info(f"Frame {frame_count}: Detected faces: {face_names}")
                    else:
                        logger.info(f"Frame {frame_count}: No faces detected")
                
                # Small delay to prevent overwhelming the system
                import time
                time.sleep(0.033)  # ~30 FPS
                    
        except Exception as e:
            logger.error(f"Error during attendance system operation: {str(e)}", exc_info=True)
            print(f"Error during operation: {str(e)}")
        finally:
            video_capture.release()
            # Don't call cv2.destroyAllWindows() since we're not using imshow
            logger.info("Attendance system stopped")
            print("Attendance system stopped")
            if daily_attendance:
                print(f"✓ Total attendance marked: {len(daily_attendance)} students")
                print(f"Students present: {', '.join(daily_attendance)}")
            else:
                print("ℹ No faces recognized during this session")
    
    def run_attendance_system_console(self):
        """Run attendance system in console mode with live camera display (for direct execution)"""
        if not FACE_RECOGNITION_AVAILABLE:
            print("Error: Face recognition not available!")
            return
        
        # Use robust camera initialization
        video_capture = self.initialize_camera_robust()
        
        if video_capture is None:
            error_msg = "Error: Could not access any camera!"
            print(error_msg)
            return

        print("Attendance system running - Press 'q' to stop")
        print("This will show a live camera window...")
        today = date.today()
        daily_attendance = set()
        process_this_frame = True
        face_names = []
        
        try:
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    print("Failed to read frame from camera")
                    break
                
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = small_frame[:, :, ::-1]
                
                if process_this_frame and FACE_RECOGNITION_AVAILABLE and yolo_face_recognizer:
                    face_locations = yolo_face_recognizer.detect_faces(rgb_small_frame)
                    face_names = []
                    
                    for face_location in face_locations:
                        face_encoding = yolo_face_recognizer.extract_face_encoding(rgb_small_frame, face_location)
                        
                        if face_encoding is not None and len(self.known_face_encodings) > 0:
                            name, distance = yolo_face_recognizer.find_face_match(
                                face_encoding, self.known_face_encodings, self.known_face_names
                            )
                            
                            if name != "Unknown" and distance < 0.6:
                                if name not in daily_attendance:
                                    self.mark_attendance(name)
                                    daily_attendance.add(name)
                                    print(f"✓ Attendance marked for {name}")
                            
                            face_names.append(name)
                        else:
                            face_names.append("Unknown")
                else:
                    face_locations = []
                    
                process_this_frame = not process_this_frame
                
                # Draw rectangles around faces
                for i, (top, right, bottom, left) in enumerate(face_locations):
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    display_name = face_names[i] if i < len(face_names) else "Unknown"
                    cv2.putText(frame, display_name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)
                
                cv2.imshow('Smart Attendance System', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except Exception as e:
            print(f"Error during operation: {str(e)}")
        finally:
            video_capture.release()
            cv2.destroyAllWindows()
            print("Attendance system stopped")
            if daily_attendance:
                print(f"✓ Total attendance marked: {len(daily_attendance)} students")

    def mark_attendance(self, student_name):
        """Mark attendance for a student"""
        try:
            cursor = self.db_connection.cursor()
            
            # Get student_id from name
            cursor.execute("SELECT student_id FROM students WHERE name = ?", (student_name,))
            result = cursor.fetchone()
            
            if result:
                student_id = result[0]
                current_date = date.today()
                current_time = datetime.now().time()
                
                # Check if already marked for today
                cursor.execute("SELECT id FROM attendance WHERE student_id = ? AND date = ?", 
                             (student_id, current_date))
                existing = cursor.fetchone()
                
                if existing:
                    logger.info(f"Attendance already marked for {student_name} today")
                    print(f"✓ {student_name} - Already marked present today")
                    return False
                
                # Insert attendance record
                query = """INSERT INTO attendance (student_id, date, time, status) 
                          VALUES (?, ?, ?, 'Present')"""
                cursor.execute(query, (student_id, current_date, current_time))
                self.db_connection.commit()
                
                logger.info(f"Attendance marked for {student_name} ({student_id}) at {current_time}")
                print(f"✅ ATTENDANCE MARKED: {student_name} - {current_date} {current_time}")
                return True
            else:
                logger.warning(f"Student {student_name} not found in database")
                print(f"⚠️ Student {student_name} not found in database")
                return False
                
        except Exception as e:
            logger.error(f"Error marking attendance: {e}")
            print(f"❌ Error marking attendance for {student_name}: {e}")
            return False
    
    def view_attendance_records(self):
        """Display attendance records in a new window"""
        # Fetch and return records for Flask route
        try:
            if not self.db_connection:
                logger.error("No database connection available")
                return []
                
            cursor = self.db_connection.cursor()
            query = """
                SELECT s.student_id, s.name, a.date, a.time, a.status 
                FROM attendance a 
                JOIN students s ON a.student_id = s.student_id 
                ORDER BY a.date DESC, a.time DESC
            """
            cursor.execute(query)
            records = cursor.fetchall()
            logger.info(f"Retrieved {len(records)} attendance records")
            return records
        except Exception as e:
            logger.error(f"Error fetching records: {e}")
            print(f"Error fetching records: {e}")
            return []
    
    def export_attendance_data(self):
        """Export attendance data to CSV"""
        try:
            cursor = self.db_connection.cursor()
            query = """
                SELECT s.student_id, s.name, a.date, a.time, a.status 
                FROM attendance a 
                JOIN students s ON a.student_id = s.student_id 
                ORDER BY a.date DESC, a.time DESC
            """
            cursor.execute(query)
            records = cursor.fetchall()
            
            if not records:
                print("Info: No attendance records found!")
                return
            
            # Create DataFrame and export
            df = pd.DataFrame(records, columns=['Student ID', 'Name', 'Date', 'Time', 'Status'])
            
            filename = f"attendance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            # In Flask, file download will be handled by send_file
                
        except Exception as e:
            print(f"Error exporting data: {e}")
            print("Error: Failed to export data")
    
    def run(self):
        # Flask will run the web server, so this method is not needed.
        pass

class CameraStream:
    """Camera streaming class for web interface"""
    def __init__(self, attendance_system):
        self.attendance_system = attendance_system
        self.video_capture = None
        self.is_streaming = False
        self.daily_attendance = set()
        self.lock = threading.Lock()
        
    def start_stream(self):
        """Start the camera stream"""
        if self.is_streaming:
            return True
            
        self.video_capture = self.attendance_system.initialize_camera_robust()
        if self.video_capture is None:
            return False
            
        self.is_streaming = True
        self.daily_attendance = set()
        return True
        
    def stop_stream(self):
        """Stop the camera stream cleanly"""
        with self.lock:
            self.is_streaming = False
        
        # Give the thread a moment to recognize the state change
        time.sleep(0.1)

        if self.video_capture:
            try:
                self.video_capture.release()
                self.video_capture = None
                logger.info("Camera capture released.")
            except Exception as e:
                logger.error(f"Error releasing camera capture: {e}")
            
    def generate_frames(self):
        """Generate frames for streaming"""
        process_this_frame = True
        
        while True:
            with self.lock:
                if not self.is_streaming or not self.video_capture:
                    logger.info("Stopping frame generation loop.")
                    break
                
                ret, frame = self.video_capture.read()

            if not ret:
                logger.warning("Could not read frame from camera, stopping stream.")
                self.stop_stream()
                break
                
            # Process frame for face recognition
            if process_this_frame and FACE_RECOGNITION_AVAILABLE and yolo_face_recognizer:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = small_frame[:, :, ::-1]
                
                face_locations = yolo_face_recognizer.detect_faces(rgb_small_frame)
                face_names = []
                
                for face_location in face_locations:
                    face_encoding = yolo_face_recognizer.extract_face_encoding(rgb_small_frame, face_location)
                    
                    if face_encoding is not None and len(self.attendance_system.known_face_encodings) > 0:
                        name, distance = yolo_face_recognizer.find_face_match(
                            face_encoding, 
                            self.attendance_system.known_face_encodings, 
                            self.attendance_system.known_face_names
                        )
                        
                        if name != "Unknown" and distance < 0.6:
                            if name not in self.daily_attendance:
                                self.attendance_system.mark_attendance(name)
                                self.daily_attendance.add(name)
                                logger.info(f"Attendance marked for {name}")
                        
                        face_names.append(name)
                    else:
                        face_names.append("Unknown")
                
                # Draw rectangles around faces
                for i, (top, right, bottom, left) in enumerate(face_locations):
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4
                    
                    # Choose color based on recognition
                    name = face_names[i] if i < len(face_names) else "Unknown"
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)  # Green for known, red for unknown
                    
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                    cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                    
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)
                    
                    # Add attendance status
                    if name in self.daily_attendance:
                        cv2.putText(frame, "PRESENT", (left + 6, top - 10), font, 0.5, (0, 255, 0), 1)
            
            process_this_frame = not process_this_frame
            
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_bytes = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Requirements installation message
def show_requirements():
    requirements = """
    Required Libraries (install using pip):
    
    pip install opencv-python
    pip install face-recognition
    pip install mysql-connector-python
    pip install pillow
    pip install pandas
    pip install numpy
    
    Also ensure you have:
    - MySQL server running
    - A webcam connected
    - Good lighting for face recognition
    
    Database Setup:
    1. Create a MySQL database named 'attendance_system'
    2. Update the database credentials in the code
    3. The tables will be created automatically
    """
    print(requirements)

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this to a random secret key

# Initialize AttendanceSystem instance
attendance_system = AttendanceSystem()

# Initialize camera stream
camera_stream = CameraStream(attendance_system)

# Flask routes
@app.route('/')
def index():
    # Get system statistics
    stats = {
        'total_students': 0,
        'today_attendance': 0,
        'attendance_rate': 0
    }
    
    db_status = attendance_system.db_connection is not None
    
    if db_status:
        try:
            cursor = attendance_system.db_connection.cursor()
            
            # Total students
            cursor.execute("SELECT COUNT(*) FROM students")
            stats['total_students'] = cursor.fetchone()[0]
            
            # Today's attendance
            today = date.today()
            cursor.execute("SELECT COUNT(*) FROM attendance WHERE date = ?", (today,))
            stats['today_attendance'] = cursor.fetchone()[0]
            
            # Attendance rate (this week) - SQLite compatible
            cursor.execute("""
                SELECT 
                    COUNT(DISTINCT a.student_id) * 100.0 / COUNT(DISTINCT s.student_id) as rate
                FROM students s 
                LEFT JOIN attendance a ON s.student_id = a.student_id 
                    AND a.date >= date('now', '-7 days')
            """)
            result = cursor.fetchone()
            stats['attendance_rate'] = round(result[0] if result[0] else 0, 1)
            
        except Exception as e:
            print(f"Error getting stats: {e}")
    
    return render_template('index.html', 
                         stats=stats, 
                         db_status=db_status, 
                         face_recognition_status=FACE_RECOGNITION_AVAILABLE)

@app.route('/test_server', methods=['GET', 'POST'])
def test_server():
    """Simple test endpoint to verify server is working"""
    logger.info(f"Test server endpoint called with method: {request.method}")
    return jsonify({
        'status': 'success',
        'message': 'Server is working!',
        'method': request.method,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/register_ajax', methods=['POST'])
def register_ajax():
    """AJAX endpoint for robust student registration"""
    try:
        logger.info("=== AJAX REGISTRATION REQUEST RECEIVED ===")
        logger.info(f"Request method: {request.method}")
        logger.info(f"Request content type: {request.content_type}")
        logger.info(f"Request form keys: {list(request.form.keys())}")
        logger.info(f"Request files keys: {list(request.files.keys())}")
        
        student_id = request.form.get('student_id')
        name = request.form.get('name')
        email = request.form.get('email')
        captured_photo_path = request.form.get('captured_photo_path')

        logger.info(f"Form data: ID='{student_id}', Name='{name}', Email='{email}'")
        logger.info(f"Captured photo path from form: '{captured_photo_path}'")
        logger.info(f"Uploaded files: {list(request.files.keys())}")

        # Validate required fields
        if not student_id or not name:
            logger.error("Missing required fields: student_id or name")
            return jsonify({'success': False, 'message': 'Student ID and Name are required fields.'})

        temp_path = None
        is_uploaded_file = False

        # Handle photo source
        if captured_photo_path and os.path.exists(captured_photo_path):
            temp_path = captured_photo_path
            logger.info(f"Using photo captured from stream: {temp_path}")
        elif 'photo' in request.files:
            photo = request.files['photo']
            if photo and photo.filename:
                is_uploaded_file = True
                # Create a safe filename
                filename = f"upload_{student_id}_{int(time.time())}.jpg"
                temp_path = os.path.join(tempfile.gettempdir(), filename)
                
                logger.info(f"Saving uploaded photo to: {temp_path}")
                photo.save(temp_path)
                
                # Verify file was saved
                if os.path.exists(temp_path):
                    file_size = os.path.getsize(temp_path)
                    logger.info(f"Photo saved successfully, size: {file_size} bytes")
                else:
                    logger.error(f"Failed to save photo to {temp_path}")
                    return jsonify({'success': False, 'message': 'Failed to save uploaded photo.'})
            else:
                logger.warning("'photo' in request.files but filename is empty.")
        
        if not temp_path:
            logger.error("No valid photo source found in request.")
            return jsonify({'success': False, 'message': 'No photo provided. Please upload or capture a photo.'})

        logger.info(f"Proceeding with registration using image at: {temp_path}")

        # Check if database is available
        if not attendance_system.db_connection:
            logger.error("Database connection not available")
            return jsonify({'success': False, 'message': 'Database connection error. Please try again.'})

        # Use the main, robust registration method instead of the "fast" one
        logger.info("Starting face recognition registration process...")
        success = attendance_system.register_student_with_photo(student_id, name, email, temp_path)
        
        logger.info(f"Registration processing finished. Success: {success}")

        # Clean up the temporary file if it was uploaded
        if is_uploaded_file and temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
                logger.info(f"Cleaned up temporary file: {temp_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up temporary file {temp_path}: {e}")
        
        if success:
            logger.info(f"Successfully registered student {name} ({student_id})")
            return jsonify({'success': True, 'message': f'Student {name} registered successfully!'})
        else:
            logger.error(f"Registration failed for student {name} ({student_id})")
            return jsonify({'success': False, 'message': 'Registration failed. A face might not be detected, or the Student ID may already exist.'})
            
    except Exception as e:
        logger.error(f"--- Unhandled exception in /register_ajax: {str(e)} ---", exc_info=True)
        return jsonify({'success': False, 'message': f'A server error occurred: {str(e)}'})

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        route_start_time = time.time()
        student_id = request.form.get('student_id')
        name = request.form.get('name')
        email = request.form.get('email')
        
        logger.info(f"Registration request received for {student_id}: {name}")
        
        if 'photo' in request.files:
            photo = request.files['photo']
            if photo.filename != '':
                try:
                    logger.info(f"Processing photo file: {photo.filename}")
                    
                    # Save uploaded file temporarily - minimal processing
                    file_start_time = time.time()
                    temp_path = os.path.join(tempfile.gettempdir(), f"temp_{student_id}_{int(time.time())}.jpg")
                    photo.save(temp_path)
                    file_save_time = time.time() - file_start_time
                    logger.info(f"File saved in {file_save_time:.4f} seconds")
                    
                    # Ultra-aggressive image optimization for speed
                    img_opt_start_time = time.time()
                    img = cv2.imread(temp_path)
                    if img is not None:
                        height, width = img.shape[:2]
                        logger.info(f"Original image size: {width}x{height}")
                        # Extremely aggressive resizing for lightning-fast processing
                        if width > 200:  # Very small for maximum speed
                            scale = 200.0 / width
                            new_width = 200
                            new_height = int(height * scale)
                            img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)  # Fastest interpolation
                            cv2.imwrite(temp_path, img, [cv2.IMWRITE_JPEG_QUALITY, 50])  # Lower quality for speed
                            logger.info(f"Image ultra-resized to: {new_width}x{new_height} for maximum speed")
                    img_opt_time = time.time() - img_opt_start_time
                    logger.info(f"Image optimization completed in {img_opt_time:.4f} seconds")
                    
                    # Register student with ultra-fast mode
                    registration_start_time = time.time()
                    success = attendance_system.register_student_with_photo_fast(student_id, name, email, temp_path)
                    registration_time = time.time() - registration_start_time
                    total_route_time = time.time() - route_start_time
                    
                    if success:
                        logger.info(f"Registration completed in {registration_time:.4f} seconds, total route time: {total_route_time:.4f} seconds")
                        flash('Student registered successfully!', 'success')
                    else:
                        logger.error("Registration failed")
                        flash('Failed to register student. Please check if Student ID already exists or no face detected.', 'error')
                    
                except Exception as e:
                    logger.error(f"Registration error: {str(e)}")
                    flash(f'Error processing registration: {str(e)}', 'error')
                finally:
                    # Clean up temp file
                    try:
                        if 'temp_path' in locals() and os.path.exists(temp_path):
                            os.remove(temp_path)
                            logger.info("Temporary file cleaned up")
                    except Exception as cleanup_error:
                        logger.warning(f"Could not clean up temp file: {cleanup_error}")
            else:
                flash('Please select a photo file.', 'error')
        else:
            flash('Please upload a photo for face recognition.', 'error')
        
        return redirect(url_for('register'))
    
    return render_template('register.html')

@app.route('/start_attendance')
def start_attendance():
    """Display the attendance page with camera controls"""
    try:
        # Get statistics for the page
        cursor = attendance_system.db_connection.cursor()
        
        # Get registered students count
        cursor.execute("SELECT COUNT(*) FROM students")
        registered_students = cursor.fetchone()[0]
        
        # Get today's attendance count
        today = date.today()
        cursor.execute("SELECT COUNT(*) FROM attendance WHERE date = ?", (today,))
        today_marked = cursor.fetchone()[0]
        
        logger.info(f"Attendance page accessed - Students: {registered_students}, Today's attendance: {today_marked}")
        
        message = f"Attendance system ready. {registered_students} students registered, {today_marked} marked today."
        
        return render_template('attendance.html', 
                             message=message,
                             registered_students=registered_students,
                             today_marked=today_marked,
                             face_recognition_status=FACE_RECOGNITION_AVAILABLE,
                             db_status=attendance_system.db_connection is not None)
                             
    except Exception as e:
        logger.error(f"Error loading attendance page: {str(e)}")
        return render_template('attendance.html', 
                             message="Error loading attendance data",
                             registered_students=0,
                             today_marked=0,
                             face_recognition_status=False,
                             db_status=False)

@app.route('/start_camera_attendance', methods=['POST'])
def start_camera_attendance():
    """API endpoint to start camera for attendance"""
    try:
        logger.info("Camera start request received")
        
        if camera_stream.start_stream():
            logger.info("Camera stream started successfully")
            return {"status": "success", "message": "Camera started successfully"}, 200
        else:
            logger.error("Failed to start camera stream")
            return {"status": "error", "message": "Failed to start camera. Check permissions."}, 500
            
    except Exception as e:
        logger.error(f"Error starting camera: {str(e)}", exc_info=True)
        return {"status": "error", "message": f"Failed to start camera: {str(e)}"}, 500

@app.route('/stop_camera_attendance', methods=['POST'])
def stop_camera_attendance():
    """API endpoint to stop camera for attendance"""
    try:
        camera_stream.stop_stream()
        logger.info("Camera stream stopped")
        return {"status": "success", "message": "Camera stopped successfully"}, 200
    except Exception as e:
        logger.error(f"Error stopping camera: {str(e)}")
        return {"status": "error", "message": f"Failed to stop camera: {str(e)}"}, 500

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(camera_stream.generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera_status')
def camera_status():
    """Get camera streaming status"""
    return {
        "is_streaming": camera_stream.is_streaming,
        "attendance_count": len(camera_stream.daily_attendance),
        "present_students": list(camera_stream.daily_attendance)
    }

@app.route('/start_registration_camera', methods=['POST'])
def start_registration_camera():
    """API endpoint to start camera for registration preview"""
    try:
        logger.info("Registration camera start request received")
        
        if camera_stream.start_stream():
            logger.info("Registration camera stream started successfully")
            return {"status": "success", "message": "Camera started successfully"}, 200
        else:
            logger.error("Failed to start registration camera stream")
            return {"status": "error", "message": "Failed to start camera. Check permissions."}, 500
            
    except Exception as e:
        logger.error(f"Error starting registration camera: {str(e)}", exc_info=True)
        return {"status": "error", "message": f"Failed to start camera: {str(e)}"}, 500

@app.route('/stop_registration_camera', methods=['POST'])
def stop_registration_camera():
    """API endpoint to stop registration camera"""
    try:
        camera_stream.stop_stream()
        logger.info("Registration camera stream stopped")
        return {"status": "success", "message": "Camera stopped successfully"}, 200
    except Exception as e:
        logger.error(f"Error stopping registration camera: {str(e)}")
        return {"status": "error", "message": f"Failed to stop camera: {str(e)}"}, 500

@app.route('/capture_registration_photo', methods=['POST'])
def capture_registration_photo():
    """API endpoint to capture a photo from the live camera feed"""
    try:
        if not camera_stream.is_streaming or not camera_stream.video_capture:
            return {"status": "error", "message": "Camera is not active"}, 400
        
        with camera_stream.lock:
            ret, frame = camera_stream.video_capture.read()
            
        if not ret or frame is None:
            return {"status": "error", "message": "Failed to capture frame"}, 500
        
        # Save captured frame to temporary file
        timestamp = int(time.time())
        temp_filename = f"captured_photo_{timestamp}.jpg"
        temp_path = os.path.join(tempfile.gettempdir(), temp_filename)
        
        # Optimize captured image
        cv2.imwrite(temp_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        logger.info(f"Photo captured and saved to {temp_path}")
        
        return {
            "status": "success", 
            "message": "Photo captured successfully",
            "temp_path": temp_path,
            "filename": temp_filename,
            "preview_url": url_for('get_captured_photo', filename=temp_filename)
        }, 200
        
    except Exception as e:
        logger.error(f"Error capturing photo: {str(e)}", exc_info=True)
        return {"status": "error", "message": f"Failed to capture photo: {str(e)}"}, 500

@app.route('/get_captured_photo/<filename>')
def get_captured_photo(filename):
    """Serve the temporarily captured photo for preview"""
    try:
        # Security check: ensure filename is safe
        if '..' in filename or filename.startswith('/'):
            return "Invalid filename", 400
            
        temp_path = os.path.join(tempfile.gettempdir(), filename)
        if os.path.exists(temp_path):
            return send_file(temp_path, mimetype='image/jpeg')
        else:
            return "File not found", 404
    except Exception as e:
        logger.error(f"Error serving captured photo: {e}")
        return "Error", 500

@app.route('/records')
def records():
    try:
        records = attendance_system.view_attendance_records()
        logger.info(f"Records route: Retrieved {len(records)} records")
        
        # Calculate today's present count
        today = date.today()
        today_present = sum(1 for record in records if str(record[2]) == str(today) and record[4] == 'Present')
        
        return render_template('records.html', 
                             records=records, 
                             today_present=today_present,
                             today_date=today.strftime('%Y-%m-%d'))
    except Exception as e:
        logger.error(f"Error in records route: {e}")
        flash(f"Error loading records: {e}")
        return render_template('records.html', 
                             records=[], 
                             today_present=0,
                             today_date=date.today().strftime('%Y-%m-%d'))

@app.route('/export')
def export():
    try:
        cursor = attendance_system.db_connection.cursor()
        query = """
            SELECT s.student_id, s.name, a.date, a.time, a.status 
            FROM attendance a 
            JOIN students s ON a.student_id = s.student_id 
            ORDER BY a.date DESC, a.time DESC
        """
        cursor.execute(query)
        records = cursor.fetchall()
        
        if not records:
            flash("No attendance records found!", 'warning')
            return redirect(url_for('records'))
        
        # Create DataFrame and save to temp file
        df = pd.DataFrame(records, columns=['Student ID', 'Name', 'Date', 'Time', 'Status'])
        filename = f"attendance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        temp_path = os.path.join(tempfile.gettempdir(), filename)
        df.to_csv(temp_path, index=False)
        
        return send_file(temp_path, as_attachment=True, download_name=filename)
        
    except Exception as e:
        flash(f"Error exporting data: {e}", 'error')
        return redirect(url_for('records'))

@app.route('/api/stats')
def api_stats():
    """API endpoint for live stats updates"""
    stats = {
        'total_students': 0,
        'today_attendance': 0,
        'attendance_rate': 0
    }
    
    if attendance_system.db_connection:
        try:
            cursor = attendance_system.db_connection.cursor()
            
            cursor.execute("SELECT COUNT(*) FROM students")
            stats['total_students'] = cursor.fetchone()[0]
            
            today = date.today()
            cursor.execute("SELECT COUNT(*) FROM attendance WHERE date = ?", (today,))
            stats['today_attendance'] = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT 
                    COUNT(DISTINCT a.student_id) * 100.0 / COUNT(DISTINCT s.student_id) as rate
                FROM students s 
                LEFT JOIN attendance a ON s.student_id = a.student_id 
                    AND a.date >= date('now', '-7 days')
            """)
            result = cursor.fetchone()
            stats['attendance_rate'] = round(result[0] if result[0] else 0, 1)
            
        except Exception as e:
            print(f"Error getting API stats: {e}")
    
    return stats

if __name__ == "__main__":
    print("Starting Flask application...")
    show_requirements()
    print("Flask app starting on http://127.0.0.1:5000")
    app.run(debug=True, host='127.0.0.1', port=5000)