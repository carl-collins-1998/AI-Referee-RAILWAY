import os
from pathlib import Path
from typing import Dict, Any, List
import tempfile
import shutil
from contextlib import asynccontextmanager
import urllib.request
import sys

from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware

# Global variables - Define these BEFORE they're used
# Better Railway detection
IS_RAILWAY = any([
    "RAILWAY_ENVIRONMENT" in os.environ,
    "RAILWAY_STATIC_URL" in os.environ,
    "RAILWAY_PROJECT_ID" in os.environ,
    os.path.exists("/app")  # Railway uses /app as working directory
])

if IS_RAILWAY:
    # On Railway, use relative path
    MODEL_PATH = "/app/models/best.pt"
    print(f"Running on Railway - Model path: {MODEL_PATH}")
else:
    # Local development path
    MODEL_PATH = r"C:/Users/carlc/Desktop/API  AI REFEREE MODEL/runs/detect/train3/weights/best.pt"
    print(f"Running locally - Model path: {MODEL_PATH}")

MODEL_URL = os.getenv("MODEL_URL")  # Get from environment variable
scorer_instance = None

# Import the fix BEFORE importing basketball_referee
try:
    import yolo_loader_fix
except ImportError:
    print("Warning: yolo_loader_fix not found")

import cv2
from basketball_referee import ImprovedFreeThrowScorer, CVATDatasetConverter, FreeThrowModelTrainer


def download_model():
    """Download model from URL if not present"""
    if os.path.exists(MODEL_PATH):
        print(f"Model already exists at {MODEL_PATH}")
        return True
    
    if not MODEL_URL:
        print("Model not found locally and MODEL_URL not set")
        print("Please set MODEL_URL environment variable in Railway")
        return False
    
    try:
        print(f"Downloading model from {MODEL_URL}")
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        
        # Download with progress
        def download_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = min(downloaded * 100 / total_size, 100)
            print(f"Download progress: {percent:.1f}%", end='\r')
        
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH, reporthook=download_progress)
        print("\nModel downloaded successfully")
        return True
    except Exception as e:
        print(f"Failed to download model: {e}")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for startup and shutdown events.
    """
    # Startup logic
    global scorer_instance

    print("\n" + "=" * 60)
    print("🏀 AI BASKETBALL REFEREE API STARTING")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    print(f"Platform: Railway" if "RAILWAY_ENVIRONMENT" in os.environ else "Local")
    print(f"Model path: {MODEL_PATH}")
    print(f"Model exists: {os.path.exists(MODEL_PATH)}")
    print(f"Model URL set: {'Yes' if MODEL_URL else 'No'}")
    
    # Railway-specific info
    if "RAILWAY_ENVIRONMENT" in os.environ:
        print(f"Railway environment: {os.getenv('RAILWAY_ENVIRONMENT')}")
        print(f"Railway static URL: {os.getenv('RAILWAY_STATIC_URL', 'Not set')}")
    
    # Try to download model if on Railway
    if "RAILWAY_ENVIRONMENT" in os.environ and not os.path.exists(MODEL_PATH):
        if download_model():
            print("✅ Model downloaded from URL")
        else:
            print("⚠️ Model download failed - API will run without scoring functionality")

    if os.path.exists(MODEL_PATH):
        try:
            print("Loading model...")
            scorer_instance = ImprovedFreeThrowScorer(MODEL_PATH)
            print("✅ Model loaded successfully!")
            print(f"Scorer type: {type(scorer_instance)}")
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("❌ Model file not found!")
        print("API will run but scoring functionality will be disabled.")

    print("=" * 60 + "\n")

    # This yield is where the application runs
    yield

    # Shutdown logic
    print("\n" + "=" * 60)
    print("AI BASKETBALL REFEREE API SHUTTING DOWN")
    print("=" * 60)
    print("Goodbye! 👋")


# Create FastAPI app with lifespan
print("Creating FastAPI app...")
app = FastAPI(
    title="AI Basketball Referee API",
    description="Automated basketball free throw scoring using computer vision",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint with status info."""
    railway_url = os.getenv("RAILWAY_STATIC_URL", "")
    
    return {
        "message": "🏀 AI Basketball Referee API",
        "status": "ready" if scorer_instance is not None else "model not loaded",
        "model_loaded": scorer_instance is not None,
        "endpoints": {
            "root": "/",
            "status": "/model_status",
            "score_video": "/score_video/",
            "docs": "/docs",
            "health": "/health"
        },
        "deployment": {
            "platform": "railway" if "RAILWAY_ENVIRONMENT" in os.environ else "local",
            "url": f"https://{railway_url}" if railway_url else "local",
            "environment": os.getenv("RAILWAY_ENVIRONMENT", "local")
        },
        "instructions": "Set MODEL_URL environment variable to enable scoring" if not scorer_instance else None
    }


@app.get("/model_status")
async def model_status():
    """Detailed model status."""
    return {
        "loaded": scorer_instance is not None,
        "path": MODEL_PATH,
        "exists": os.path.exists(MODEL_PATH),
        "size_mb": round(os.path.getsize(MODEL_PATH) / 1024 / 1024, 2) if os.path.exists(MODEL_PATH) else 0,
        "scorer_type": str(type(scorer_instance)) if scorer_instance else None,
        "model_url": "Set" if MODEL_URL else "Not set",
        "environment": {
            "platform": "railway" if "RAILWAY_ENVIRONMENT" in os.environ else "local",
            "python_version": sys.version.split()[0]
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_ready": scorer_instance is not None
    }


@app.get("/debug")
async def debug():
    """Debug endpoint to check environment"""
    import socket
    
    return {
        "environment_variables": {
            "PORT": os.getenv("PORT"),
            "MODEL_URL": "Set" if os.getenv("MODEL_URL") else "Not set",
            "RAILWAY_ENVIRONMENT": os.getenv("RAILWAY_ENVIRONMENT"),
            "RAILWAY_STATIC_URL": os.getenv("RAILWAY_STATIC_URL"),
            "RAILWAY_PROJECT_ID": os.getenv("RAILWAY_PROJECT_ID"),
        },
        "paths": {
            "working_dir": os.getcwd(),
            "model_path": MODEL_PATH,
            "model_exists": os.path.exists(MODEL_PATH),
            "is_railway": IS_RAILWAY,
            "directory_contents": os.listdir(".") if os.path.exists(".") else [],
            "app_directory": os.listdir("/app") if os.path.exists("/app") else "Not found"
        },
        "server": {
            "hostname": socket.gethostname(),
            "port": int(os.getenv("PORT", 8000))
        }
    }


@app.post("/score_video/")
async def score_video(video_file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Analyzes an uploaded video to detect and score free throws.
    
    Returns:
        - made_shots: Number of successful free throws
        - missed_shots: Number of missed free throws
        - total_attempts: Total shot attempts detected
        - accuracy_percentage: Shooting accuracy
        - frames_processed: Number of video frames analyzed
    """
    global scorer_instance

    print(f"\n=== Processing video: {video_file.filename} ===")

    if scorer_instance is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please set MODEL_URL environment variable."
        )

    # Check file size
    content = await video_file.read()
    size_mb = len(content) / (1024 * 1024)
    
    if size_mb > 100:  # 100MB limit
        raise HTTPException(
            status_code=413,
            detail=f"Video too large ({size_mb:.1f}MB). Maximum size is 100MB."
        )

    # Save and process video
    with tempfile.TemporaryDirectory() as temp_dir:
        video_path = Path(temp_dir) / video_file.filename
        
        with open(video_path, "wb") as f:
            f.write(content)
        print(f"Video saved: {size_mb:.2f} MB")

        # Reset scorer
        scorer_instance.made_shots = 0
        scorer_instance.missed_shots = 0
        scorer_instance.shot_attempts = 0
        scorer_instance.shot_tracker.reset()

        # Process video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")

        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Video info: {total_frames} frames, {fps:.1f} FPS, {total_frames/fps:.1f} seconds")

        # Process every frame (or skip frames for faster processing)
        frame_skip = 1  # Process every frame
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_skip == 0:
                # Run detection
                detections = scorer_instance.detect_objects(frame)
                hoop_info = scorer_instance.update_hoop_position(detections)
                ball_info = scorer_instance.find_ball(detections)
                player_bboxes = scorer_instance.find_players(detections)

                # Update shot tracking
                old_phase = scorer_instance.shot_tracker.shot_phase
                result = scorer_instance.shot_tracker.update(ball_info, hoop_info, player_bboxes, False)

                # Count attempts
                if old_phase == 'idle' and scorer_instance.shot_tracker.shot_phase == 'rising':
                    scorer_instance.shot_attempts += 1
                    print(f"Shot attempt #{scorer_instance.shot_attempts} at frame {frame_count}")

                # Count results
                if result == 'score':
                    scorer_instance.made_shots += 1
                    print(f"SCORE! Total: {scorer_instance.made_shots}")
                    scorer_instance.shot_tracker.reset()
                elif result == 'miss':
                    scorer_instance.missed_shots += 1
                    print(f"MISS! Total: {scorer_instance.missed_shots}")
                    scorer_instance.shot_tracker.reset()

            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"Progress: {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)")

        cap.release()

        print(f"Processing complete. Frames: {frame_count}")

        # Calculate accuracy
        accuracy = 0.0
        if scorer_instance.shot_attempts > 0:
            accuracy = round((scorer_instance.made_shots / scorer_instance.shot_attempts) * 100, 1)

        return {
            "made_shots": scorer_instance.made_shots,
            "missed_shots": scorer_instance.missed_shots,
            "total_attempts": scorer_instance.shot_attempts,
            "frames_processed": frame_count,
            "accuracy_percentage": accuracy,
            "video_info": {
                "filename": video_file.filename,
                "size_mb": round(size_mb, 2),
                "duration_seconds": round(total_frames / fps, 1) if fps > 0 else 0,
                "fps": round(fps, 1),
                "total_frames": total_frames
            }
        }
@app.get("/file_check")
async def file_check():
    with open(__file__, 'r') as f:
        first_lines = f.readlines()[:30]
    return {"first_30_lines": first_lines}

if __name__ == "__main__":
    import uvicorn
    
    # Railway provides PORT env var - MUST use this!
    port = int(os.getenv("PORT", 8000))
    host = "0.0.0.0"
    
    print(f"Starting server on {host}:{port}")
    print(f"Railway URL will be: https://{os.getenv('RAILWAY_STATIC_URL', 'your-app.railway.app')}")
    
    uvicorn.run(app, host=host, port=port)