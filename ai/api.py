
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import psycopg2
from typing import List, Optional, Tuple
import os
import numpy as np
from datetime import datetime, timedelta
import torch
import joblib
from pathlib import Path
import json
from itertools import combinations

from model import TrajectoryGRU
from preprocessing import parse_tle_to_position

app = FastAPI(title="Satellite AI Service - PyTorch with Collision Detection", version="2.0.0")

# Database configuration
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "postgres-db"),
    "database": os.getenv("DB_NAME", "satellites"),
    "user": os.getenv("DB_USER", "admin"),
    "password": os.getenv("DB_PASS", "admin")
}

MODELS_DIR = Path("/app/models")

# Collision detection thresholds
COLLISION_THRESHOLDS = {
    "critical": 1.0,    # < 1 km
    "high": 5.0,        # < 5 km
    "medium": 10.0,     # < 10 km
    "low": 25.0         # < 25 km
}

# Pydantic models
class PredictionRequest(BaseModel):
    satellite_name: str
    sequence_length: int = 10

class PredictionResponse(BaseModel):
    satellite_name: str
    predictions: List[dict]
    model_info: dict
    generated_at: str

class CollisionCheckRequest(BaseModel):
    satellite1: str
    satellite2: str
    time_horizon_steps: int = 5
    sequence_length: int = 10

class CollisionAnalysisRequest(BaseModel):
    satellites: Optional[List[str]] = None  # If None, check all satellites
    time_horizon_steps: int = 5
    sequence_length: int = 10
    min_risk_level: str = "low"  # "critical", "high", "medium", "low"

class CollisionEvent(BaseModel):
    satellite1: str
    satellite2: str
    time_step: int
    distance_km: float
    risk_level: str
    relative_velocity_km_s: Optional[float] = None
    predicted_time: str
    positions: dict

class CollisionCheckResponse(BaseModel):
    satellite1: str
    satellite2: str
    collision_events: List[CollisionEvent]
    min_distance_km: float
    analysis_timestamp: str

class CollisionAnalysisResponse(BaseModel):
    total_satellites_analyzed: int
    total_pairs_checked: int
    collision_events: List[CollisionEvent]
    summary: dict
    analysis_timestamp: str

class HealthResponse(BaseModel):
    status: str
    database: str
    models_available: List[str]
    device: str

class TrainingStatus(BaseModel):
    satellite_name: str
    status: str
    metadata: Optional[dict] = None

# Database connection
def get_db_connection():
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")

# Global variables
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loaded_models = {}

def load_satellite_model(satellite_name: str):
    """Load PyTorch model, scalers, and metadata for a satellite"""
    model_dir = MODELS_DIR / satellite_name.replace(' ', '_')
    
    if not model_dir.exists():
        return None
    
    if satellite_name in loaded_models:
        return loaded_models[satellite_name]
    
    try:
        with open(model_dir / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        model = TrajectoryGRU(
            input_size=6,
            hidden_size=metadata['hyperparameters']['hidden_size'],
            num_layers=metadata['hyperparameters']['num_layers'],
            output_horizon=metadata['hyperparameters']['pred_horizon'],
            dropout=metadata['hyperparameters']['dropout']
        ).to(device)
        
        checkpoint = torch.load(model_dir / 'best_model.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        scaler_X = joblib.load(model_dir / 'scaler_X.pkl')
        scaler_Y = joblib.load(model_dir / 'scaler_Y.pkl')
        
        loaded_models[satellite_name] = {
            'model': model,
            'scaler_X': scaler_X,
            'scaler_Y': scaler_Y,
            'metadata': metadata
        }
        
        print(f"✅ Loaded model for {satellite_name}")
        return loaded_models[satellite_name]
        
    except Exception as e:
        print(f"❌ Failed to load model for {satellite_name}: {e}")
        return None

def get_available_models():
    """List all trained models"""
    if not MODELS_DIR.exists():
        return []
    
    models = []
    for model_dir in MODELS_DIR.iterdir():
        if model_dir.is_dir() and (model_dir / 'best_model.pth').exists():
            models.append(model_dir.name.replace('_', ' '))
    
    return models

def predict_satellite_trajectory(satellite_name: str, sequence_length: int, time_horizon_steps: int) -> Optional[np.ndarray]:
    """
    Predict trajectory for a satellite
    Returns: numpy array of shape (time_horizon_steps, 6) with [x, y, z, vx, vy, vz]
    """
    model_data = load_satellite_model(satellite_name)
    if model_data is None:
        print(f"No model found for {satellite_name}")
        return None
    
    model = model_data['model']
    scaler_X = model_data['scaler_X']
    scaler_Y = model_data['scaler_Y']
    metadata = model_data['metadata']
    
    # Get the model's actual prediction horizon
    model_pred_horizon = metadata['hyperparameters']['pred_horizon']
    actual_steps = min(time_horizon_steps, model_pred_horizon)
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT line1, line2, timestamp 
            FROM tle_data 
            WHERE satellite_name = %s 
            ORDER BY timestamp DESC 
            LIMIT %s
        """, (satellite_name, sequence_length))
        
        tle_records = cursor.fetchall()
        
        if len(tle_records) < sequence_length:
            print(f"Not enough TLE data for {satellite_name}: {len(tle_records)}/{sequence_length}")
            return None
        
        positions = []
        velocities = []
        for line1, line2, timestamp in reversed(tle_records):
            pos = parse_tle_to_position(line1, line2, timestamp)
            if pos:
                positions.append([pos['pos_x'], pos['pos_y'], pos['pos_z']])
                velocities.append([pos['vel_x'], pos['vel_y'], pos['vel_z']])
        
        if len(positions) < sequence_length:
            print(f"Failed to parse enough positions for {satellite_name}")
            return None
        
        # Prepare input with positions and velocities
        X_input = []
        for i in range(sequence_length):
            X_input.append(positions[i] + velocities[i])
        
        X = np.array([X_input])
        X_scaled = scaler_X.transform(X.reshape(-1, 6)).reshape(X.shape)
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        
        with torch.no_grad():
            Y_pred_scaled = model(X_tensor)
            Y_pred_scaled = Y_pred_scaled.cpu().numpy()
        
        # Inverse transform predictions (positions only)
        Y_pred = scaler_Y.inverse_transform(Y_pred_scaled.reshape(-1, 3)).reshape(Y_pred_scaled.shape)
        
        # Build output with positions and estimated velocities
        predictions_with_velocity = []
        last_pos = np.array(positions[-1])
        
        for i in range(actual_steps):
            if i < len(Y_pred[0]):
                pred_pos = Y_pred[0][i]
                
                # Estimate velocity
                if i < len(Y_pred[0]) - 1:
                    next_pos = Y_pred[0][i + 1]
                    velocity = (next_pos - pred_pos) / 60.0  # Assuming 60 second intervals
                elif i > 0:
                    prev_pos = Y_pred[0][i - 1]
                    velocity = (pred_pos - prev_pos) / 60.0
                else:
                    velocity = (pred_pos - last_pos) / 60.0
                
                predictions_with_velocity.append([
                    float(pred_pos[0]), float(pred_pos[1]), float(pred_pos[2]),
                    float(velocity[0]), float(velocity[1]), float(velocity[2])
                ])
        
        if not predictions_with_velocity:
            print(f"No predictions generated for {satellite_name}")
            return None
        
        return np.array(predictions_with_velocity)
        
    except Exception as e:
        print(f"Error predicting trajectory for {satellite_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        cursor.close()
        conn.close()

def calculate_collision_risk(distance_km: float) -> str:
    """Determine collision risk level based on distance"""
    if distance_km < COLLISION_THRESHOLDS["critical"]:
        return "critical"
    elif distance_km < COLLISION_THRESHOLDS["high"]:
        return "high"
    elif distance_km < COLLISION_THRESHOLDS["medium"]:
        return "medium"
    elif distance_km < COLLISION_THRESHOLDS["low"]:
        return "low"
    else:
        return "none"

def check_collision_between_satellites(
    satellite1: str, 
    satellite2: str, 
    time_horizon_steps: int,
    sequence_length: int
) -> Tuple[List[CollisionEvent], float]:
    """
    Check for potential collisions between two satellites
    Returns: (list of collision events, minimum distance)
    """
    # Get trajectory predictions for both satellites
    traj1 = predict_satellite_trajectory(satellite1, sequence_length, time_horizon_steps)
    traj2 = predict_satellite_trajectory(satellite2, sequence_length, time_horizon_steps)
    
    if traj1 is None or traj2 is None:
        raise HTTPException(
            status_code=404,
            detail=f"Could not predict trajectories for one or both satellites"
        )
    
    collision_events = []
    min_distance = float('inf')
    base_time = datetime.utcnow()
    
    # Check each time step
    for step in range(min(len(traj1), len(traj2))):
        pos1 = traj1[step][:3]
        pos2 = traj2[step][:3]
        vel1 = traj1[step][3:6]
        vel2 = traj2[step][3:6]
        
        # Calculate distance between satellites
        distance = np.linalg.norm(pos1 - pos2)
        min_distance = min(min_distance, distance)
        
        risk_level = calculate_collision_risk(distance)
        
        if risk_level != "none":
            # Calculate relative velocity
            rel_velocity = np.linalg.norm(vel1 - vel2)
            
            # Estimate time of this prediction (assuming 60 second intervals)
            predicted_time = base_time + timedelta(seconds=60 * (step + 1))
            
            collision_events.append(CollisionEvent(
                satellite1=satellite1,
                satellite2=satellite2,
                time_step=step + 1,
                distance_km=float(distance),
                risk_level=risk_level,
                relative_velocity_km_s=float(rel_velocity),
                predicted_time=predicted_time.isoformat(),
                positions={
                    satellite1: {
                        "x": float(pos1[0]),
                        "y": float(pos1[1]),
                        "z": float(pos1[2])
                    },
                    satellite2: {
                        "x": float(pos2[0]),
                        "y": float(pos2[1]),
                        "z": float(pos2[2])
                    }
                }
            ))
    
    return collision_events, min_distance

@app.on_event("startup")
async def startup_event():
    print(f"🚀 AI Service starting on device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    models = get_available_models()
    print(f"📦 Available models: {len(models)}")
    for model_name in models:
        print(f"   - {model_name}")
    
    print("✅ AI Service started successfully with collision detection")

@app.get("/")
async def root():
    return {
        "service": "Satellite AI Service with Collision Detection",
        "version": "2.0.0",
        "status": "running",
        "features": [
            "Trajectory prediction",
            "Collision detection",
            "Multi-satellite analysis"
        ]
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    db_status = "connected"
    try:
        conn = get_db_connection()
        conn.close()
    except:
        db_status = "disconnected"
    
    models = get_available_models()
    
    return HealthResponse(
        status="healthy" if db_status == "connected" else "degraded",
        database=db_status,
        models_available=models,
        device=str(device)
    )

@app.get("/models", response_model=List[TrainingStatus])
async def list_models():
    models = get_available_models()
    result = []
    
    for model_name in models:
        model_dir = MODELS_DIR / model_name.replace(' ', '_')
        metadata_path = model_dir / 'metadata.json'
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            result.append(TrainingStatus(
                satellite_name=model_name,
                status="trained",
                metadata=metadata
            ))
        else:
            result.append(TrainingStatus(
                satellite_name=model_name,
                status="incomplete",
                metadata=None
            ))
    
    return result

@app.get("/satellites", response_model=List[dict])
async def get_satellites():
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT satellite_name, COUNT(*) as tle_count
            FROM tle_data
            GROUP BY satellite_name
            ORDER BY tle_count DESC
            LIMIT 100
        """)
        
        satellites = []
        available_models = get_available_models()
        
        for row in cursor.fetchall():
            satellites.append({
                "satellite_name": row[0],
                "tle_count": row[1],
                "has_model": row[0] in available_models
            })
        
        return satellites
    
    finally:
        cursor.close()
        conn.close()

@app.post("/predict", response_model=PredictionResponse)
async def predict_trajectory(request: PredictionRequest):
    model_data = load_satellite_model(request.satellite_name)
    if model_data is None:
        raise HTTPException(
            status_code=404, 
            detail=f"No trained model found for {request.satellite_name}"
        )
    
    model = model_data['model']
    scaler_X = model_data['scaler_X']
    scaler_Y = model_data['scaler_Y']
    metadata = model_data['metadata']
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT line1, line2, timestamp 
            FROM tle_data 
            WHERE satellite_name = %s 
            ORDER BY timestamp DESC 
            LIMIT %s
        """, (request.satellite_name, request.sequence_length))
        
        tle_records = cursor.fetchall()
        
        if len(tle_records) < request.sequence_length:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough TLE data. Need {request.sequence_length}, found {len(tle_records)}"
            )
        
        positions = []
        for line1, line2, timestamp in reversed(tle_records):
            pos = parse_tle_to_position(line1, line2, timestamp)
            if pos:
                positions.append([
                    pos['pos_x'], pos['pos_y'], pos['pos_z'],
                    pos['vel_x'], pos['vel_y'], pos['vel_z']
                ])
        
        if len(positions) < request.sequence_length:
            raise HTTPException(
                status_code=400,
                detail="Failed to parse enough TLE records"
            )
        
        X = np.array([positions[:request.sequence_length]])
        X_scaled = scaler_X.transform(X.reshape(-1, 6)).reshape(X.shape)
        X_tensor = torch.FloatTensor(X_scaled).to(device)
        
        with torch.no_grad():
            Y_pred_scaled = model(X_tensor)
            Y_pred_scaled = Y_pred_scaled.cpu().numpy()
        
        Y_pred = scaler_Y.inverse_transform(Y_pred_scaled.reshape(-1, 3)).reshape(Y_pred_scaled.shape)
        
        predictions = []
        for i, pred in enumerate(Y_pred[0]):
            predictions.append({
                "step": i + 1,
                "position_x_km": float(pred[0]),
                "position_y_km": float(pred[1]),
                "position_z_km": float(pred[2]),
                "confidence": "high" if i < 3 else "medium" if i < 4 else "low"
            })
        
        return PredictionResponse(
            satellite_name=request.satellite_name,
            predictions=predictions,
            model_info={
                "framework": "PyTorch",
                "device": str(device),
                "trained_on": metadata.get('train_date'),
                "test_rmse": metadata.get('test_rmse'),
                "parameters": metadata.get('model_params')
            },
            generated_at=datetime.utcnow().isoformat()
        )
        
    finally:
        cursor.close()
        conn.close()

@app.post("/collision/check", response_model=CollisionCheckResponse)
async def check_collision(request: CollisionCheckRequest):
    """
    Check for potential collisions between two specific satellites
    """
    # Validate satellites exist and have models
    available_models = get_available_models()
    
    if request.satellite1 not in available_models:
        raise HTTPException(
            status_code=404,
            detail=f"No trained model found for {request.satellite1}. Available models: {', '.join(available_models[:5])}"
        )
    
    if request.satellite2 not in available_models:
        raise HTTPException(
            status_code=404,
            detail=f"No trained model found for {request.satellite2}. Available models: {', '.join(available_models[:5])}"
        )
    
    try:
        collision_events, min_distance = check_collision_between_satellites(
            request.satellite1,
            request.satellite2,
            request.time_horizon_steps,
            request.sequence_length
        )
        
        return CollisionCheckResponse(
            satellite1=request.satellite1,
            satellite2=request.satellite2,
            collision_events=collision_events,
            min_distance_km=float(min_distance),
            analysis_timestamp=datetime.utcnow().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        error_details = f"{str(e)}\n\nTraceback:\n{traceback.format_exc()}"
        raise HTTPException(status_code=500, detail=error_details)

@app.post("/collision/analyze", response_model=CollisionAnalysisResponse)
async def analyze_collisions(request: CollisionAnalysisRequest):
    """
    Analyze potential collisions across multiple satellites
    If no satellites specified, check all satellites with trained models
    """
    # Get satellites to analyze
    if request.satellites is None or len(request.satellites) == 0:
        satellites = get_available_models()
    else:
        satellites = request.satellites
    
    if len(satellites) < 2:
        raise HTTPException(
            status_code=400,
            detail="Need at least 2 satellites to check for collisions"
        )
    
    # Generate all pairs
    satellite_pairs = list(combinations(satellites, 2))
    
    all_collision_events = []
    risk_summary = {
        "critical": 0,
        "high": 0,
        "medium": 0,
        "low": 0
    }
    
    # Check each pair
    for sat1, sat2 in satellite_pairs:
        try:
            collision_events, _ = check_collision_between_satellites(
                sat1, sat2,
                request.time_horizon_steps,
                request.sequence_length
            )
            
            # Filter by minimum risk level
            risk_levels = ["critical", "high", "medium", "low"]
            min_risk_index = risk_levels.index(request.min_risk_level)
            
            for event in collision_events:
                event_risk_index = risk_levels.index(event.risk_level)
                if event_risk_index <= min_risk_index:
                    all_collision_events.append(event)
                    risk_summary[event.risk_level] += 1
        
        except Exception as e:
            print(f"Warning: Could not check collision for {sat1} - {sat2}: {e}")
            continue
    
    # Sort by risk level and distance
    risk_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    all_collision_events.sort(
        key=lambda x: (risk_order[x.risk_level], x.distance_km)
    )
    
    return CollisionAnalysisResponse(
        total_satellites_analyzed=len(satellites),
        total_pairs_checked=len(satellite_pairs),
        collision_events=all_collision_events,
        summary={
            "total_events": len(all_collision_events),
            "by_risk_level": risk_summary,
            "most_critical": all_collision_events[0].model_dump() if all_collision_events else None
        },
        analysis_timestamp=datetime.utcnow().isoformat()
    )

@app.get("/collision/thresholds")
async def get_collision_thresholds():
    """
    Get the current collision detection thresholds
    """
    return {
        "thresholds_km": COLLISION_THRESHOLDS,
        "description": {
            "critical": "Immediate danger - collision imminent",
            "high": "High risk - close approach expected",
            "medium": "Medium risk - monitor closely",
            "low": "Low risk - potential concern"
        }
    }

@app.get("/collision/demo")
async def collision_demo():
    """
    Demo endpoint showing simulated collision detection with guaranteed close approaches
    """
    base_time = datetime.utcnow()
    collision_events = []
    
    # Simulate two satellites on collision course
    # Satellite 1: Stable circular orbit
    # Satellite 2: Converging trajectory (simulating debris or malfunctioning satellite)
    
    for step in range(5):
        # Satellite 1: Circular orbit at 6800 km radius
        angle1 = step * 0.15
        sat1_pos = np.array([
            6800 * np.cos(angle1),
            6800 * np.sin(angle1),
            400.0
        ])
        
        # Satellite 2: Converging orbit - starts far, moves closer
        # Simulate a close approach scenario
        angle2 = angle1 + 0.03
        radius_offset = 20 - (step * 3.5)  # Starts 20km away, converges to 6.5km
        sat2_pos = np.array([
            (6800 + radius_offset) * np.cos(angle2),
            (6800 + radius_offset) * np.sin(angle2),
            400.0 + (2 - step * 0.4)  # Slightly different altitude, converging
        ])
        
        # Calculate actual distance between satellites
        distance = np.linalg.norm(sat1_pos - sat2_pos)
        risk_level = calculate_collision_risk(distance)
        
        # Calculate relative velocity (km/s)
        if step > 0:
            prev_distance = np.linalg.norm(sat1_positions[-1] - sat2_positions[-1])
            rel_velocity = abs(prev_distance - distance) / 60.0  # Assuming 60 sec intervals
        else:
            rel_velocity = 0.12
        
        # Store positions for next iteration
        if step == 0:
            sat1_positions = [sat1_pos]
            sat2_positions = [sat2_pos]
        else:
            sat1_positions.append(sat1_pos)
            sat2_positions.append(sat2_pos)
        
        predicted_time = base_time + timedelta(seconds=60 * (step + 1))
        
        # Add event regardless of risk level for demo purposes
        collision_events.append({
            "step": step + 1,
            "distance_km": round(float(distance), 3),
            "risk_level": risk_level,
            "relative_velocity_km_s": round(float(rel_velocity), 4),
            "predicted_time": predicted_time.isoformat(),
            "time_to_event": f"{(step + 1) * 60} seconds",
            "satellite1_position": {
                "x": round(float(sat1_pos[0]), 2),
                "y": round(float(sat1_pos[1]), 2),
                "z": round(float(sat1_pos[2]), 2)
            },
            "satellite2_position": {
                "x": round(float(sat2_pos[0]), 2),
                "y": round(float(sat2_pos[1]), 2),
                "z": round(float(sat2_pos[2]), 2)
            }
        })
    
    # Find closest approach
    min_distance_event = min(collision_events, key=lambda x: x["distance_km"])
    
    return {
        "demo": True,
        "description": "Simulated close approach scenario between two satellites",
        "scenario": "Satellite-2 is on a converging trajectory with Satellite-1",
        "satellite1": "Demo-Sat-1 (ISS-like orbit)",
        "satellite2": "Demo-Sat-2 (Converging debris)",
        "collision_events": collision_events,
        "summary": {
            "total_events": len(collision_events),
            "closest_approach": {
                "distance_km": min_distance_event["distance_km"],
                "at_step": min_distance_event["step"],
                "risk_level": min_distance_event["risk_level"],
                "time": min_distance_event["predicted_time"]
            },
            "risk_levels": {
                "critical": len([e for e in collision_events if e["risk_level"] == "critical"]),
                "high": len([e for e in collision_events if e["risk_level"] == "high"]),
                "medium": len([e for e in collision_events if e["risk_level"] == "medium"]),
                "low": len([e for e in collision_events if e["risk_level"] == "low"])
            }
        },
        "note": "This is simulated data demonstrating collision detection. Train more satellite models to analyze real trajectories.",
        "next_steps": "Run: docker-compose exec ai-service python train.py (repeat 3-4 times)"
    }

@app.get("/collision/debug")
async def collision_debug_info():
    """
    Get debug information for collision detection
    """
    available_models = get_available_models()
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        # Get satellite data counts
        cursor.execute("""
            SELECT satellite_name, COUNT(*) as tle_count
            FROM tle_data
            GROUP BY satellite_name
            HAVING COUNT(*) >= 10
            ORDER BY tle_count DESC
            LIMIT 10
        """)
        
        satellites_with_data = []
        for row in cursor.fetchall():
            sat_name = row[0]
            satellites_with_data.append({
                "name": sat_name,
                "tle_count": row[1],
                "has_model": sat_name in available_models,
                "can_predict": sat_name in available_models and row[1] >= 10
            })
        
        return {
            "total_models": len(available_models),
            "available_models": available_models,
            "satellites_with_sufficient_data": satellites_with_data,
            "collision_ready_pairs": len([s for s in satellites_with_data if s["can_predict"]]),
            "requirements": {
                "min_tle_records": 10,
                "min_satellites_for_collision": 2
            },
            "next_steps": {
                "action": "Train more models",
                "command": "docker-compose exec ai-service python train.py",
                "repeat": "Run 3-4 times to train multiple satellites"
            }
        }
    
    finally:
        cursor.close()
        conn.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)