import numpy as np
from sgp4.api import Satrec, jday
from datetime import datetime, timedelta
import psycopg2
import pandas as pd
import os

def connect_db():
    """Connexion à PostgreSQL"""
    return psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        database=os.getenv('DB_NAME', 'satellites'),
        user=os.getenv('DB_USER', 'admin'),
        password=os.getenv('DB_PASS', 'admin')
    )

def parse_tle_to_position(line1, line2, timestamp):
    """
    Convertir un TLE en position/vitesse cartésienne
    
    Pourquoi : Les TLE sont des paramètres orbitaux encodés.
    L'IA a besoin de positions (x,y,z) et vitesses (vx,vy,vz) pour apprendre.
    """
    try:
        # Créer l'objet satellite avec SGP4
        satellite = Satrec.twoline2rv(line1, line2)
        
        # Convertir timestamp en Julian Date (format pour SGP4)
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        
        jd, fr = jday(timestamp.year, timestamp.month, timestamp.day,
                      timestamp.hour, timestamp.minute, timestamp.second)
        
        # Calculer position et vitesse (en km et km/s)
        error_code, position, velocity = satellite.sgp4(jd, fr)
        
        if error_code != 0:
            return None
        
        return {
            'pos_x': position[0],  # km
            'pos_y': position[1],
            'pos_z': position[2],
            'vel_x': velocity[0],  # km/s
            'vel_y': velocity[1],
            'vel_z': velocity[2],
            'timestamp': timestamp
        }
    except Exception as e:
        print(f"Erreur parsing TLE: {e}")
        return None

def fetch_satellite_data(satellite_name=None, limit=1000):
    """
    Récupérer les données TLE depuis PostgreSQL
    """
    conn = connect_db()
    
    if satellite_name:
        query = """
            SELECT satellite_name, line1, line2, timestamp 
            FROM tle_data 
            WHERE satellite_name = %s 
            ORDER BY timestamp DESC 
            LIMIT %s
        """
        df = pd.read_sql(query, conn, params=(satellite_name, limit))
    else:
        query = """
            SELECT satellite_name, line1, line2, timestamp 
            FROM tle_data 
            ORDER BY timestamp DESC 
            LIMIT %s
        """
        df = pd.read_sql(query, conn, params=(limit,))
    
    conn.close()
    return df

def create_trajectory_dataset(satellite_name, seq_length=10, pred_horizon=5):
    """
    Créer un dataset de séquences pour l'entraînement
    
    Args:
        satellite_name: Nom du satellite à analyser
        seq_length: Nombre d'observations passées (input)
        pred_horizon: Nombre de positions futures à prédire (output)
    
    Pourquoi seq_length=10 :
        - Permet de capturer ~1-2 orbites complètes (selon fréquence des TLE)
        - Assez long pour les patterns, assez court pour l'entraînement
    
    Pourquoi pred_horizon=5 :
        - Prédire 5 positions futures = anticiper les prochaines heures
        - Balance entre utilité et précision
    """
    # Récupérer les TLE
    df = fetch_satellite_data(satellite_name, limit=10000)
    
    if df.empty:
        raise ValueError(f"Aucune donnée pour {satellite_name}")
    
    # Parser tous les TLE en positions
    positions = []
    for _, row in df.iterrows():
        pos = parse_tle_to_position(row['line1'], row['line2'], row['timestamp'])
        if pos:
            positions.append(pos)
    
    if len(positions) < seq_length + pred_horizon:
        raise ValueError(f"Pas assez de données ({len(positions)} < {seq_length + pred_horizon})")
    
    # Convertir en numpy array
    position_data = np.array([[
        p['pos_x'], p['pos_y'], p['pos_z'],
        p['vel_x'], p['vel_y'], p['vel_z']
    ] for p in positions])
    
    # Créer les séquences
    X, Y = [], []
    for i in range(len(position_data) - seq_length - pred_horizon):
        X.append(position_data[i:i+seq_length])           # 10 dernières observations
        Y.append(position_data[i+seq_length:i+seq_length+pred_horizon, :3])  # 5 positions futures (x,y,z seulement)
    
    return np.array(X), np.array(Y), positions

def normalize_data(X, Y):
    """
    Normaliser les données entre -1 et 1
    
    Pourquoi : Les positions sont en milliers de km, les vitesses en km/s.
    Les réseaux de neurones apprennent mieux avec des valeurs normalisées.
    """
    from sklearn.preprocessing import StandardScaler
    
    # Scaler pour X (positions + vitesses)
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X.reshape(-1, X.shape[-1])).reshape(X.shape)
    
    # Scaler pour Y (positions seulement)
    scaler_Y = StandardScaler()
    Y_scaled = scaler_Y.fit_transform(Y.reshape(-1, Y.shape[-1])).reshape(Y.shape)
    
    return X_scaled, Y_scaled, scaler_X, scaler_Y

# Script de test
if __name__ == "__main__":
    # Lister les satellites disponibles
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT satellite_name FROM tle_data LIMIT 10")
    satellites = [row[0] for row in cur.fetchall()]
    conn.close()
    
    print("Satellites disponibles:", satellites)
    
    # Tester avec le premier satellite
    if satellites:
        sat_name = satellites[0]
        print(f"\n🛰️ Test avec {sat_name}")
        
        try:
            X, Y, positions = create_trajectory_dataset(sat_name)
            print(f"✅ Dataset créé:")
            print(f"   - X shape: {X.shape} (séquences d'entrée)")
            print(f"   - Y shape: {Y.shape} (positions à prédire)")
            print(f"   - Nombre total de positions: {len(positions)}")
        except Exception as e:
            print(f"❌ Erreur: {e}")