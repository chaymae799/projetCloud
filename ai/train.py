"""
Script d'entraînement PyTorch - Optimisé pour petits datasets
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
from datetime import datetime
import json
from pathlib import Path

from model import TrajectoryGRU
from preprocessing import create_trajectory_dataset, normalize_data, connect_db

def train_model(satellite_name, epochs=50, batch_size=8, learning_rate=0.001):
    """
    Entraîner le modèle GRU pour un satellite
    Optimisé pour petits datasets
    """
    
    print(f"🚀 Entraînement PyTorch pour {satellite_name}")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Créer le dataset avec paramètres réduits pour petits datasets
    print("\n📊 Création du dataset...")
    try:
        # Réduire seq_length et pred_horizon pour avoir plus de séquences
        X, Y, positions = create_trajectory_dataset(
            satellite_name, 
            seq_length=5,    # Réduit de 10 à 5
            pred_horizon=3   # Réduit de 5 à 3
        )
        print(f"   ✅ {len(X)} séquences créées")
        print(f"   ✅ {len(positions)} positions totales")
        
        # Vérifier qu'on a assez de données
        if len(X) < 3:
            raise ValueError(f"Pas assez de séquences ({len(X)}). Minimum requis: 3")
            
    except Exception as e:
        print(f"   ❌ Erreur: {e}")
        return None
    
    # Normaliser
    print("\n🔢 Normalisation des données...")
    X_scaled, Y_scaled, scaler_X, scaler_Y = normalize_data(X, Y)
    print(f"   ✅ Données normalisées")
    
    # Split adaptatif basé sur la taille du dataset
    total_samples = len(X_scaled)
    
    if total_samples >= 10:
        # Dataset normal: 70/15/15
        train_size = int(0.7 * total_samples)
        val_size = int(0.15 * total_samples)
    elif total_samples >= 5:
        # Petit dataset: 60/20/20
        train_size = int(0.6 * total_samples)
        val_size = int(0.2 * total_samples)
    else:
        # Très petit dataset: 60/20/20 avec minimum 1 pour chaque
        train_size = max(1, int(0.6 * total_samples))
        val_size = max(1, int(0.2 * total_samples))
    
    # S'assurer qu'il reste au moins 1 échantillon pour le test
    if train_size + val_size >= total_samples:
        val_size = max(1, total_samples - train_size - 1)
    
    X_train = torch.FloatTensor(X_scaled[:train_size])
    Y_train = torch.FloatTensor(Y_scaled[:train_size])
    
    X_val = torch.FloatTensor(X_scaled[train_size:train_size+val_size])
    Y_val = torch.FloatTensor(Y_scaled[train_size:train_size+val_size])
    
    X_test = torch.FloatTensor(X_scaled[train_size+val_size:])
    Y_test = torch.FloatTensor(Y_scaled[train_size+val_size:])
    
    print(f"\n📂 Split des données:")
    print(f"   - Total:  {total_samples} séquences")
    print(f"   - Train:  {len(X_train)} ({len(X_train)/total_samples*100:.1f}%)")
    print(f"   - Val:    {len(X_val)} ({len(X_val)/total_samples*100:.1f}%)")
    print(f"   - Test:   {len(X_test)} ({len(X_test)/total_samples*100:.1f}%)")
    
    # Vérifier qu'on a des données dans chaque split
    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        print("\n❌ Erreur: Un des splits est vide!")
        print("   Collectez plus de données TLE pour ce satellite.")
        return None
    
    # DataLoaders avec batch_size adaptatif
    actual_batch_size = min(batch_size, len(X_train))
    
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=actual_batch_size, 
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=len(X_val),  # Tout le validation set en un batch
        shuffle=False
    )
    
    # Modèle plus léger pour petits datasets
    print("\n🧠 Construction du modèle GRU...")
    model = TrajectoryGRU(
        input_size=6,
        hidden_size=64,   # Réduit de 128 à 64
        num_layers=1,     # Réduit de 2 à 1
        output_horizon=3, # Match pred_horizon
        dropout=0.1       # Réduit de 0.2 à 0.1
    ).to(device)
    
    print(f"   ✅ Modèle créé avec {model.count_parameters():,} paramètres")
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5,
        patience=5,
        verbose=True,
        min_lr=1e-6
    )
    
    print("\n🏋️  Début de l'entraînement...")
    print("="*60)
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 10  # Réduit pour petits datasets
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rates': []
    }
    
    model_dir = Path(f"/app/models/{satellite_name.replace(' ', '_')}")
    model_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(epochs):
        # TRAINING
        model.train()
        train_loss = 0
        num_batches = 0
        
        for batch_X, batch_Y in train_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            num_batches += 1
        
        train_loss = train_loss / num_batches if num_batches > 0 else float('inf')
        
        # VALIDATION
        model.eval()
        val_loss = 0
        num_val_batches = 0
        
        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_Y)
                val_loss += loss.item()
                num_val_batches += 1
        
        val_loss = val_loss / num_val_batches if num_val_batches > 0 else float('inf')
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['learning_rates'].append(optimizer.param_groups[0]['lr'])
        
        scheduler.step(val_loss)
        
        # EARLY STOPPING
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, model_dir / 'best_model.pth')
            
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train: {train_loss:.6f} | "
                  f"Val: {val_loss:.6f} | "
                  f"✅ Saved")
        else:
            patience_counter += 1
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train: {train_loss:.6f} | "
                  f"Val: {val_loss:.6f}")
            
            if patience_counter >= patience:
                print(f"\n⏹️  Early stopping après {epoch+1} epochs")
                break
    
    # Charger le meilleur modèle
    print("\n📥 Chargement du meilleur modèle...")
    checkpoint = torch.load(model_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Évaluation finale
    print("\n📈 Évaluation finale sur test set:")
    model.eval()
    
    if len(X_test) > 0:
        with torch.no_grad():
            X_test_gpu = X_test.to(device)
            Y_test_gpu = Y_test.to(device)
            predictions = model(X_test_gpu)
            
            test_loss = criterion(predictions, Y_test_gpu)
            mae = torch.mean(torch.abs(predictions - Y_test_gpu))
            rmse = torch.sqrt(test_loss)
        
        print(f"   - MSE:  {test_loss.item():.6f}")
        print(f"   - MAE:  {mae.item():.6f}")
        print(f"   - RMSE: {rmse.item():.6f}")
    else:
        test_loss = mae = rmse = torch.tensor(0.0)
        print("   ⚠️  Pas de données de test")
    
    # Sauvegarder les scalers
    import joblib
    joblib.dump(scaler_X, model_dir / 'scaler_X.pkl')
    joblib.dump(scaler_Y, model_dir / 'scaler_Y.pkl')
    print(f"\n💾 Scalers sauvegardés")
    
    # Métadonnées
    metadata = {
        'satellite_name': satellite_name,
        'train_date': datetime.now().isoformat(),
        'framework': 'PyTorch',
        'device': str(device),
        'model_params': model.count_parameters(),
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test),
        'test_mse': float(test_loss.item()),
        'test_mae': float(mae.item()),
        'test_rmse': float(rmse.item()),
        'best_epoch': checkpoint['epoch'] + 1,
        'total_epochs': epoch + 1,
        'hyperparameters': {
            'seq_length': 5,
            'pred_horizon': 3,
            'batch_size': actual_batch_size,
            'learning_rate': learning_rate,
            'hidden_size': 64,
            'num_layers': 1,
            'dropout': 0.1
        }
    }
    
    with open(model_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    np.save(model_dir / 'history.npy', history)
    
    print(f"\n✅ Entraînement terminé!")
    print(f"📁 Modèle sauvegardé dans: {model_dir}")
    print("="*60)
    
    return model, history, metadata


if __name__ == "__main__":
    print("🔍 Recherche du meilleur satellite pour l'entraînement...\n")
    
    try:
        conn = connect_db()
        cur = conn.cursor()
        cur.execute("""
            SELECT satellite_name, COUNT(*) as count 
            FROM tle_data 
            GROUP BY satellite_name 
            ORDER BY count DESC 
            LIMIT 5
        """)
        satellites = cur.fetchall()
        conn.close()
        
        if not satellites:
            print("❌ Aucune donnée TLE trouvée dans la base!")
            exit(1)
        
        print("🛰️  Top 5 satellites par nombre de TLE:")
        for i, (name, count) in enumerate(satellites, 1):
            print(f"   {i}. {name}: {count} TLE")
        
        satellite_name, count = satellites[0]
        
        print(f"\n✨ Satellite sélectionné: {satellite_name}")
        print(f"   Nombre de TLE: {count}")
        
        if count < 8:
            print("\n⚠️  ATTENTION: Très peu de données!")
            print(f"   Minimum recommandé: 15 TLE")
            print(f"   Disponible: {count} TLE")
            print("\n   L'entraînement continuera mais la qualité sera limitée.")
        
        print("\n" + "="*60)
        result = train_model(
            satellite_name,
            epochs=50,
            batch_size=8,
            learning_rate=0.001
        )
        
        if result is None:
            print("\n❌ Entraînement échoué!")
            exit(1)
        
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        exit(1)