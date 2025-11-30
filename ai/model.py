"""
Modèle GRU en PyTorch pour prédiction de trajectoires satellitaires
Pourquoi PyTorch : Plus stable sur Windows, meilleure compatibilité
"""
import torch
import torch.nn as nn

class TrajectoryGRU(nn.Module):
    """
    Modèle GRU pour prédire les trajectoires satellites
    
    Architecture:
        - 2 couches GRU (128 → 64 unités)
        - Dropout pour régularisation
        - Dense layers pour la sortie
    
    Args:
        input_size: Nombre de features (6: x,y,z,vx,vy,vz)
        hidden_size: Taille de la couche GRU
        num_layers: Nombre de couches GRU empilées
        output_horizon: Nombre de positions futures à prédire
        dropout: Taux de dropout
    """
    
    def __init__(self, input_size=6, hidden_size=128, num_layers=2, 
                 output_horizon=5, dropout=0.2):
        super(TrajectoryGRU, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_horizon = output_horizon
        
        # Couches GRU
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True  # Input shape: (batch, seq, features)
        )
        
        # Couches fully connected
        self.fc1 = nn.Linear(hidden_size, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, output_horizon * 3)  # 5 positions * (x,y,z)
        
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Tensor de shape (batch_size, seq_length, input_size)
        
        Returns:
            Tensor de shape (batch_size, output_horizon, 3)
        """
        # GRU forward
        # gru_out shape: (batch, seq, hidden_size)
        # hidden shape: (num_layers, batch, hidden_size)
        gru_out, hidden = self.gru(x)
        
        # Prendre la dernière sortie de la séquence
        last_output = gru_out[:, -1, :]
        
        # Passer par les couches denses
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        # Reshape vers (batch_size, output_horizon, 3)
        # 3 = positions (x, y, z)
        out = out.view(-1, self.output_horizon, 3)
        
        return out
    
    def count_parameters(self):
        """Compter le nombre de paramètres entraînables"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class LightweightGRU(nn.Module):
    """
    Version légère pour tests rapides ou déploiement avec ressources limitées
    """
    
    def __init__(self, input_size=6, output_horizon=5):
        super(LightweightGRU, self).__init__()
        
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, output_horizon * 3)
        )
        
        self.output_horizon = output_horizon
    
    def forward(self, x):
        gru_out, _ = self.gru(x)
        out = self.fc(gru_out[:, -1, :])
        return out.view(-1, self.output_horizon, 3)


# Test du modèle
if __name__ == "__main__":
    print("🧠 Test du modèle TrajectoryGRU\n")
    
    # Créer le modèle
    model = TrajectoryGRU(
        input_size=6,
        hidden_size=128,
        num_layers=2,
        output_horizon=5,
        dropout=0.2
    )
    
    print("Architecture:")
    print(model)
    print(f"\n📊 Nombre de paramètres: {model.count_parameters():,}")
    
    # Test forward pass
    batch_size = 32
    seq_length = 10
    input_features = 6
    
    # Créer un batch de test
    x = torch.randn(batch_size, seq_length, input_features)
    print(f"\n🔢 Input shape: {x.shape}")
    print(f"   (batch={batch_size}, sequence={seq_length}, features={input_features})")
    
    # Forward pass
    y = model(x)
    print(f"\n🎯 Output shape: {y.shape}")
    print(f"   (batch={batch_size}, horizon=5, positions=3)")
    
    # Test modèle léger
    print("\n" + "="*50)
    print("🪶 Test du modèle LightweightGRU\n")
    
    light_model = LightweightGRU()
    print(f"Paramètres: {sum(p.numel() for p in light_model.parameters()):,}")
    
    y_light = light_model(x)
    print(f"Output shape: {y_light.shape}")
    
    print("\n✅ Tests réussis!")