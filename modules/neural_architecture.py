"""
DEMIR AI v8.0 - Advanced Neural Network Architecture
TRANSFORMER-BASED DEEP LEARNING MODELS
PROFESSIONAL ENTERPRISE IMPLEMENTATION
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import logging
from datetime import datetime, timedelta
import math

logger = logging.getLogger(__name__)


# ====================== ATTENTION MECHANISMS ======================

class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention Mechanism
    SCALED DOT-PRODUCT ATTENTION WITH MULTIPLE HEADS
    """
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "Embedding dimension must be divisible by number of heads"
        
        # Linear projections for Q, K, V
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        
        # Output projection
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        
        # Dropout and layer norm
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        
        # Positional encoding
        self.register_buffer('positional_encoding', self._create_positional_encoding(5000, embed_dim))
        
    def _create_positional_encoding(self, max_len: int, embed_dim: int) -> torch.Tensor:
        """Create sinusoidal positional encoding"""
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                            (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of multi-head attention
        Args:
            x: Input tensor (batch_size, seq_len, embed_dim)
            mask: Attention mask (batch_size, seq_len, seq_len)
        """
        batch_size, seq_len, embed_dim = x.size()
        
        # Add positional encoding
        x = x + self.positional_encoding[:, :seq_len, :]
        
        # Linear projections
        Q = self.q_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.k_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.v_linear(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Attention weights
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        attention_output = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        
        # Output projection
        output = self.out_linear(attention_output)
        output = self.dropout(output)
        
        # Residual connection and layer norm
        output = self.layer_norm(output + x)
        
        return output


class CrossModalAttention(nn.Module):
    """
    Cross-Modal Attention for Multi-Modal Learning
    FUSES INFORMATION FROM DIFFERENT DATA MODALITIES
    """
    
    def __init__(self, 
                 price_dim: int, 
                 volume_dim: int, 
                 sentiment_dim: int,
                 hidden_dim: int = 256):
        super().__init__()
        
        # Modality-specific encoders
        self.price_encoder = nn.Linear(price_dim, hidden_dim)
        self.volume_encoder = nn.Linear(volume_dim, hidden_dim)
        self.sentiment_encoder = nn.Linear(sentiment_dim, hidden_dim)
        
        # Cross-attention layers
        self.price_to_volume = nn.MultiheadAttention(hidden_dim, num_heads=8)
        self.price_to_sentiment = nn.MultiheadAttention(hidden_dim, num_heads=8)
        self.volume_to_sentiment = nn.MultiheadAttention(hidden_dim, num_heads=8)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
    def forward(self, 
                price_data: torch.Tensor,
                volume_data: torch.Tensor,
                sentiment_data: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with cross-modal attention
        """
        # Encode modalities
        price_encoded = self.price_encoder(price_data)
        volume_encoded = self.volume_encoder(volume_data)
        sentiment_encoded = self.sentiment_encoder(sentiment_data)
        
        # Cross-attention
        price_volume, _ = self.price_to_volume(price_encoded, volume_encoded, volume_encoded)
        price_sentiment, _ = self.price_to_sentiment(price_encoded, sentiment_encoded, sentiment_encoded)
        volume_sentiment, _ = self.volume_to_sentiment(volume_encoded, sentiment_encoded, sentiment_encoded)
        
        # Concatenate all representations
        combined = torch.cat([price_volume, price_sentiment, volume_sentiment], dim=-1)
        
        # Fusion
        fused = self.fusion(combined)
        
        return fused


# ====================== TRANSFORMER ARCHITECTURE ======================

class MarketTransformer(nn.Module):
    """
    Transformer Model for Financial Time Series
    STATE-OF-THE-ART SEQUENCE MODELING
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        # Model configuration
        self.input_dim = config['input_dim']
        self.hidden_dim = config.get('hidden_dim', 512)
        self.num_layers = config.get('num_layers', 6)
        self.num_heads = config.get('num_heads', 8)
        self.dropout = config.get('dropout', 0.1)
        self.max_seq_len = config.get('max_seq_len', 1000)
        
        # Input embedding
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(self.hidden_dim, self.max_seq_len)
        
        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer = TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        
        # Output heads for different tasks
        self.price_prediction_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 1)
        )
        
        self.classification_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 3)  # Buy/Hold/Sell
        )
        
        self.volatility_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 1)
        )
        
        logger.info(f"MarketTransformer initialized with {self.num_layers} layers, {self.num_heads} heads")
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through transformer
        Returns multiple predictions
        """
        # Input projection
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Transformer encoding
        encoded = self.transformer(x, src_key_padding_mask=mask)
        
        # Global pooling (use last token or mean)
        if mask is not None:
            # Masked mean pooling
            mask_expanded = mask.unsqueeze(-1).expand(encoded.size())
            sum_embeddings = torch.sum(encoded * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            pooled = sum_embeddings / sum_mask
        else:
            pooled = encoded.mean(dim=1)
        
        # Multi-task predictions
        outputs = {
            'price_prediction': self.price_prediction_head(pooled),
            'trading_signal': F.softmax(self.classification_head(pooled), dim=-1),
            'volatility': torch.abs(self.volatility_head(pooled))
        }
        
        return outputs
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Extract attention weights for interpretability"""
        with torch.no_grad():
            x = self.input_projection(x)
            x = self.positional_encoding(x)
            
            # Get attention weights from first layer
            attn_weights = []
            for layer in self.transformer.layers:
                _, weights = layer.self_attn(x, x, x, need_weights=True)
                attn_weights.append(weights)
            
            return torch.stack(attn_weights)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


# ====================== LSTM WITH ATTENTION ======================

class AttentionLSTM(nn.Module):
    """
    LSTM with Attention Mechanism
    COMBINES RECURRENT AND ATTENTION ARCHITECTURES
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.input_dim = config['input_dim']
        self.hidden_dim = config.get('hidden_dim', 256)
        self.num_layers = config.get('num_layers', 3)
        self.dropout = config.get('dropout', 0.2)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=True
        )
        
        # Attention layer
        self.attention = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1)
        )
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 1)
        )
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with attention
        Returns predictions and attention weights
        """
        # LSTM encoding
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Calculate attention scores
        attention_scores = self.attention(lstm_out)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Apply attention
        attended = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Final prediction
        output = self.output_layer(attended)
        
        return output, attention_weights


# ====================== GAN FOR MARKET SIMULATION ======================

class MarketGAN(nn.Module):
    """
    Generative Adversarial Network for Market Simulation
    GENERATES REALISTIC MARKET SCENARIOS
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.latent_dim = config.get('latent_dim', 100)
        self.sequence_length = config.get('sequence_length', 100)
        self.feature_dim = config.get('feature_dim', 5)  # OHLCV
        
        # Generator
        self.generator = self._build_generator()
        
        # Discriminator
        self.discriminator = self._build_discriminator()
        
        # Wasserstein loss with gradient penalty
        self.use_wasserstein = config.get('use_wasserstein', True)
        self.gradient_penalty_weight = config.get('gradient_penalty_weight', 10)
        
        logger.info("MarketGAN initialized with Wasserstein loss")
    
    def _build_generator(self) -> nn.Module:
        """Build generator network"""
        return nn.Sequential(
            # Input: (batch_size, latent_dim)
            nn.Linear(self.latent_dim, 256),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(256),
            
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(512),
            
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(1024),
            
            nn.Linear(1024, self.sequence_length * self.feature_dim),
            nn.Tanh()
        )
    
    def _build_discriminator(self) -> nn.Module:
        """Build discriminator network"""
        return nn.Sequential(
            # Input: (batch_size, sequence_length * feature_dim)
            nn.Linear(self.sequence_length * self.feature_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(128, 1)
            # No sigmoid for Wasserstein GAN
        )
    
    def generate(self, batch_size: int) -> torch.Tensor:
        """Generate synthetic market data"""
        noise = torch.randn(batch_size, self.latent_dim)
        generated = self.generator(noise)
        
        # Reshape to (batch_size, sequence_length, feature_dim)
        generated = generated.view(batch_size, self.sequence_length, self.feature_dim)
        
        return generated
    
    def discriminate(self, x: torch.Tensor) -> torch.Tensor:
        """Discriminate real vs fake data"""
        # Flatten input
        x_flat = x.view(x.size(0), -1)
        return self.discriminator(x_flat)
    
    def calculate_gradient_penalty(self, real_data: torch.Tensor, fake_data: torch.Tensor) -> torch.Tensor:
        """Calculate gradient penalty for WGAN-GP"""
        batch_size = real_data.size(0)
        
        # Random interpolation
        alpha = torch.rand(batch_size, 1, 1).expand_as(real_data)
        interpolated = alpha * real_data + (1 - alpha) * fake_data
        interpolated.requires_grad_(True)
        
        # Calculate discriminator output
        d_interpolated = self.discriminate(interpolated)
        
        # Calculate gradients
        gradients = torch.autograd.grad(
            outputs=d_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(d_interpolated),
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Gradient penalty
        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        
        return gradient_penalty


# ====================== GRAPH NEURAL NETWORK ======================

class MarketGraphNetwork(nn.Module):
    """
    Graph Neural Network for Market Structure
    MODELS RELATIONSHIPS BETWEEN ASSETS
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.node_features = config['node_features']
        self.hidden_dim = config.get('hidden_dim', 128)
        self.num_layers = config.get('num_layers', 3)
        self.dropout = config.get('dropout', 0.2)
        
        # Graph convolution layers
        self.graph_convs = nn.ModuleList()
        
        for i in range(self.num_layers):
            in_dim = self.node_features if i == 0 else self.hidden_dim
            self.graph_convs.append(
                GraphConvLayer(in_dim, self.hidden_dim)
            )
        
        # Global graph pooling
        self.global_pool = GlobalAttentionPool(self.hidden_dim)
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, 1)
        )
        
    def forward(self, 
                node_features: torch.Tensor,
                edge_index: torch.Tensor,
                edge_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass through graph network
        Args:
            node_features: (num_nodes, node_features)
            edge_index: (2, num_edges) - source and target nodes
            edge_weights: Optional edge weights
        """
        x = node_features
        
        # Graph convolutions
        for conv in self.graph_convs:
            x = conv(x, edge_index, edge_weights)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Global pooling
        graph_embedding = self.global_pool(x)
        
        # Output
        output = self.output_layer(graph_embedding)
        
        return output


class GraphConvLayer(nn.Module):
    """Single graph convolution layer"""
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.neighbor_aggregator = nn.Linear(in_features, out_features)
        
    def forward(self, 
                x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Graph convolution forward pass
        """
        # Self features
        self_features = self.linear(x)
        
        # Aggregate neighbor features
        row, col = edge_index
        neighbor_features = x[col]
        
        if edge_weights is not None:
            neighbor_features = neighbor_features * edge_weights.unsqueeze(-1)
        
        # Mean aggregation
        aggregated = torch.zeros_like(self_features)
        aggregated.index_add_(0, row, self.neighbor_aggregator(neighbor_features))
        
        # Combine self and neighbor features
        out = self_features + aggregated
        
        return out


class GlobalAttentionPool(nn.Module):
    """Global attention pooling for graphs"""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool node features to graph representation"""
        attention_scores = self.attention(x)
        attention_weights = F.softmax(attention_scores, dim=0)
        
        pooled = torch.sum(attention_weights * x, dim=0)
        
        return pooled


# ====================== VARIATIONAL AUTOENCODER ======================

class MarketVAE(nn.Module):
    """
    Variational Autoencoder for Market Data
    LEARNS LATENT REPRESENTATIONS OF MARKET STATES
    """
    
    def __init__(self, config: Dict):
        super().__init__()
        
        self.input_dim = config['input_dim']
        self.latent_dim = config.get('latent_dim', 32)
        self.hidden_dims = config.get('hidden_dims', [128, 64])
        
        # Encoder
        encoder_layers = []
        in_dim = self.input_dim
        
        for hidden_dim in self.hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            in_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space
        self.fc_mu = nn.Linear(self.hidden_dims[-1], self.latent_dim)
        self.fc_logvar = nn.Linear(self.hidden_dims[-1], self.latent_dim)
        
        # Decoder
        decoder_layers = []
        in_dim = self.latent_dim
        
        for hidden_dim in reversed(self.hidden_dims):
            decoder_layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            in_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(self.hidden_dims[0], self.input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent vector to reconstruction"""
        return self.decoder(z)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through VAE"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        reconstruction = self.decode(z)
        return reconstruction, mu, logvar
    
    def loss_function(self, 
                     reconstruction: torch.Tensor,
                     x: torch.Tensor,
                     mu: torch.Tensor,
                     logvar: torch.Tensor,
                     beta: float = 1.0) -> Dict[str, torch.Tensor]:
        """VAE loss with optional beta for disentanglement"""
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstruction, x, reduction='sum')
        
        # KL divergence
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        loss = recon_loss + beta * kl_loss
        
        return {
            'loss': loss,
            'reconstruction_loss': recon_loss,
            'kl_loss': kl_loss
        }


# ====================== NEURAL ARCHITECTURE SEARCH ======================

class AutoML_NAS:
    """
    Neural Architecture Search
    AUTOMATICALLY FINDS OPTIMAL ARCHITECTURE
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.search_space = self._define_search_space()
        self.population_size = config.get('population_size', 20)
        self.num_generations = config.get('num_generations', 50)
        self.mutation_rate = config.get('mutation_rate', 0.1)
        
        self.best_architecture = None
        self.best_performance = -float('inf')
        
        logger.info("AutoML NAS initialized with evolutionary search")
    
    def _define_search_space(self) -> Dict:
        """Define the neural architecture search space"""
        return {
            'num_layers': list(range(2, 10)),
            'hidden_dims': [64, 128, 256, 512, 1024],
            'activation': ['relu', 'gelu', 'swish', 'mish'],
            'dropout': [0.0, 0.1, 0.2, 0.3, 0.4],
            'optimizer': ['adam', 'sgd', 'rmsprop', 'adamw'],
            'learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01],
            'batch_norm': [True, False],
            'skip_connections': [True, False]
        }
    
    def create_model(self, architecture: Dict) -> nn.Module:
        """Create model from architecture specification"""
        layers = []
        input_dim = self.config['input_dim']
        
        for i in range(architecture['num_layers']):
            # Linear layer
            output_dim = architecture['hidden_dims']
            layers.append(nn.Linear(input_dim, output_dim))
            
            # Batch normalization
            if architecture['batch_norm']:
                layers.append(nn.BatchNorm1d(output_dim))
            
            # Activation
            activation = self._get_activation(architecture['activation'])
            layers.append(activation)
            
            # Dropout
            if architecture['dropout'] > 0:
                layers.append(nn.Dropout(architecture['dropout']))
            
            input_dim = output_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, self.config['output_dim']))
        
        return nn.Sequential(*layers)
    
    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name"""
        activations = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'mish': nn.Mish()
        }
        return activations.get(name, nn.ReLU())
    
    def evolutionary_search(self, train_data: DataLoader, val_data: DataLoader) -> Dict:
        """
        Perform evolutionary neural architecture search
        """
        # Initialize population
        population = self._initialize_population()
        
        for generation in range(self.num_generations):
            # Evaluate population
            fitness_scores = []
            for architecture in population:
                fitness = self._evaluate_architecture(architecture, train_data, val_data)
                fitness_scores.append(fitness)
                
                # Track best
                if fitness > self.best_performance:
                    self.best_performance = fitness
                    self.best_architecture = architecture
            
            # Selection
            parents = self._selection(population, fitness_scores)
            
            # Crossover and mutation
            offspring = self._crossover(parents)
            offspring = self._mutation(offspring)
            
            # Create new population
            population = parents[:self.population_size//2] + offspring
            
            logger.info(f"Generation {generation}: Best fitness = {max(fitness_scores):.4f}")
        
        return self.best_architecture
    
    def _initialize_population(self) -> List[Dict]:
        """Initialize random population"""
        population = []
        
        for _ in range(self.population_size):
            architecture = {}
            for param, choices in self.search_space.items():
                architecture[param] = np.random.choice(choices)
            population.append(architecture)
        
        return population
    
    def _evaluate_architecture(self, 
                              architecture: Dict,
                              train_data: DataLoader,
                              val_data: DataLoader) -> float:
        """Evaluate architecture performance"""
        model = self.create_model(architecture)
        
        # Quick training (few epochs for NAS)
        optimizer = self._get_optimizer(model, architecture)
        criterion = nn.MSELoss()
        
        # Train for a few epochs
        model.train()
        for epoch in range(5):  # Quick evaluation
            for batch_x, batch_y in train_data:
                optimizer.zero_grad()
                output = model(batch_x)
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_data:
                output = model(batch_x)
                val_loss += criterion(output, batch_y).item()
        
        # Return negative loss as fitness
        return -val_loss / len(val_data)
    
    def _get_optimizer(self, model: nn.Module, architecture: Dict) -> optim.Optimizer:
        """Get optimizer based on architecture"""
        lr = architecture['learning_rate']
        
        if architecture['optimizer'] == 'adam':
            return optim.Adam(model.parameters(), lr=lr)
        elif architecture['optimizer'] == 'sgd':
            return optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        elif architecture['optimizer'] == 'rmsprop':
            return optim.RMSprop(model.parameters(), lr=lr)
        elif architecture['optimizer'] == 'adamw':
            return optim.AdamW(model.parameters(), lr=lr)
        else:
            return optim.Adam(model.parameters(), lr=lr)
    
    def _selection(self, population: List[Dict], fitness_scores: List[float]) -> List[Dict]:
        """Tournament selection"""
        parents = []
        tournament_size = 3
        
        for _ in range(self.population_size // 2):
            # Tournament
            tournament_idx = np.random.choice(len(population), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_idx]
            winner_idx = tournament_idx[np.argmax(tournament_fitness)]
            parents.append(population[winner_idx])
        
        return parents
    
    def _crossover(self, parents: List[Dict]) -> List[Dict]:
        """Uniform crossover"""
        offspring = []
        
        for i in range(0, len(parents) - 1, 2):
            child1, child2 = {}, {}
            
            for param in self.search_space.keys():
                if np.random.random() < 0.5:
                    child1[param] = parents[i][param]
                    child2[param] = parents[i+1][param]
                else:
                    child1[param] = parents[i+1][param]
                    child2[param] = parents[i][param]
            
            offspring.extend([child1, child2])
        
        return offspring
    
    def _mutation(self, offspring: List[Dict]) -> List[Dict]:
        """Random mutation"""
        for child in offspring:
            for param, choices in self.search_space.items():
                if np.random.random() < self.mutation_rate:
                    child[param] = np.random.choice(choices)
        
        return offspring


# ====================== TRAINING UTILITIES ======================

class ModelTrainer:
    """
    Professional model training orchestrator
    HANDLES TRAINING, VALIDATION, AND DEPLOYMENT
    """
    
    def __init__(self, model: nn.Module, config: Dict):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Training configuration
        self.epochs = config.get('epochs', 100)
        self.batch_size = config.get('batch_size', 32)
        self.learning_rate = config.get('learning_rate', 0.001)
        self.early_stopping_patience = config.get('early_stopping_patience', 10)
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.5
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        
        logger.info(f"ModelTrainer initialized on {self.device}")
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            
            # Handle dictionary outputs
            if isinstance(output, dict):
                output = output['price_prediction']
            
            loss = self.criterion(output, target)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            epoch_loss += loss.item()
        
        return epoch_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate model"""
        self.model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                if isinstance(output, dict):
                    output = output['price_prediction']
                
                loss = self.criterion(output, target)
                val_loss += loss.item()
        
        return val_loss / len(val_loader)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """Full training loop"""
        logger.info("Starting training...")
        
        for epoch in range(self.epochs):
            # Training
            train_loss = self.train_epoch(train_loader)
            
            # Validation
            val_loss = self.validate(val_loader)
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.early_stopping_counter = 0
                self.save_checkpoint(epoch)
            else:
                self.early_stopping_counter += 1
                
                if self.early_stopping_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break
            
            # Logging
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        logger.info("Training completed")
        return self.history
    
    def save_checkpoint(self, epoch: int):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'history': self.history
        }
        
        torch.save(checkpoint, f'checkpoint_epoch_{epoch}.pth')
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        self.history = checkpoint['history']
        
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
