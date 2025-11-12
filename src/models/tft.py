import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np

class TFTModel:
    def __init__(self, config, input_shape):
        """
        Inicializa modelo TFT simplificado (Temporal Fusion Transformer)
        
        Args:
            config: Diccionario de configuración
            input_shape: (timesteps, features) ej: (30, 11)
        """
        self.config = config['models']['tft']
        self.training_config = config['training']
        self.input_shape = input_shape
        self.model = None
        self.history = None
        
    def attention_block(self, x, num_heads, key_dim):
        """Bloque de multi-head attention"""
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=key_dim,
            dropout=self.config['dropout']
        )(x, x)
        
        # Add & Norm
        x = layers.Add()([x, attention_output])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        # Feed-forward
        ffn = layers.Dense(self.config['hidden_size'] * 2, activation='relu')(x)
        ffn = layers.Dropout(self.config['dropout'])(ffn)
        ffn = layers.Dense(self.config['hidden_size'])(ffn)
        
        # Add & Norm
        x = layers.Add()([x, ffn])
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        
        return x
    
    def build(self):
        """Construye arquitectura TFT simplificada"""
        inputs = layers.Input(shape=self.input_shape, name='input_layer')
        
        # Proyección de entrada al hidden_size
        x = layers.Dense(self.config['hidden_size'], name='input_projection')(inputs)
        
        # Positional encoding FIJO (no trainable)
        timesteps = self.input_shape[0]
        position_enc = self.get_positional_encoding(timesteps, self.config['hidden_size'])
        # Broadcast to batch dimension
        x = x + position_enc
        
        # Bloques de atención
        num_layers = 2  # 2 capas de transformer
        for i in range(num_layers):
            x = self.attention_block(
                x,
                num_heads=self.config['num_attention_heads'],
                key_dim=self.config['hidden_size'] // self.config['num_attention_heads']
            )
        
        # Global pooling
        x = layers.GlobalAveragePooling1D()(x)
        
        # Capas densas finales
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(self.config['dropout'])(x)
        x = layers.Dense(32, activation='relu')(x)
        x = layers.Dropout(self.config['dropout'])(x)
        outputs = layers.Dense(1, activation='linear', name='output')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs, name='TFT_Model')
        
        # Compilar
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        return self.model
    
    def get_positional_encoding(self, seq_len, d_model):
        """
        Genera positional encoding fijo (sin entrenamiento)
        
        Args:
            seq_len: Longitud de la secuencia (timesteps)
            d_model: Dimensión del modelo (hidden_size)
        """
        position = np.arange(seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        
        pos_encoding = np.zeros((seq_len, d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)
        
        # Convertir a tensor y agregar batch dimension
        pos_encoding = tf.cast(pos_encoding[np.newaxis, :, :], dtype=tf.float32)
        
        return pos_encoding
    
    def get_callbacks(self, model_path):
        """Retorna callbacks para entrenamiento"""
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=self.training_config['early_stopping_patience'],
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=self.training_config['reduce_lr_patience'],
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        return callbacks
    
    def train(self, X_train, y_train, X_val, y_val, model_path):
        """
        Entrena el modelo
        
        Args:
            X_train: (n_samples, timesteps, features)
            y_train: (n_samples,)
            X_val: Validation data
            y_val: Validation targets
            model_path: Ruta para guardar mejor modelo
        """
        if self.model is None:
            self.build()
        
        print(f"\n{'='*60}")
        print(f"ENTRENANDO TFT (Transformer)")
        print(f"{'='*60}")
        print(f"Train shape: {X_train.shape}")
        print(f"Val shape: {X_val.shape}")
        print(f"Epochs: {self.config['epochs']}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Attention heads: {self.config['num_attention_heads']}")
        print(f"Hidden size: {self.config['hidden_size']}")
        print(f"{'='*60}\n")
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.config['epochs'],
            batch_size=self.config['batch_size'],
            callbacks=self.get_callbacks(model_path),
            verbose=1
        )
        
        return self.history
    
    def predict(self, X):
        """Realiza predicciones"""
        return self.model.predict(X, verbose=0).flatten()
    
    def evaluate(self, X, y):
        """Evalúa el modelo"""
        return self.model.evaluate(X, y, verbose=0)
    
    def summary(self):
        """Muestra arquitectura del modelo"""
        if self.model:
            self.model.summary()
        else:
            print("Modelo no construido. Llama a build() primero.")
    
    def load_weights(self, path):
        """Carga pesos guardados"""
        if self.model is None:
            self.build()
        self.model.load_weights(path)
        print(f"✓ Pesos cargados desde {path}")
    
    def save(self, path):
        """Guarda modelo completo"""
        self.model.save(path)
        print(f"✓ Modelo guardado en {path}")