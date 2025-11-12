import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np

class TCNModel:
    def __init__(self, config, input_shape):
        """
        Inicializa modelo TCN (Temporal Convolutional Network)
        
        Args:
            config: Diccionario de configuración
            input_shape: (timesteps, features) ej: (60, 5)
        """
        self.config = config['models']['tcn']
        self.training_config = config['training']
        self.input_shape = input_shape
        self.model = None
        self.history = None
        
    def residual_block(self, x, dilation_rate, filters, kernel_size, dropout):
        """
        Bloque residual con convolución dilatada
        
        Args:
            x: Input tensor
            dilation_rate: Tasa de dilatación
            filters: Número de filtros
            kernel_size: Tamaño del kernel
            dropout: Tasa de dropout
        """
        # Rama convolucional
        conv = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding='causal',
            activation='relu'
        )(x)
        conv = layers.SpatialDropout1D(dropout)(conv)
        
        conv = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding='causal',
            activation='relu'
        )(conv)
        conv = layers.SpatialDropout1D(dropout)(conv)
        
        # Rama residual (ajustar dimensiones si es necesario)
        if x.shape[-1] != filters:
            residual = layers.Conv1D(filters=filters, kernel_size=1, padding='same')(x)
        else:
            residual = x
        
        # Sumar y activar
        output = layers.Add()([conv, residual])
        output = layers.Activation('relu')(output)
        
        return output
    
    def build(self):
        """Construye arquitectura TCN"""
        inputs = layers.Input(shape=self.input_shape)
        x = inputs
        
        # Bloques TCN con dilaciones crecientes
        for dilation in self.config['dilations']:
            x = self.residual_block(
                x,
                dilation_rate=dilation,
                filters=self.config['filters'],
                kernel_size=self.config['kernel_size'],
                dropout=self.config['dropout']
            )
        
        # Global pooling para reducir dimensión temporal
        x = layers.GlobalAveragePooling1D()(x)
        
        # Capas densas finales
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(self.config['dropout'])(x)
        outputs = layers.Dense(1, activation='linear', name='output')(x)
        
        self.model = Model(inputs=inputs, outputs=outputs, name='TCN_Model')
        
        # Compilar
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.config['learning_rate']),
            loss='mse',
            metrics=['mae', 'mape']
        )
        
        return self.model
    
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
        print(f"ENTRENANDO TCN")
        print(f"{'='*60}")
        print(f"Train shape: {X_train.shape}")
        print(f"Val shape: {X_val.shape}")
        print(f"Epochs: {self.config['epochs']}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Dilations: {self.config['dilations']}")
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