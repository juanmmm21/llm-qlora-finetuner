import os
import json
import math
import logging
from typing import Dict, List, Any, Tuple, Optional
import torch
import torch.nn as nn
from pydantic import BaseModel, Field

# Configuración del registrador de eventos
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Representación matemática de los 16 cuantiles del tipo NormalFloat4 (NF4)
# Diseñados por Tim Dettmers et al. para cuantización óptima de pesos con distribución normal.
NF4_QUANTILES = [
    -1.0,
    -0.6961917006969452,
    -0.5250730514526367,
    -0.3949161171913147,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105026721954346,
    0.0,
    0.07958029866218567,
    0.1609302043914795,
    0.24611230194568634,
    0.33791524171829224,
    0.4407000780105591,
    0.5626170039176941,
    0.7229568362236023,
    1.0
]


class LoraConfig(BaseModel):
    """Configuración de parámetros para la adaptación de bajo rango (LoRA)."""
    r: int = Field(default=8, description="Rango de las matrices de adaptacion")
    alpha: float = Field(default=16.0, description="Factor de escala constante para regular la influencia de LoRA")
    target_modules: List[str] = Field(default=["linear"], description="Lista de nombres de modulos a adaptar")
    dropout: float = Field(default=0.05, description="Tasa de dropout aplicada a las activaciones de entrada de LoRA")


class NF4Quantizer:
    """
    Simulador de cuantización NormalFloat4 (NF4) bloque por bloque.
    
    Permite emular la compresión de precisión de 32/16 bits a 4 bits (NF4)
    sin requerir compilaciones complejas de CUDA de bitsandbytes en local.
    """

    @staticmethod
    def quantize(tensor: torch.Tensor, block_size: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Cuantiza un tensor flotante a NF4 de forma bloque-a-bloque.
        
        Args:
            tensor: Tensor a cuantizar.
            block_size: Tamaño del bloque para normalización absmax.
            
        Returns:
            - indices: Tensor del mismo tamaño con valores enteros (0-15) representando indices NF4.
            - scales: Tensor con las escalas absmax correspondientes a cada bloque.
        """
        original_shape = tensor.shape
        flat_tensor = tensor.flatten()
        num_elements = flat_tensor.numel()
        
        # Relleno (padding) si el número de elementos no es divisible por block_size
        padding_size = (block_size - (num_elements % block_size)) % block_size
        if padding_size > 0:
            flat_tensor = torch.cat([flat_tensor, torch.zeros(padding_size, device=tensor.device)])
            
        # Redimensionar en bloques
        blocked_tensor = flat_tensor.view(-1, block_size)
        
        # Calcular el máximo absoluto por bloque (evitamos división por cero con epsilon)
        scales = torch.max(torch.abs(blocked_tensor), dim=1, keepdim=True)[0]
        scales = torch.clamp(scales, min=1e-8)
        
        # Normalizar el contenido del bloque al rango [-1.0, 1.0]
        normalized_blocked = blocked_tensor / scales
        
        # Buscar el cuantil NF4 más cercano para cada elemento
        quantiles = torch.tensor(NF4_QUANTILES, device=tensor.device, dtype=tensor.dtype)
        # Calculamos la distancia absoluta de cada elemento a los 16 cuantiles
        diffs = torch.abs(normalized_blocked.unsqueeze(-1) - quantiles) # Shape: [num_blocks, block_size, 16]
        indices = torch.argmin(diffs, dim=-1).to(torch.uint8) # Shape: [num_blocks, block_size]
        
        # Devolver a la forma original (recortando el padding si existiera)
        indices_flat = indices.flatten()
        if padding_size > 0:
            indices_flat = indices_flat[:-padding_size]
            
        return indices_flat.view(original_shape), scales.flatten()

    @staticmethod
    def dequantize(indices: torch.Tensor, scales: torch.Tensor, original_shape: torch.Size, block_size: int = 64) -> torch.Tensor:
        """
        Decuantiza los índices NF4 de vuelta a valores flotantes usando las escalas guardadas.
        """
        flat_indices = indices.flatten()
        num_elements = flat_indices.numel()
        
        padding_size = (block_size - (num_elements % block_size)) % block_size
        if padding_size > 0:
            flat_indices = torch.cat([flat_indices, torch.zeros(padding_size, device=indices.device, dtype=torch.uint8)])
            
        blocked_indices = flat_indices.view(-1, block_size)
        
        # Mapear índices a los valores reales de cuantiles NF4
        quantiles = torch.tensor(NF4_QUANTILES, device=indices.device, dtype=torch.float32)
        dequantized_blocked = quantiles[blocked_indices.long()] # Shape: [num_blocks, block_size]
        
        # Multiplicar por las escalas de bloque correspondiente
        scales_expanded = scales.unsqueeze(-1) # Shape: [num_blocks, 1]
        reconstructed_blocked = dequantized_blocked * scales_expanded
        
        # Aplanar y recortar padding
        reconstructed_flat = reconstructed_blocked.flatten()
        if padding_size > 0:
            reconstructed_flat = reconstructed_flat[:-padding_size]
            
        return reconstructed_flat.view(original_shape)
