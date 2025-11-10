"""
Visualization Utilities

Funções para desenhar anotações em frames de vídeo, incluindo bounding boxes,
labels e HUD (heads-up display) com informações.
"""

from typing import Optional, Dict, Any, Tuple

import cv2
import numpy as np


# Cores padrão em BGR
COLORS = {
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'blue': (255, 0, 0),
    'yellow': (0, 255, 255),
    'cyan': (255, 255, 0),
    'magenta': (255, 0, 255),
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'gray': (128, 128, 128),
    'orange': (0, 165, 255),
    'purple': (128, 0, 128),
}


def draw_box_and_label(
    frame: np.ndarray,
    box: Tuple[int, int, int, int],
    label: str,
    color: Tuple[int, int, int] = COLORS['green'],
    thickness: int = 2,
    font_scale: float = 0.6,
    font_thickness: int = 2,
    background_alpha: float = 0.7
) -> np.ndarray:
    """
    Desenha uma bounding box e label em um frame.
    
    Args:
        frame: Frame de imagem (numpy array BGR)
        box: Bounding box no formato (x, y, width, height)
        label: Texto do label a ser exibido
        color: Cor da box e texto em formato BGR
        thickness: Espessura da linha da box
        font_scale: Escala da fonte do texto
        font_thickness: Espessura da fonte
        background_alpha: Transparência do fundo do label (0.0 a 1.0)
        
    Returns:
        Frame com a box e label desenhados
        
    Example:
        >>> frame = np.zeros((480, 640, 3), dtype=np.uint8)
        >>> box = (100, 100, 200, 150)
        >>> frame = draw_box_and_label(frame, box, "Person: 0.95")
    """
    if frame is None or frame.size == 0:
        raise ValueError("Frame is None or empty")
    
    if len(box) != 4:
        raise ValueError("Box must have 4 values: (x, y, width, height)")
    
    # Fazer cópia para não modificar o original
    output = frame.copy()
    
    x, y, w, h = box
    
    # Garantir que coordenadas são válidas
    x = max(0, int(x))
    y = max(0, int(y))
    w = max(1, int(w))
    h = max(1, int(h))
    
    # Desenhar retângulo da bounding box
    cv2.rectangle(output, (x, y), (x + w, y + h), color, thickness)
    
    # Preparar label
    if label:
        # Calcular tamanho do texto
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )
        
        # Posição do label (acima da box)
        label_y = y - 10 if y > 30 else y + h + 20
        label_x = x
        
        # Garantir que o label não saia da imagem
        if label_x + text_width > output.shape[1]:
            label_x = output.shape[1] - text_width - 5
        if label_x < 0:
            label_x = 5
        
        # Desenhar fundo do label com transparência
        padding = 5
        bg_top_left = (label_x - padding, label_y - text_height - padding)
        bg_bottom_right = (label_x + text_width + padding, label_y + baseline + padding)
        
        # Criar overlay para transparência
        overlay = output.copy()
        cv2.rectangle(
            overlay,
            bg_top_left,
            bg_bottom_right,
            color,
            -1  # Preenchido
        )
        
        # Aplicar transparência
        cv2.addWeighted(overlay, background_alpha, output, 1 - background_alpha, 0, output)
        
        # Desenhar texto
        cv2.putText(
            output,
            label,
            (label_x, label_y),
            font,
            font_scale,
            COLORS['white'],
            font_thickness,
            cv2.LINE_AA
        )
    
    return output


def put_hud(
    frame: np.ndarray,
    stats_dict: Dict[str, Any],
    position: str = "top-left",
    font_scale: float = 0.5,
    font_thickness: int = 1,
    line_spacing: int = 25,
    background_color: Tuple[int, int, int] = (0, 0, 0),
    text_color: Tuple[int, int, int] = COLORS['white'],
    background_alpha: float = 0.6,
    padding: int = 10
) -> np.ndarray:
    """
    Desenha um HUD (heads-up display) com estatísticas no frame.
    
    Args:
        frame: Frame de imagem (numpy array BGR)
        stats_dict: Dicionário com estatísticas a exibir
                   Ex: {'FPS': 30.5, 'Frame': 120, 'Faces': 2}
        position: Posição do HUD ('top-left', 'top-right', 'bottom-left', 'bottom-right')
        font_scale: Escala da fonte
        font_thickness: Espessura da fonte
        line_spacing: Espaçamento entre linhas
        background_color: Cor de fundo do HUD
        text_color: Cor do texto
        background_alpha: Transparência do fundo (0.0 a 1.0)
        padding: Padding interno do HUD
        
    Returns:
        Frame com o HUD desenhado
        
    Example:
        >>> frame = np.zeros((480, 640, 3), dtype=np.uint8)
        >>> stats = {'FPS': 30.0, 'Frame': 120, 'Faces': 2, 'Timestamp': '00:04'}
        >>> frame = put_hud(frame, stats)
    """
    if frame is None or frame.size == 0:
        raise ValueError("Frame is None or empty")
    
    if not stats_dict:
        return frame
    
    # Fazer cópia para não modificar o original
    output = frame.copy()
    
    # Preparar linhas de texto
    lines = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    for key, value in stats_dict.items():
        # Formatar valor
        if isinstance(value, float):
            text = f"{key}: {value:.2f}"
        elif isinstance(value, int):
            text = f"{key}: {value}"
        else:
            text = f"{key}: {value}"
        lines.append(text)
    
    if not lines:
        return output
    
    # Calcular dimensões do HUD
    max_text_width = 0
    total_height = 0
    
    for line in lines:
        (text_width, text_height), baseline = cv2.getTextSize(
            line, font, font_scale, font_thickness
        )
        max_text_width = max(max_text_width, text_width)
        total_height += line_spacing
    
    # Adicionar padding
    hud_width = max_text_width + 2 * padding
    hud_height = total_height + padding
    
    # Calcular posição do HUD
    frame_height, frame_width = output.shape[:2]
    
    if position == "top-left":
        x_start = padding
        y_start = padding
    elif position == "top-right":
        x_start = frame_width - hud_width - padding
        y_start = padding
    elif position == "bottom-left":
        x_start = padding
        y_start = frame_height - hud_height - padding
    elif position == "bottom-right":
        x_start = frame_width - hud_width - padding
        y_start = frame_height - hud_height - padding
    else:
        raise ValueError(f"Invalid position: {position}")
    
    # Garantir que o HUD não saia da imagem
    x_start = max(0, min(x_start, frame_width - hud_width))
    y_start = max(0, min(y_start, frame_height - hud_height))
    
    # Desenhar fundo do HUD com transparência
    overlay = output.copy()
    cv2.rectangle(
        overlay,
        (x_start, y_start),
        (x_start + hud_width, y_start + hud_height),
        background_color,
        -1
    )
    
    # Aplicar transparência
    cv2.addWeighted(overlay, background_alpha, output, 1 - background_alpha, 0, output)
    
    # Desenhar borda
    cv2.rectangle(
        output,
        (x_start, y_start),
        (x_start + hud_width, y_start + hud_height),
        text_color,
        1
    )
    
    # Desenhar texto
    y_offset = y_start + padding + 15
    for line in lines:
        cv2.putText(
            output,
            line,
            (x_start + padding, y_offset),
            font,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA
        )
        y_offset += line_spacing
    
    return output


def draw_landmarks(
    frame: np.ndarray,
    landmarks: Dict[str, Tuple[int, int]],
    color: Tuple[int, int, int] = COLORS['cyan'],
    radius: int = 3,
    thickness: int = -1
) -> np.ndarray:
    """
    Desenha landmarks faciais no frame.
    
    Args:
        frame: Frame de imagem (numpy array BGR)
        landmarks: Dicionário com pontos faciais {'nome': (x, y)}
        color: Cor dos pontos
        radius: Raio dos círculos
        thickness: Espessura (-1 para preenchido)
        
    Returns:
        Frame com landmarks desenhados
    """
    if frame is None or frame.size == 0:
        raise ValueError("Frame is None or empty")
    
    if not landmarks:
        return frame
    
    output = frame.copy()
    
    for name, (x, y) in landmarks.items():
        cv2.circle(output, (int(x), int(y)), radius, color, thickness)
    
    return output


def format_timestamp(seconds: float) -> str:
    """
    Formata timestamp em segundos para MM:SS.
    
    Args:
        seconds: Tempo em segundos
        
    Returns:
        String formatada "MM:SS"
    """
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def create_color_palette(n_colors: int) -> list[Tuple[int, int, int]]:
    """
    Cria uma paleta de cores distintas.
    
    Args:
        n_colors: Número de cores a gerar
        
    Returns:
        Lista de cores em formato BGR
    """
    colors = []
    for i in range(n_colors):
        hue = int(180 * i / n_colors)
        # Usar HSV para gerar cores distintas
        hsv = np.uint8([[[hue, 255, 255]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        color = tuple(map(int, bgr[0, 0]))
        colors.append(color)
    return colors
