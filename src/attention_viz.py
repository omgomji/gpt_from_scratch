"""
attention_viz.py
----------------
Extract and visualize attention weights.

Satisfies: V2 Upgrade (Attention Visualization)
"""

from typing import List, Optional

import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def capture_attention_weights(model: torch.nn.Module, idx: torch.Tensor) -> List[List[torch.Tensor]]:
    """Capture attention weights from a single forward pass.
    
    Parameters
    ----------
    model : torch.nn.Module
        The GPT model.
    idx : torch.Tensor
        Input token IDs of shape (1, T).
        
    Returns
    -------
    list[list[torch.Tensor]]
        A list of layers, where each layer is a list of heads.
        Each tensor is of shape (T, T).
    """
    model.eval()
    with torch.no_grad():
        model(idx)
        
    # Extract weights
    weights = []
    
    if not hasattr(model, "blocks"):
        return weights  # Bigram model has no attention
        
    for block in model.blocks:
        layer_weights = []
        for head in block.attn.heads:
            if hasattr(head, "_last_attn_weights") and head._last_attn_weights is not None:
                # _last_attn_weights is (B, T, T), we take B=0
                layer_weights.append(head._last_attn_weights[0].cpu())
        weights.append(layer_weights)
        
    return weights


def plot_attention_heatmap(
    weights: torch.Tensor, 
    tokens: List[str], 
    title: str = "Attention Weights", 
    save_path: Optional[str] = None
) -> plt.Figure:
    """Plot attention weights as a heatmap.
    
    Parameters
    ----------
    weights : torch.Tensor
        Attention weights of shape (T, T).
    tokens : list[str]
        List of tokens of length T.
    title : str
        Plot title.
    save_path : str | None
        If provided, save the plot to this path.
        
    Returns
    -------
    plt.Figure
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    
    cax = ax.matshow(weights.numpy(), cmap='viridis')
    fig.colorbar(cax)
    
    # Set up axes
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    
    ax.set_xticklabels(tokens, rotation=90)
    ax.set_yticklabels(tokens)
    
    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    
    ax.set_title(title, pad=20)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        
    plt.close(fig) # Prevent display if not in interactive mode
    return fig
