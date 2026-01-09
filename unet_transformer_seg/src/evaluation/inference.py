"""Single image inference and prediction utilities."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Optional, Tuple, Union, List
from pathlib import Path
import logging

from ..data.transforms import get_transforms
from ..utils.logging_utils import setup_logger


class SegmentationInference:
    """
    Single image inference pipeline for segmentation models.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: Optional[torch.device] = None,
        num_classes: int = 3,
        class_names: Optional[List[str]] = None,
        image_size: Tuple[int, int] = (256, 256)
    ):
        """
        Initialize inference pipeline.
        
        Args:
            model: Trained segmentation model
            device: Inference device
            num_classes: Number of classes
            class_names: Optional class names for visualization
            image_size: Input image size for model
        """
        self.model = model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.image_size = image_size
        
        # Move model to device and set to eval mode
        self.model.to(self.device)
        self.model.eval()
        
        # Setup transforms
        _, self.transform = get_transforms(
            image_size=image_size,
            augment=False  # No augmentation for inference
        )
        
        # Setup logging
        self.logger = setup_logger(
            name="inference",
            level=logging.INFO
        )
        
        self.logger.info(f"Initialized inference pipeline on {self.device}")
    
    def predict_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor],
        return_probabilities: bool = False,
        apply_softmax: bool = True
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Predict segmentation mask for a single image.
        
        Args:
            image: Input image (path, PIL Image, numpy array, or tensor)
            return_probabilities: Whether to return class probabilities
            apply_softmax: Whether to apply softmax to logits
            
        Returns:
            Predicted mask (and probabilities if requested)
        """
        # Preprocess image
        input_tensor = self._preprocess_image(image)
        
        with torch.no_grad():
            # Forward pass
            logits = self.model(input_tensor)
            
            # Apply softmax if requested
            if apply_softmax:
                probabilities = F.softmax(logits, dim=1)
            else:
                probabilities = logits
            
            # Get predicted classes
            predicted_mask = probabilities.argmax(dim=1).squeeze(0).cpu().numpy()
            
            if return_probabilities:
                prob_array = probabilities.squeeze(0).cpu().numpy()
                return predicted_mask, prob_array
            else:
                return predicted_mask
    
    def predict_batch(
        self,
        images: List[Union[str, Path, Image.Image, np.ndarray]],
        batch_size: int = 8
    ) -> List[np.ndarray]:
        """
        Predict segmentation masks for multiple images.
        
        Args:
            images: List of input images
            batch_size: Batch size for processing
            
        Returns:
            List of predicted masks
        """
        predictions = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            
            # Preprocess batch
            batch_tensors = []
            for img in batch_images:
                tensor = self._preprocess_image(img)
                batch_tensors.append(tensor.squeeze(0))
            
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            with torch.no_grad():
                # Forward pass
                logits = self.model(batch_tensor)
                probabilities = F.softmax(logits, dim=1)
                
                # Get predicted classes
                batch_masks = probabilities.argmax(dim=1).cpu().numpy()
                
                predictions.extend(batch_masks)
        
        return predictions
    
    def _preprocess_image(
        self,
        image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """
        Preprocess input image for model inference.
        
        Args:
            image: Input image in various formats
            
        Returns:
            Preprocessed tensor ready for model input
        """
        # Convert to PIL Image if needed
        if isinstance(image, (str, Path)):
            pil_image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            if image.dtype == np.uint8:
                pil_image = Image.fromarray(image)
            else:
                # Assume float array in [0, 1] range
                pil_image = Image.fromarray((image * 255).astype(np.uint8))
        elif isinstance(image, torch.Tensor):
            # Convert tensor to PIL Image
            if image.dim() == 3:  # [C, H, W]
                image_np = image.permute(1, 2, 0).cpu().numpy()
            else:  # [H, W, C]
                image_np = image.cpu().numpy()
            
            if image_np.dtype == np.float32 or image_np.dtype == np.float64:
                image_np = (image_np * 255).astype(np.uint8)
            
            pil_image = Image.fromarray(image_np)
        elif isinstance(image, Image.Image):
            pil_image = image.convert('RGB')
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Apply transforms
        if self.transform:
            tensor = self.transform(pil_image)
        else:
            # Basic transform if no transform provided
            tensor = torch.from_numpy(np.array(pil_image)).permute(2, 0, 1).float() / 255.0
        
        # Add batch dimension and move to device
        tensor = tensor.unsqueeze(0).to(self.device)
        
        return tensor
    
    def visualize_prediction(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        save_path: Optional[str] = None,
        show_probabilities: bool = False,
        alpha: float = 0.6
    ) -> plt.Figure:
        """
        Visualize prediction with overlay on original image.
        
        Args:
            image: Input image
            save_path: Optional path to save visualization
            show_probabilities: Whether to show class probabilities
            alpha: Transparency for overlay
            
        Returns:
            Matplotlib figure
        """
        # Get prediction
        if show_probabilities:
            predicted_mask, probabilities = self.predict_image(
                image, return_probabilities=True
            )
        else:
            predicted_mask = self.predict_image(image)
            probabilities = None
        
        # Load original image for visualization
        if isinstance(image, (str, Path)):
            original_image = Image.open(image).convert('RGB')
        elif isinstance(image, Image.Image):
            original_image = image.convert('RGB')
        elif isinstance(image, np.ndarray):
            if image.dtype == np.uint8:
                original_image = Image.fromarray(image)
            else:
                original_image = Image.fromarray((image * 255).astype(np.uint8))
        else:
            raise ValueError(f"Unsupported image type for visualization: {type(image)}")
        
        # Resize original image to match prediction
        original_image = original_image.resize(predicted_mask.shape[::-1], Image.LANCZOS)
        original_array = np.array(original_image)
        
        # Create figure
        if show_probabilities and probabilities is not None:
            fig, axes = plt.subplots(2, self.num_classes + 1, 
                                   figsize=(4 * (self.num_classes + 1), 8))
            
            # Top row: original and prediction
            axes[0, 0].imshow(original_array)
            axes[0, 0].set_title('Original Image')
            axes[0, 0].axis('off')
            
            # Create colored mask
            colored_mask = self._create_colored_mask(predicted_mask)
            overlay = (1 - alpha) * original_array + alpha * colored_mask
            
            axes[0, 1].imshow(overlay.astype(np.uint8))
            axes[0, 1].set_title('Prediction Overlay')
            axes[0, 1].axis('off')
            
            # Hide unused axes in top row
            for i in range(2, self.num_classes + 1):
                axes[0, i].axis('off')
            
            # Bottom row: class probabilities
            for class_idx in range(self.num_classes):
                prob_map = probabilities[class_idx]
                im = axes[1, class_idx].imshow(prob_map, cmap='hot', vmin=0, vmax=1)
                axes[1, class_idx].set_title(f'{self.class_names[class_idx]} Probability')
                axes[1, class_idx].axis('off')
                plt.colorbar(im, ax=axes[1, class_idx], fraction=0.046, pad=0.04)
            
            # Hide unused axis in bottom row
            if self.num_classes < len(axes[1]):
                axes[1, -1].axis('off')
        
        else:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Original image
            axes[0].imshow(original_array)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # Prediction mask
            axes[1].imshow(predicted_mask, cmap='tab10', vmin=0, vmax=self.num_classes-1)
            axes[1].set_title('Predicted Mask')
            axes[1].axis('off')
            
            # Overlay
            colored_mask = self._create_colored_mask(predicted_mask)
            overlay = (1 - alpha) * original_array + alpha * colored_mask
            
            axes[2].imshow(overlay.astype(np.uint8))
            axes[2].set_title('Prediction Overlay')
            axes[2].axis('off')
        
        plt.tight_layout()
        
        # Save if requested
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            self.logger.info(f"Visualization saved to {save_path}")
        
        return fig
    
    def _create_colored_mask(self, mask: np.ndarray) -> np.ndarray:
        """Create colored visualization of segmentation mask."""
        # Define colors for each class
        colors = [
            [0, 0, 0],        # Background - black
            [255, 0, 0],      # Class 1 - red
            [0, 255, 0],      # Class 2 - green
            [0, 0, 255],      # Class 3 - blue
            [255, 255, 0],    # Class 4 - yellow
            [255, 0, 255],    # Class 5 - magenta
            [0, 255, 255],    # Class 6 - cyan
        ]
        
        # Extend colors if needed
        while len(colors) < self.num_classes:
            colors.append([np.random.randint(0, 256) for _ in range(3)])
        
        # Create colored mask
        colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
        
        for class_idx in range(self.num_classes):
            mask_indices = (mask == class_idx)
            colored_mask[mask_indices] = colors[class_idx]
        
        return colored_mask
    
    def save_prediction(
        self,
        image: Union[str, Path, Image.Image, np.ndarray],
        output_path: str,
        save_overlay: bool = True,
        save_mask: bool = True,
        save_probabilities: bool = False
    ):
        """
        Save prediction results to files.
        
        Args:
            image: Input image
            output_path: Base output path (without extension)
            save_overlay: Whether to save overlay visualization
            save_mask: Whether to save raw mask
            save_probabilities: Whether to save probability maps
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get prediction
        if save_probabilities:
            predicted_mask, probabilities = self.predict_image(
                image, return_probabilities=True
            )
        else:
            predicted_mask = self.predict_image(image)
            probabilities = None
        
        # Save raw mask
        if save_mask:
            mask_path = output_path.with_suffix('.png')
            mask_image = Image.fromarray(predicted_mask.astype(np.uint8))
            mask_image.save(mask_path)
            self.logger.info(f"Saved mask to {mask_path}")
        
        # Save overlay
        if save_overlay:
            overlay_path = output_path.with_name(f"{output_path.stem}_overlay.png")
            fig = self.visualize_prediction(image, save_path=str(overlay_path))
            plt.close(fig)
        
        # Save probabilities
        if save_probabilities and probabilities is not None:
            prob_path = output_path.with_suffix('.npz')
            np.savez_compressed(prob_path, probabilities=probabilities)
            self.logger.info(f"Saved probabilities to {prob_path}")
    
    def load_model_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to model checkpoint
            
        Returns:
            Checkpoint metadata
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.logger.info(f"Loaded model from {checkpoint_path}")
        
        return checkpoint