import numpy as np

class MacenkoNormalizer:
    """
    Macenko Stain Normalization
    Reference:
    Macenko, M., et al. (2009). A method for normalizing histology slides for quantitative analysis.
    In IEEE International Symposium on Biomedical Imaging: From Nano to Macro.
    """
    def __init__(self, Io=240, alpha=1, beta=0.15):
        self.Io = Io
        self.alpha = alpha
        self.beta = beta
        self.HERef = np.array([[0.5626, 0.2159],
                               [0.7201, 0.8012],
                               [0.4062, 0.5581]])
        self.maxCRef = np.array([1.9705, 1.0308])

    def fit(self, target_image):
        """Fit the normalizer to a target reference image (optional). 
        By default, uses fixed HERef."""
        pass

    def normalize(self, img):
        """
        Normalize the stain of an image.
        Args:
            img: RGB image (H, W, 3) in uint8 format.
        Outputs:
            Normalized RGB image.
        """
        h, w, c = img.shape
        img = img.reshape((-1, 3))
        
        # Calculate optical density
        OD = -np.log((img.astype(float) + 1) / self.Io)
        
        # Remove transparent pixels
        ODhat = OD[~np.any(OD < self.beta, axis=1)]
        
        if ODhat.size == 0:
            return img.reshape((h, w, c)).astype(np.uint8)

        # Calculate eigenvectors
        eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
        
        # Select the two largest eigenvectors
        eigvecs = eigvecs[:, [1, 2]]
        
        # Ensure vectors are pointing in the right direction
        if eigvecs[0, 0] < 0: eigvecs[:, 0] *= -1
        if eigvecs[0, 1] < 0: eigvecs[:, 1] *= -1

        # Project on the plane spanned by the eigenvectors
        That = np.dot(ODhat, eigvecs)
        
        # Calculate angle of each point to the first eigenvector
        phi = np.arctan2(That[:, 1], That[:, 0])
        
        # Find robust extremes
        minPhi = np.percentile(phi, self.alpha)
        maxPhi = np.percentile(phi, 100 - self.alpha)
        
        # Convert back to OD space
        vMin = np.dot(eigvecs, np.array([np.cos(minPhi), np.sin(minPhi)]))
        vMax = np.dot(eigvecs, np.array([np.cos(maxPhi), np.sin(maxPhi)]))
        
        # Order vectors: H has larger x component than E
        if vMin[0] > vMax[0]:
            HE = np.array([vMin, vMax]).T
        else:
            HE = np.array([vMax, vMin]).T
            
        # Calculate concentrations
        C = np.linalg.lstsq(HE, OD.T, rcond=None)[0]
        
        # Normalize concentrations
        maxC = np.percentile(C, 99, axis=1)
        # Avoid division by zero
        maxC[maxC == 0] = 1e-6
        
        # Apply reference concentrations
        C = C * (self.maxCRef / maxC)[:, np.newaxis]
        
        # Recreate image
        Inorm = self.Io * np.exp(-np.dot(self.HERef, C))
        Inorm = Inorm.T.reshape((h, w, c))
        Inorm = np.clip(Inorm, 0, 255).astype(np.uint8)
        
        return Inorm

def apply_macenko(image: np.ndarray) -> np.ndarray:
    """Wrapper function for Macenko normalization."""
    normalizer = MacenkoNormalizer()
    try:
        return normalizer.normalize(image)
    except Exception as e:
        # Fallback to original image if normalization fails (e.g., pure background patch)
        return image
