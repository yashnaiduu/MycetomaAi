import numpy as np

class MacenkoNormalizer:
    """Macenko stain normalization."""
    def __init__(self, Io=240, alpha=1, beta=0.15):
        self.Io = Io
        self.alpha = alpha
        self.beta = beta
        self.HERef = np.array([[0.5626, 0.2159],
                               [0.7201, 0.8012],
                               [0.4062, 0.5581]])
        self.maxCRef = np.array([1.9705, 1.0308])

    def fit(self, target_image):
        """Fit to reference image."""
        pass

    def normalize(self, img):
        """Normalize stain colors."""
        h, w, c = img.shape
        img = img.reshape((-1, 3))
        
        OD = -np.log((img.astype(float) + 1) / self.Io)
        
        ODhat = OD[~np.any(OD < self.beta, axis=1)]
        
        if ODhat.size == 0:
            return img.reshape((h, w, c)).astype(np.uint8)

        eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
        
        eigvecs = eigvecs[:, [1, 2]]
        
        if eigvecs[0, 0] < 0: eigvecs[:, 0] *= -1
        if eigvecs[0, 1] < 0: eigvecs[:, 1] *= -1

        That = np.dot(ODhat, eigvecs)
        
        phi = np.arctan2(That[:, 1], That[:, 0])
        
        minPhi = np.percentile(phi, self.alpha)
        maxPhi = np.percentile(phi, 100 - self.alpha)
        
        vMin = np.dot(eigvecs, np.array([np.cos(minPhi), np.sin(minPhi)]))
        vMax = np.dot(eigvecs, np.array([np.cos(maxPhi), np.sin(maxPhi)]))
        
        if vMin[0] > vMax[0]:
            HE = np.array([vMin, vMax]).T
        else:
            HE = np.array([vMax, vMin]).T
            
        C = np.linalg.lstsq(HE, OD.T, rcond=None)[0]
        
        maxC = np.percentile(C, 99, axis=1)
        maxC[maxC == 0] = 1e-6
        
        C = C * (self.maxCRef / maxC)[:, np.newaxis]
        
        Inorm = self.Io * np.exp(-np.dot(self.HERef, C))
        Inorm = Inorm.T.reshape((h, w, c))
        Inorm = np.clip(Inorm, 0, 255).astype(np.uint8)
        
        return Inorm

def apply_macenko(image: np.ndarray) -> np.ndarray:
    """Wrapper for Macenko normalization."""
    normalizer = MacenkoNormalizer()
    try:
        return normalizer.normalize(image)
    except Exception as e:
        return image
