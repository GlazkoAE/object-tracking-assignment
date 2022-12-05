from .feature_extractor import extract_features


class FeatureTracker:
    def __init__(self, img, bbox, label):

        self.time_since_last_update = 0
        self.extract_features = extract_features
        self.id = label
        self.features = []
        self.update(img, bbox)

    def update(self, img, bbox):
        # if we have called predict twice in a row , it will set update flag as 0
        self.features = self.extract_features(img, bbox)

    @property
    def current_state(self):
        """
        returns current features
        """
        return self.features
