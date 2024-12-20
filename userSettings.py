class userSettings:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super().__new__(cls, *args, **kwargs)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self.ml_models = []
        self.translate_text = False
        self.remove_noise = False
        self.explainable = False

    def update_settings(self, ml_model=None, translate_text=None, remove_noise=None, verbose=None, explainable=None):
        if ml_model is not None:
            self.ml_models = ml_model if isinstance(ml_model, list) else [ml_model]
        if translate_text is not None:
            self.translate_text = translate_text
        if remove_noise is not None:
            self.remove_noise = remove_noise
        if explainable is not None:
            self.explainable = explainable

    def __str__(self):
        """Return the current settings as a string for debugging."""
        return (f"ML Models: {self.ml_models}, "
                f"Translate Text: {self.translate_text}, "
                f"Noise Removal: {self.remove_noise}, "
                f"Explainable: {self.explainable}")
