from strategy.strategy import Strategy
from user_settings_singleton.Settings_manager import SettingsManager

class QuickStrategy(Strategy):
    def __init__(self, settings_manager: SettingsManager):
        self.settings_manager = settings_manager

    def configure_context(self):
        self.settings_manager.update_settings('SVM', False, False)

class VerboseStrategy(Strategy):
    def __init__(self, settings_manager: SettingsManager):
        self.settings_manager = settings_manager

    def configure_context(self):
        self.settings_manager.update_settings(["HGBC", "SVM", "NB", "KNN", "CB"], True, True)
        
class NoiseRemovalStrategy(Strategy):
    def __init__(self, settings_manager: SettingsManager):
        self.settings_manager = settings_manager

    def configure_context(self):
        self.settings_manager.update_settings(["HGBC", "SVM", "NB"], False, True)
                                    
class TranslateStrategy(Strategy):
    def __init__(self, settings_manager: SettingsManager):
        self.settings_manager = settings_manager

    def configure_context(self):
        self.settings_manager.update_settings(["HGBC", "SVM", "NB"], True, False)
        
class HighPerformanceStrategy(Strategy):
    def __init__(self, settings_manager: SettingsManager):
        self.settings_manager = settings_manager

    def configure_context(self):
        self.settings_manager.update_settings(["SVM", "KNN"], False, True)