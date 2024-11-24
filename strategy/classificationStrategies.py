from strategy.strategy import Strategy
from user_settings_singleton.Settings_manager import SettingsManager

class QuickStrategy(Strategy):
    def __init__(self, settings_manager: SettingsManager):
        self.settings_manager = settings_manager

    def configure_context(self):
        self.settings_manager.update_settings('SVM', False, False, False, False)

class VerboseStrategy(Strategy):
    def __init__(self, settings_manager: SettingsManager):
        self.settings_manager = settings_manager

    def configure_context(self):
        self.settings_manager.update_settings(["HGBC", "SVM", "NB", "KNN", "CB"], True, True, True, True)
        
class NoiseRemovalStrategy(Strategy):
    def __init__(self, settings_manager: SettingsManager):
        self.settings_manager = settings_manager

    def configure_context(self):
        self.settings_manager.update_settings(["HGBC", "SVM", "NB"], False, True, False, False)
                                    
class TranslateStrategy(Strategy):
    def __init__(self, settings_manager: SettingsManager):
        self.settings_manager = settings_manager

    def configure_context(self):
        self.settings_manager.update_settings(["HGBC", "SVM", "NB"], True, False, False, False)
        
class HighPerformanceStrategy(Strategy):
    def __init__(self, settings_manager: SettingsManager):
        self.settings_manager = settings_manager

    def configure_context(self):
        self.settings_manager.update_settings(["SVM", "KNN"], False, True, False, True)