from strategy.strategy import Strategy
from user_settings_singleton.UserSettingsSingleton import UserSettingsSingleton
class QuickStrategy(Strategy):
    def configure_context(self):
        settings = UserSettingsSingleton.get_instance()
        settings.update_settings('SVM', False, False, False, False)
        
class VerboseStrategy(Strategy):
    def configure_context(self):
        settings = UserSettingsSingleton.get_instance()
        settings.update_settings(["HGBC", "SVM", "NB", "KNN", "CB"], True, True, True, True)
        
class NoiseRemovalStrategy(Strategy):
    def configure_context(self):
        settings = UserSettingsSingleton.get_instance()
        settings.update_settings(["HGBC", "SVM", "NB"], False, True, False, False)
                                    
class TranslateStrategy(Strategy):
    def configure_context(self):
        settings = UserSettingsSingleton.get_instance()
        settings.update_settings(["HGBC", "SVM", "NB"], True, False, False, False)
        
class HighPerformanceStrategy(Strategy):
    def configure_context(self):
        settings = UserSettingsSingleton.get_instance()
        settings.update_settings(["SVM", "KNN"], False, True, False, True)