from abc import ABC, abstractmethod
class SettingsManager(ABC):
    @abstractmethod
    def update_settings(self, *args, **kwargs):
        pass