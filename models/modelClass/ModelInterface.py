from abc import ABC, abstractmethod

class IModel(ABC):
    @abstractmethod
    def categorize(self, subject: str, email: str):
        pass
