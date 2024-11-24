from user_settings_singleton.userSettings import userSettings

class UserSettingsSingleton:
    _instance = None
    @classmethod
    def get_instance(cls):
        if not cls._instance:
            cls._instance = userSettings()
        return cls._instance