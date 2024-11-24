class MenuInvoker:
    def __init__(self):
        self.commands = {}

    def register_command(self,option, command):
        self.commands[option] = command

    def execute_command(self, option):
        command = self.commands.get(option)
        if command:
            command.execute()
        else:
            print(f"Command {option} not recognized.")