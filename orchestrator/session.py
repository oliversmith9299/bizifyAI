class Session:
    def __init__(self, user_input):
        self.state= "START"
        self.user_input= user_input
        self.data= {}

    def save (self, key, value):
        self.data[key]= value