__version__ = '0.0.1'


class Notes:
    def __init__(self):
        self.show = True

    def print_notes(self):
        if self.show:
            print(f'''
 version = {__version__}
            ''')
            self.close()

    def close(self):
        self.show = False


bw_notes = Notes()
