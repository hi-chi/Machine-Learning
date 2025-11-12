import pickle


class History:
    def __init__(self):
        self.train_history = {}
        self.valid_history = {}

        self.past_size = 0

    def load_history(self, folder):

        path = folder + '/metric/history.pkl'
        with open(path, 'rb') as f:
            history = pickle.load(f)
            self.train_history = history['train']
            self.valid_history = history['valid']

            self.past_size = len(self.train_history)

    def save_history(self, folder):

        path = folder + '/history.pkl'
        history = {'train': self.train_history, 'valid': self.valid_history}
        with open(path, 'wb') as f:
            pickle.dump(history, f)
