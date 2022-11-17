"""Functions that allow the players to communicate, they are common to all users"""
import pickle
import threading


class Communication:
    """Send and receive messages"""
    def __init__(self, length, header=10000):
        """The maximum length of the messages that are to be sent"""
        self.length = length
        self.header = header

    def send(self, receiver, message):
        """Send a message"""
        message_s = pickle.dumps(message, protocol=pickle.HIGHEST_PROTOCOL)
        message_s = bytes(f'{len(message_s):<{self.header}}', "utf-8") + message_s
        receiver.send(message_s)

    def get(self, sender):
        """Receive a message in packets"""
        full_message = b''
        new_message = True
        while True:
            message = sender.recv(self.length)
            if new_message:
                message_len = int(message[:self.header])
                new_message = False
            full_message += message

            if len(full_message) - self.header == message_len:
                message = pickle.loads(full_message[self.header:])
                return message

    def broadcast(self, receivers, message):
        """Broadcast the same message to everyone"""
        threads = [threading.Thread(target=self.send, args=[receiver, message]) for receiver in receivers]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    def send_shares(self, receivers, shares):
        """Broadcast one message for every player, not necessarily the same message"""
        threads = [threading.Thread(target=self.send, args=[receivers[i], shares[i]]) for i in range(len(receivers))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()