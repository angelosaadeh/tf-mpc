import mpc.network.network as network
import mpc.protocols.protocols as protocols



class Run(protocols.Protocols):
    """Protocols for MPC"""
    def __init__(self, host, port, ip, identity, mod, precision, length):
        """Identity is a string equals to alice or bob
        host is binary: True or False"""
        if host:
            _ = network.serve(ip, port)
            partner = network.accept_connections(_)
        else:
            partner = network.connect(ip, port)
        super().__init__(identity, partner, mod, precision, length)
