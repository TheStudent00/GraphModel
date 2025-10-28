class Complexity:
    """
    """
    def __init__(self):
        self.time_complexity = None
        self.time_complexity_tokens = None

        self.space_complexity = None
        self.space_complexity_tokens = None


class Core:
    """
    """
    def __init__(self):
        self.embedding_size = None
        self.features = None
        self.space_complexity = None
        self.time_complexity = None

    def __call__(self):
        pass


class Logistics:
    """
    """
    def __init__(self, requester_dict, requester_contact, requester_content_location):
        self.requester_dict = requester_dict
        self.requester_contact = requester_contact
        self.requester_content_location = requester_content_location
        self.responder_dict = {}

    def receive_request(self, responder_contact_id):
        """
            request:
                    state_request: {0,1}
                        0: not requested
                        1: requested
                    context_request: {0,1}
                        0: not requested
                        1: requested
                    service_request: {0,1}
                        0: not requested
                        1: requested
            requester_content_location
        """
        requests = self.requester_dict[responder_contact_id]
        requester_content_location = self.request_content_location
        return state_request, context_request, requester_content_location

    def send_response(self, responder_contact_id, responder_content_location):
        self.responder_dict[responder_contact_id] = responder_content_location


class Memory:
    """
        storage of memories in an embedded space
        recent memories are stored verbatim
        older memories become mixed/compressed
        compression (rate) curvature increases exponentially
            from recent to older
        compression curvature and space complexity are learnable
    """


class Module:
    """
        state:
            state of the module in relation to of modules
            updated with 
        context:
            more dynamic structure than `state`
        service:
            applies
        
    """
    def __init__(self):
        self.id = None
        self.level = None
        self.contacts = []
        self.complexity = Complexity()

        self.logistics_queue = []

        self.state = None
        self.state_core = Core()
        self.state_memory = Memory()
        self.state_timestamp = None

        self.context = None
        self.context_core = Core()
        self.context_memory = Memory()
        self.context_timestamp = None
        
        self.service_core = Core()
        self.service_memory = Memory()
        self.service_timestamp = None


class GraphModel:
    """
    """
    def __init__(self, modules, training_data, test_data):
        self.modules = modules
        self.training_data = training_data
        self.test_data = test_data
    
