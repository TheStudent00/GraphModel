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
        default single layer to project input into embedding space
    """
    def __init__(self):
        self.input_shape = None
        self.output_shape = None
        self.feature_layers = None
        self.space_complexity = None
        self.time_complexity = None

    def __call__(self, input):
        pass


class Logistics:
    """
    """
    def __init__(self, requester_dict, requester_contact, requester_content_location, timestamp):
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
                    optimization_request: {0,1}
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
        recency, compression curvature, and space complexity are learnable
        includes tracing for optimization
    """
    def __init__(self):
        self.compressor_core = Core()
        self.uncompressed_memories = None
        self.compressed_memories = None
        self.space_complexity = None
        self.space_complexity_limit = None
        
    def store(self, new_memory):
        """
            total_size = new_memory.shape + self.uncompress_memories.shape + self.compressed_memories.shape
            if total_size > self.space_complexity_limit:
                compressed_old_memories, uncompressed_old_memories = self.compress()
                self.update_uncompressed(new_memory, uncompressed_old_memories)
                self.update_compressed(compressed_old_memories)
            else:
                self.update_uncompressed(new_memory, uncompressed_old_memories)
        """
    
    def update_uncompressed(self):
        pass

    def update_compressed(self):
        pass

    def compress(self):
        pass


class Module:
    """
        design notes:
            level 0 (accepting known input/output data):
                cores project data into an embedded space
            aside from level 0, core input/output have a learnable standard size
                core_input.shape: (num_rows, num_cols)
                    num_rows: not directly constrained
                    num_cols: same as feature filter size
                core_output.shape: (num_rows, num_cols)
                    num_rows: same as input num_rows
                    num_cols: standard embedding size

        state_core:
            state of the module in relation to of modules

        state_update_core:
            updates according the embedded mixing of attributes
            initiates requests for updating

        context_core:
            more dynamic/eager structure than `state`
            updates per context (incoming) request
                mixes input with context memory

        service_core:
            applies feature filters to requester input
                independent of responder and other modules attributes
        
    """
    def __init__(self):
        self.id = None
        self.level = None
        self.contacts = []
        self.complexity = Complexity()

        self.logistics_queue = []
        self.logistics_memory = Memory()

        self.state = None
        self.state_core = Core()
        self.state_memory = Memory()
        self.state_timestamp = None
        self.state_update_core = Core()

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
    
