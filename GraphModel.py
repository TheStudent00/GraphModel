from collections.abc import Iterable

class Complexity:
    """
    """
    def __init__(self, space_tokens, time_tokens):
        self.time_complexity = None
        self.time_tokens = space_tokens

        self.space_complexity = None
        self.space_tokens = time_tokens


class SplineLUT:

    def __init__(self):
        self.squared_LUT = None
        self.cubed_LUT = None

    def __call__(self, x_key):
        squared_x_value = self.squared_LUT[x_key]
        cubed_x_value = self.cubed_LUT[x_key]
        return (squared_x_value, cubed_x_value)


class Feature:
    """
        to resolve input/output size mismatching the following represents the
        monotonic curve (defined smoothly with splines) of a large feature vector
        with values sorted from largest to smallest.

        the concept leverages the idea of sorting parameters and using a learned
        permutation matrix (dense while learning, sparse in use) on the input
        vectors to correctly map elements to their optimal parameters.

        it also leverages the idea of a pseudo-continuous feature curvature and
        enabling expandability into any arbitrary size of input vector. there are
        practical limitations and the SplineLUT attempts to reduce the need to
        constantly compute squares/cubes on the fly, especially when the anticipated
        resolutions are within reasonable bounds. the key-value returned are
        then multiplied by their respective spline constants.

        it is possible that resulting discrete sorted-features could be cached
        to increase efficiency -- especially if the number of input sizes is
        mostly static.
    """
    def __init__(self):
        self.monotonic_splines = []

    def __call__(self):
        pass


class Layer:
    """
    """
    def __init__(self, num_features=None):
        self.num_features = num_features
        self.features: list[Feature] = None


class Core:
    """
        default single layer to project input into embedding space
        default assumption is to not mix the embedded vectors within the core
            expandability of layers of features (parallel and downstream) does enable mixing similar to attention embedded patch mixing
            (( softmax(QK)*V ))
            attention-like mixing can also occur through Core-to-Core or Module-to-Module communication
    """
    def __init__(self):
        # self.input_shape = None
        # self.output_shape = None
        self.feature_layers: list[Layer] = None
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
            revision:
                * compression curvature (rate of compression) can be done implicitly through the learnability of space-time complexity of memory types (compressed and uncompressed)
                * the compressor/decompressor are meant to act as a sort of autoencoder to model the relevant elements of the embedded vectors
        includes tracing for optimization
    """
    def __init__(self):
        self.compressor_core: Core = None
        self.decompressor_core: Core = None
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


class Contact:

    def __init__(self, module_id, module_level):
        self.module_id = module_id
        self.module_level = module_level

        self.request_sparse_permutation = None
        self.response_sparse_permutation = None

        self.request_dense_permutation = None
        self.response_dense_permutation = None

    def initialize_permutation(self):
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
        self.contacts: list[Contact] = None
        self.complexity: Complexity = None

        self.logistics_queue: list[Logistics] = None
        self.logistics_memory: Memory = None

        self.state = None
        self.state_core: Core = None
        self.state_memory: Memory = None
        self.state_timestamp = None
        self.state_update_core: Core = None

        self.context = None
        self.context_core: Core = None
        self.context_memory: Memory = None
        self.context_timestamp = None

        self.service_core: Core = None
        self.service_memory: Memory = None
        self.service_timestamp = None


class MindsEye(Module):
    def __init__(self):
        self.architecture_core: Core = None
        self.architecture_memory: Memory = None
        self.architecture_timestamp = None


class GraphModel:
    """
    """
    def __init__(self,
                 training_data,
                 test_data,
                 training_space_tokens,
                 training_time_tokens,
                 operational_space_tokens,
                 operational_time_tokens,
                 modules=None):
        self.modules = [Module()] if modules is None else modules
        self.training_data = training_data
        self.test_data = test_data

        self.minds_eye: MindsEye = None
        self.training_complexity = Complexity(training_space_tokens, training_time_tokens)
        self.operational_complexity = Complexity(operational_space_tokens, operational_time_tokens)

    def __call__(self):
        pass
