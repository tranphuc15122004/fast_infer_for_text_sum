class LazyBlockMaskCache:
    """Cache block masks on demand for arbitrarily long KV sequences.

    FAFO indexes masks by ``key_length // block_size``.  Prebuilding a fixed
    list only up to 6.5k tokens made long prompts fail with ``IndexError``.
    The factory is called once for each required rounded-up KV length.
    """

    def __init__(self, factory, block_size: int = 128):
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self._factory = factory
        self._block_size = block_size
        self._cache = {}

    def __getitem__(self, index: int):
        if not isinstance(index, int):
            raise TypeError("mask index must be an integer")
        if index < 0:
            raise IndexError("mask index must be non-negative")
        if index not in self._cache:
            kv_len = (index + 1) * self._block_size
            self._cache[index] = self._factory(kv_len)
        return self._cache[index]
