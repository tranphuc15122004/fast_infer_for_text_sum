class LazyBlockMaskCache:
    """Cache block masks on demand for arbitrarily long KV sequences.

    The exact-length API is used by the attention path because FlexAttention
    requires the BlockMask KV dimension to match the tensor exactly.  The
    integer-index API remains for older FAFO call sites and preserves its
    block-rounded behavior.
    """

    def __init__(self, factory, block_size: int = 128):
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self._factory = factory
        self._block_size = block_size
        self._cache = {}

    def for_length(self, kv_len: int):
        """Return a mask whose KV dimension is exactly ``kv_len``.

        FlexAttention rejects a BlockMask that is larger than the actual KV
        tensor.  The decoder can have padding/cache-manager bookkeeping that
        leaves KV lengths which are not multiples of ``block_size``; those
        lengths must not be rounded up.
        """
        if not isinstance(kv_len, int):
            raise TypeError("kv_len must be an integer")
        if kv_len <= 0:
            raise ValueError("kv_len must be positive")
        if kv_len not in self._cache:
            self._cache[kv_len] = self._factory(kv_len)
        return self._cache[kv_len]

    def __getitem__(self, index: int):
        if not isinstance(index, int):
            raise TypeError("mask index must be an integer")
        if index < 0:
            raise IndexError("mask index must be non-negative")
        if index not in self._cache:
            kv_len = (index + 1) * self._block_size
            self._cache[index] = self._factory(kv_len)
        return self._cache[index]
