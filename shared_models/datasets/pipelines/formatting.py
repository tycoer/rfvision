from mmdet3d.datasets.builder import PIPELINES

@PIPELINES.register_module()
class RenameKeys:
    """Rename the keys.

    Args:
        key_pairs (Sequence[tuple]): Required keys to be renamed.
            If a tuple (key_src, key_tgt) is given as an element,
            the item retrieved by key_src will be renamed as key_tgt.
    """

    def __init__(self, key_pairs):
        self.key_pairs = key_pairs

    def __call__(self, results):
        """Rename keys."""
        for key_pair in self.key_pairs:
            assert len(key_pair) == 2
            key_src, key_tgt = key_pair
            results[key_tgt] = results.pop(key_src)
        return results