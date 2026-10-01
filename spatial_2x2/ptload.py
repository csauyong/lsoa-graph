"""Minimal torch-free loader for torch.save zip files (tensors -> numpy)."""
import zipfile, pickle, numpy as np

_DT = {"LongStorage": np.int64, "IntStorage": np.int32, "FloatStorage": np.float32,
       "DoubleStorage": np.float64, "BoolStorage": np.bool_, "ByteStorage": np.uint8,
       "HalfStorage": np.float16, "ShortStorage": np.int16, "CharStorage": np.int8}

class _StorageType:
    def __init__(self, name): self.name = name

def _rebuild_tensor_v2(storage, offset, size, stride, *args):
    arr, = [storage]
    if len(size) == 0:
        return arr[offset]
    itemsize = arr.itemsize
    return np.lib.stride_tricks.as_strided(arr[offset:], shape=tuple(size),
                                           strides=tuple(s * itemsize for s in stride)).copy()

def load(path):
    z = zipfile.ZipFile(path)
    prefix = z.namelist()[0].split("/")[0]
    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if mod == "torch._utils" and name == "_rebuild_tensor_v2":
                return _rebuild_tensor_v2
            if mod == "torch" and name.endswith("Storage"):
                return _StorageType(name)
            if mod == "collections" and name == "OrderedDict":
                import collections; return collections.OrderedDict
            if mod.startswith("torch"):
                return lambda *a, **k: None
            return super().find_class(mod, name)
        def persistent_load(self, pid):
            typename, stype, key, location, numel = pid
            dt = _DT[stype.name]
            buf = z.read(f"{prefix}/data/{key}")
            return np.frombuffer(buf, dtype=dt)
    return U(z.open(f"{prefix}/data.pkl")).load()
