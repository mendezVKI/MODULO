import numpy as np

def load_or_make(filename):
    def decorator(func):
        def wraps(*args, **kwargs):
            try:
                with open(filename, 'r') as f:
                    return np.load(f)
            except Exception:
                data = func(*args, **kwargs)
                with open(filename, 'w') as out:
                    np.dump(data, out)
                return data
        return wraps
    return decorator