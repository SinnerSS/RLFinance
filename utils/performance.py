
import time
from functools import wraps

def monitor_time(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        qualname = func.__qualname__
        if '.' in qualname:
            cls_name, func_name = qualname.rsplit('.', 1)
            print(f"{cls_name}.{func_name} took {end - start:.6f} seconds")
        else:
            print(f"{func.__name__} took {end - start:.6f} seconds")
        return result
    return wrapper
