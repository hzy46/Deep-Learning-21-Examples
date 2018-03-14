import os
import time
import pprint
import tensorflow as tf
from six.moves import range
from logging import getLogger

logger = getLogger(__name__)
pp = pprint.PrettyPrinter().pprint

def get_model_dir(config, exceptions=None):
  attrs = config.__dict__['__flags']
  pp(attrs)

  keys = list(attrs.keys())
  keys.sort()
  keys.remove('env_name')
  keys = ['env_name'] + keys

  names = [config.env_name]
  for key in keys:
    # Only use useful flags
    if key not in exceptions:
      names.append("%s=%s" % (key, ",".join([str(i) for i in attrs[key]])
          if type(attrs[key]) == list else attrs[key]))
  return os.path.join('checkpoints', *names) + '/'

def timeit(f):
  def timed(*args, **kwargs):
    start_time = time.time()
    result = f(*args, **kwargs)
    end_time = time.time()

    logger.info("%s : %2.2f sec" % (f.__name__, end_time - start_time))
    return result
  return timed
