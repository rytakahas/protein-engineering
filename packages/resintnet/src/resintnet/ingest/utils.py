
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def make_session(total=3, backoff=0.5, status=(500,502,503,504)):
    s = requests.Session()
    r = Retry(total=total, read=total, connect=total, backoff_factor=backoff, status_forcelist=status)
    s.mount("https://", HTTPAdapter(max_retries=r))
    s.mount("http://", HTTPAdapter(max_retries=r))
    s.headers.update({"User-Agent": "resintnet/0.1"})
    return s
