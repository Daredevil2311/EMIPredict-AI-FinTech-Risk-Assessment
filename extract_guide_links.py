import urllib.request
import re

docs = {
    'ML_Guide': '1-LEvCmnH6GRmU_ysHD7QwgdR1tCKvw5fLqAkJ-w3gQ8',
    'ML_USVL': '1JBdCEW6kKcJaPkzDO-Oo_pq0-zfMTabAUfEoymC6I0g',
    'EDA_Guide': '1kbAZAtGhABWK5S7IAWXs3qiQ8PZybqCpv67S0DwppJ0',
    'Streamlit': '1aR3pUZFlCi8gicpF6aPHPESeFdOtGMlfob5PckresZk'
}

for name, doc_id in docs.items():
    url = f'https://docs.google.com/document/d/{doc_id}/export?format=html'
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        with urllib.request.urlopen(req) as resp:
            html = resp.read().decode('utf-8')
            links = re.findall(r'href=["\'](https?://[^"\']+)["\']', html)
            print(f'=== {name} ===')
            for l in links:
                if 'google.com/url?q=' in l:
                    actual = l.split('google.com/url?q=')[1].split('&')[0]
                    print(' ', actual)
                else:
                    print(' ', l)
    except Exception as e:
        print(f'{name} error: {e}')
