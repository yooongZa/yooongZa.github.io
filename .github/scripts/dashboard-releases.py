"""Export only published release metadata from the four public app repositories."""
import datetime
import json
import os
import urllib.request

repositories = {}
for repo in ['imgest-releases', 'DiskOUT', 'tabledown', 'xbar']:
    releases = {}
    for page in range(1, 11):
        request = urllib.request.Request(
            f'https://api.github.com/repos/yooongZa/{repo}/releases?per_page=100&page={page}',
            headers={'Accept': 'application/vnd.github+json',
                     'Authorization': 'Bearer ' + os.environ['GH_TOKEN'],
                     'X-GitHub-Api-Version': '2022-11-28',
                     'User-Agent': 'LIMOD-dashboard-metrics'})
        with urllib.request.urlopen(request, timeout=20) as response:
            data = json.load(response)
            has_next = 'rel="next"' in response.headers.get('Link', '')
        for release in data:
            if release['draft'] or not release['published_at']:
                continue
            releases[release['id']] = {
                key: release[key] for key in ['id', 'tag_name', 'published_at', 'draft', 'prerelease']}
            releases[release['id']]['assets'] = [
                {key: asset[key] for key in ['name', 'download_count']}
                for asset in release['assets']]
        if not has_next:
            break
    else:
        raise RuntimeError('Incomplete pagination: ' + repo)
    repositories[repo] = list(releases.values())

print(json.dumps({'generatedAt': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'repositories': repositories}, separators=(',', ':')))
