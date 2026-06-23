from util import get_username
from df import add_col
from gcp import get_adctoken
import os
import subprocess
import requests
import pandas as pd

class Repo:
    def __init__(self,name):
        self.name=name
        self.path=fr"{os.environ['HOME']}/Documents/{name}"
        self.msg=''
        self.owner = "<<name>>"
        self.cur_branch()
    
    def run(self,cmd):
        # print('cmd=',cmd)
        result = subprocess.run(cmd, cwd=self.path, capture_output=True, text=True)
        # print('result=',result)
        return result

    def cur_branch(self):    
        cmd = ["git", "rev-parse", "--abbrev-ref", "HEAD"]
        result = subprocess.run(cmd, cwd=self.path, capture_output=True, text=True)
        branch = result.stdout.strip()
        self.branch = branch
        print('cur branch is',branch)
        return branch

    def change_branch(self,to):
        self.run(["git", "checkout", to])
        self.cur_branch()
        self.run(["git", "pull"])
        return self

    def branches(self):
        print(subprocess.run(['git','branch'], cwd=self.path, capture_output=True))
        return self

    def add(self):
        self.run(['git','add','.'])
        return self

    def commit(self,msg):
        self.msg = msg
        print(subprocess.run(['git','commit','-m', msg], cwd=self.path, capture_output=True))
        return self

    def pull(self):
        print(subprocess.run(['git','pull'], cwd=self.path, capture_output=True))
        return self

    def push(self):
        print(subprocess.run(['git','push','-u','origin',self.branch], cwd=self.path, capture_output=True))
        return self

    def code(self):
        cmd = ['cmd','/c','code',self.path]
        print(subprocess.run(cmd))
        return self
    
    def pr(self,base,msg='',body=''):
        if msg != '':
            print('setting pr msg to ',msg) 
            self.msg = msg
        title = f'{self.branch} to {base}: {self.msg}'
        self.run(['gh','pr','create','--base',base,'--head',self.branch,'--title',title,'--body', body])
        return self
    
    def merge(self):
        self.run(['sleep','5'])
        self.run(['gh','pr','merge','--merge','--auto'])
        return self
    
    def token(self):
        result = subprocess.run(['gh','auth','token'], cwd=self.path, capture_output=True)
        return result.stdout.strip().decode()
    
    def prs(self):
        url = f"https://api.github.com/repos/{self.owner}/{self.name}/pulls"
        headers = {"Authorization": f"Bearer {self.token()}"}
        params = {"state": "open"}
        response = requests.get(url, headers=headers, params=params)
        response.raise_for_status()
        df = (
            pd.DataFrame(response.json())
            # .html_url.iloc[0]
            .add_col(pick_base=lambda x: x.base['ref'] )
            .add_col(pick_head=lambda x: x.head['ref'] )
            .add_col(pick_user=lambda x: x.user['login'] )
            [['title','number','pick_base','pick_head','pick_user','html_url']]
            .add_col(my_url=lambda x: f'[{x.title} PR {x.number}]({x.html_url})' )
        )    

        # get mergeable state
        uid = get_username()
        results = []
        for i in df.itertuples():
            if uid in i.pick_user.lower():
                results.append({
                    'number':i.number, 
                    'mergeable_state':self.pr_details(i.number)
                })
        return df.merge(pd.DataFrame(results),on='number',how='left')

    def pr_details(self,number):
        url = f"https://api.github.com/repos/{self.owner}/{self.name}/pulls/{number}"
        headers = {"Authorization": f"Bearer {self.token()}"}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        return response.json()['mergeable_state']

    def diff(self,base):
        # download latest for branch we'll compare to
        _ = subprocess.run(['git','fetch','origin',f'{base}:{base}'],cwd=self.path)
        clause = f'{base}...{self.branch}'
        results = subprocess.run(['git','diff','--name-only',clause],cwd=self.path,capture_output=True, text=True)
        diffs = None
        if results.stdout: diffs = results.stdout.replace('\n','\n\t')
        print(f'File DIFFS for {clause} diffs are', diffs)
        return self
    
    def history(self):
        self.pull()
        cmd = ["git", "log", "-300", "--pretty=format:%h;%an;%s;%cr"]
        result = subprocess.run(cmd, cwd=self.path, capture_output=True, text=True)
        commits = []
        for line in result.stdout.splitlines():
            parts = line.split(";")
            commits.append({
                "hash": parts[0],
                "author": parts[1],
                "message": parts[2][:30], 
                "date": parts[3]
            })
        return pd.DataFrame(commits).assign(branch=self.branch)

if __name__ == '__main__':
    # pass
    # print(os.environ['HOME'])
    print(
        Repo('repo_name')
        .change_branch('feature')
        .history()
        # .diff('dev')
        # .diff('qa')
        # .pull()
        # .code()
        # .prs()
        # .cur_branch()       
        # .branches()
    )