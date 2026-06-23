import requests, json, os
from dticom.gcp import get_adctoken
import datetime
from urllib.parse import urlencode
import webbrowser

# https://airflow.apache.org/docs/apache-airflow/2.1.0/stable-rest-api-ref.html


def get_airflow(env): return "airflow_url"

def get_composer_path(env,dag_id):
    return 'composer_ur'

class Dag:
    def __init__(self,env,id):
        self.dag_id=id
        self.env=env
        self.entity=(
            self.dag_id
# 
            .replace('_silver','')
        )
        print(f'env = {env}, dag_id = {self.dag_id}')
        self.airflow=get_airflow(env)
        self.url = f'{self.airflow}/dags/{self.dag_id}'
        self.composer_path=get_composer_path(env,self.dag_id)
        self.header={"Authorization": 'Bearer '+get_adctoken()}
        self.future_runs = []
    
    def web(self):
        url = f'{self.airflow}/dags/{self.dag_id}'
        print(url)
        webbrowser.open(url)
        return self
            
    def composer(self):
        webbrowser.open(self.composer_path)
        return self
    
    def update_status(self,status):
        url = f"{self.airflow}/api/v1/dags/{self.dag_id}"
        requests.patch(url,headers=self.header,json={"is_paused":status}).raise_for_status()
    
    def pause(self): 
        self.update_status(True)
        return self

    def unpause(self): 
        self.update_status(False)
        return self

    def get_future_runs(self):
        url = f"{self.airflow}/api/v1/dags/{self.dag_id}/dagRuns?state=running"
        response = requests.get(url, headers=self.header)
        response.raise_for_status()
        runs = response.json().get("dag_runs")
        if runs: self.future_runs = [r['dag_run_id'] for r in runs]
        return self

    def clear_future_runs(self,exception=None):
        self.get_future_runs()
        print('future runs',self.future_runs)
        for r in self.future_runs:
            url = f"{self.airflow}/api/v1/dags/{self.dag_id}/dagRuns/{r}"
            if not exception or exception not in r:
                print('deleting',url)
                response = requests.delete(url, headers=self.header)
                response.raise_for_status()
        return self

    def clear_queue(self):
        url = f"{self.airflow}/api/v1/dags/{self.dag_id}/dagRuns?state=queued"
        response = requests.get(url, headers=self.header)
        response.raise_for_status()
        runs = response.json().get("dag_runs")
        for r in runs:
            url = f"{self.airflow}/api/v1/dags/{self.dag_id}/dagRuns/{r['dag_run_id']}"
            # print(url)
            response = requests.delete(url, headers=self.header)
            response.raise_for_status()
        return self

    def full_load(self):
        self.pause()
        url = f"{self.airflow}/api/v1/dags/{self.dag_id}/dagRuns"
        config = {"conf": {"mode": "full_load"}}
        header = self.header.copy()
        header['Content-Type']="application/json"
        response = requests.post(url, headers=header, data=json.dumps(config))
        print(f'Trigger full load {self.dag_id}, {response.status_code} ', response.json())
        self.unpause()
        import time
        time.sleep(5)
        self.clear_future_runs(exception='manual')
        return self

    def delete(self):
        url = f"{self.airflow}/api/v1/dags/{self.dag_id}"
        self.clear_future_runs()
        response = requests.delete(url,headers=self.header)
        print('Delete ',response.status_code,' on',self.dag_id)

    def get_cur_duration(self):
        url = f"{self.airflow}/api/v1/dags/{self.dag_id}/dagRuns?state=running"
        response = requests.get(url, headers=self.header)
        response.raise_for_status()
        try:
            t_from = datetime.datetime.now().astimezone()
            t_to = datetime.datetime.fromisoformat(response.json()['dag_runs'][0]['start_date'])
            return (t_from - t_to).seconds / 60
        except Exception as e:
            return None

    def adhoc(self,date_from,date_to):
        url = f"{self.airflow}/api/v1/dags/{self.dag_id}/dagRuns"
        config = {"conf": {"mode": "adhoc", 'date_from':date_from, 'date_to':date_to}}
        header = self.header.copy()
        header['Content-Type']="application/json"
        response = requests.post(url, headers=header, data=json.dumps(config))
        print(f'Trigger adhoc {self.dag_id}, {date_from}, {date_to}, {response.status_code} ', response.json())
        self.unpause()
        return self
    
    def dagruns(self):
        url = f"{self.airflow}/api/v1/dags/{self.dag_id}/dagRuns?"
        params = {
            'start_date_gte':'2026-01-21T05:50:00Z',
            'order_by':'start_date',
            # 'limit':2,
            'start_date_lte':'2026-01-21T06:15:00Z'
        }
        url += f'{urlencode(params)}'
        response = requests.get(url, headers=self.header)
        response.raise_for_status()
        print('entries: ',response.json()['total_entries'])
        def keyfilter(d): 
            return { k:v for k,v in d.items() if k in ['dag_run_id','start_date','end_date','run_type','state']}
        return [keyfilter(d) for d in response.json()['dag_runs']]
        
    def xcom(self,dag_run_id):
        url = f"{self.airflow}/api/v1/dags/{self.dag_id}/dagRuns/{dag_run_id}/taskInstances/read_{self.entity}_query/xcomEntries/{self.entity}_query"
        response = requests.get(url, headers=self.header)
        response.raise_for_status()
        return response.json()['value']

    def latest_runs(self):
        url = f"{self.airflow}/api/v1/dags/{self.dag_id}/dagRuns?"
        params = {
            "limit": 2,
            "order_by": "-execution_date",
        #  "state": "success"  # or "failed", "skipped", etc.
        }
        url += f'{urlencode(params)}'
        response = requests.get(url, headers=self.header)
        response.raise_for_status()
        
        return([(
            self.env,
            # i['dag_id'],
            self.entity,
            i['execution_date'],i['state']) for i in response.json()['dag_runs']])

if __name__ == '__main__':
    pass