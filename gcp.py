from util import thread_first
from google.auth import default
from google.auth.transport.requests import Request

def get_creds():
    creds, project = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
    return creds

def get_adctoken():
    '''same as gcloud auth print-access-token'''
    try:
        creds = get_creds()    
        creds.refresh(Request())
        return creds.token
    except Exception as e:
        reauth_adc()

def reauth_adc():
    import subprocess
    print('trying reauth')
    subprocess.run(["gcloud", "auth", "login", "--update-adc"],check=True)

# gcloud auth application-default revoke

import requests
import pandas as pd

def get_df_jobs(env):
    '''get dataflow jobs'''
    project=f'{env}'
    location='us-east4'
    headers={"Authorization": 'Bearer '+get_adctoken()}

    response=requests.get(f'https://dataflow.googleapis.com/v1b3/projects/{project}/locations/{location}/jobs?filter=ACTIVE',headers=headers)   
    response.raise_for_status()
    cols = ['id', 'projectId', 'name', 'type', 'currentState', 'currentStateTime', 
            'createTime', 'location', 'startTime', 'jobMetadata.sdkVersion.version',
            'serviceResources.zones']
    return (
        pd.json_normalize(response.json()['jobs'])
        [cols]
        .add_col(url=lambda x: 
            f'https://console.cloud.google.com/dataflow/jobs/us-east4/{x.id};graphView=0?project={project}'
        )
    )

# gets subnetwork and number of workers if needed
# def get_dataflow_stats(jobId):
#     env='qa'
#     location='us-east4'

#     url = f'https://dataflow.googleapis.com/v1b3/projects/{project}/locations/{location}/jobs/{jobId}?view=JOB_VIEW_DESCRIPTION'

#     headers={"Authorization": 'Bearer '+get_adctoken()}

#     response=requests.get(url,headers=headers)   
#     response.raise_for_status()
#     json=response.json()
#     try:
#         numWorkers=json['environment']['sdkPipelineOptions']['options']['numWorkers']
#         subnetwork=json['environment']['sdkPipelineOptions']['options']['subnetwork']
#     except Exception as e:
#         numWorkers='?'
#         subnetwork='?'
#     return (subnetwork,numWorkers)

def parse_ds(*args):
    ''' parse project dataset table'''
    arg_str = '.'.join(args[:3])
    # if not arg_str.startswith('###'):
    #     arg_str = '####' + arg_str
    arg_tuple = arg_str.split('.')
    if len(arg_tuple) != 3:
        print(f'couldnt parse_ds {arg_tuple}')
        return (None,None,None)
    return arg_tuple

class Bq():
    def __init__(self,project=DEFAULT_PROJECT):
        self.cost_factor = 5 / (1024**4) #$5 per GB
        self.client = bigquery.Client(
            # project=project
            )
        self.project = project
    
    def dry_run(self,sql):
        job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
        try:
            job = self.client.query(sql,job_config=job_config,)
            bytes = job.total_bytes_processed
            cost = bytes * self.cost_factor
            print(f"Estimated bytes processed: {bytes:,.0f}, cost: ${cost:,.2f}")
        except Exception as e:
            print(e)

    # @track_duration
    def query(self,sql):
        if os.path.isfile(sql): 
            sql = load_sql(sql)
        
        sql += ';;'
        scripts = sql.split(';')
        results = []
        for s in scripts:
            try:
                s = s.strip()
                if len(s) > 1:
                    s = s.strip()
                    query_job = self.client.query(s)                
                    df = query_job.result().to_dataframe()
                
                    billed = query_job.total_bytes_billed or 0
                    processed = query_job.total_bytes_processed or 0
                    cost = billed * self.cost_factor
                
                    print(
                        f"Cost: ${cost:,.4f}, "
                        f"Billed Bytes: {billed:,.0f}, "
                        f"Processed Bytes: {processed:,.0f}, "
                        f"Shape: {df.shape}"
                    )
                    results.append(df)
            except Exception as e:
                print('err',e)
                continue        
        if len(results)==1: return results[0]
        return results
         
    def execute(self,sql):
        job = self.client.query(sql)
        result = job.result()  #wait
        if job.errors:
            print(f"The query failed with errors:")
            for error in job.errors:
                print(error)
        else:
            print("Query executed successfully.")
        print(f"Query job {job.job_id} finished.")
    
#     def insert(self,df,tbl):
#         job_config=bigquery.LoadJobConfig(write_disposition='WRITE_APPEND',)
#         job = self.client.load_table_from_dataframe(dataframe=df,destination=tbl,job_config=job_config)
#         job.result() #wait
#         print(f"Loaded {job.output_rows} rows into {tbl}")

    def columns(self,path=''):
        proj, db, tbl = path.split('.')
        sql = f'''
            SELECT table_catalog,table_schema,table_name,
            column_name,ordinal_position,data_type,is_nullable,is_partitioning_column
            FROM {proj}.{db}.INFORMATION_SCHEMA.COLUMNS
            where table_name=lower('{tbl}')
        '''
        return self.query(sql)
    
    def partition(self,table_name):
        return self.client.get_table(table_name).time_partitioning

    def cluster(self,table_name):
        return self.client.get_table(table_name).clustering_fields
    
    def ddl(self,*args):
        project, dataset, table = parse_ds(*args) 
        if project and dataset and table:
            sql = f"select ddl from {project}.{dataset}.INFORMATION_SCHEMA.TABLES where table_name='{table}'"
            return self.query(sql).ddl.iloc[0]
    
    def recent_cost(self):
        sql = f''' 
        select user_email,cast(creation_time as date) dt,sum(est_cost) est_cost
        from {self.project}.`region-us`.INFORMATION_SCHEMA.JOBS_BY_USER  
            ,unnest(array[{self.cost_factor} * total_bytes_billed]) as est_cost
        where creation_time >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
        group by 1,2 order by 2 desc
        '''
        return self.query(sql).assign(proj=self.project)

        
def profile_qry(sql):
    def get_cols(sql):
        result = bigquery.Client(project=DEFAULT_PROJECT).query(sql+' limit 0').result()
        return [f.name for f in result.schema]

    def get_profile_sql(cols,sql):
        sql_field = lambda c: (
            f'min(length(cast({c} as string))) {c}_minlen,'
            f'max(length(cast({c} as string))) {c}_minmax,'
            f'sum(case when {c} is null then 1 else 0 end) {c}_nulls,'
            f'count(distinct {c}) {c}_distinct'
        )
        new_sql = f'with c as ({sql} '
        # new_sql += 'order by rand() limit 1000000'
        return (
            new_sql + f' ) select ' + ', '.join([sql_field(c) for c in cols]) + ',count(*) records from c'
        )

    df = thread_first(
        sql,
        get_cols,
        (get_profile_sql,sql),
        Bq().query
    )    
    df2 = (
        df.T
        .query(''' index!='records' ''')
        .reset_index()
        .add_col(col=lambda x: x.index.split('_')[-1])
        .add_col(field=lambda x: '_'.join(x.index.split('_')[:-1]))
        .pivot_table(aggfunc='min',index=['field'],values=[0],columns=['col'])
    )
    df2.columns = df2.columns.droplevel(0)
    return (
        df2
        .assign(records=lambda x: df.records.iloc[0].astype('float'))
        .assign(nulls=lambda x: x.nulls.astype('float'))
    )

def view_bq(*args):
    project,dataset,table=parse_ds(*args)
    path=f'https://console.cloud.google.com/bigquery?ws=!1m5!1m4!4m3!1s'+project+'!2s'+dataset+'!3s'+table
    webbrowser.open(path)


if __name__ == '__main__':
    pass    


def ocr(file):
    from google.cloud import vision
    import io
    creds = get_creds()
    creds.refresh(Request())
    creds = creds.with_quota_project('gcp')
    client = vision.ImageAnnotatorClient(credentials=creds)
    resp = client.text_detection(image={"content": io.open(file, "rb").read()})
    return resp.full_text_annotation.text if resp.full_text_annotation else ""

if __name__ == '__main__':
    reauth_adc()
