''' ldap and azure fx'''

import pandas as pd
from df import clean_names

#region active directory
def get_members(dn='CN=*,OU=g00958,OU=00958,DC=*,DC=*,DC=net'):
    from pyad import adquery
    q = adquery.ADQuery()
    idx = ['cn','sn','title']
    # idx += ['mail','givenname']
    q.execute_query(attributes=idx,where_clause=''' memberof='{}' '''.format(dn))
    return pd.DataFrame(q.get_results())[idx]


#region ldap 
#higher level would be get_members in nm.py using pyad
def ldap_query(cols,where):
    import win32com.client #pip install pywin32

    adoConnection = win32com.client.Dispatch('ADODB.Connection')
    adoConnection.Provider = 'ADSDSOObject'
    adoConnection.Properties('Encrypt Password').Value = True
    adoConnection.Properties('ADSI Flag').Value = 1 
    adoConnection.Open('Active Directory Provider')

    qry = 'select ' + ','.join(cols) + ''' from 'LDAP://<<DOMAIN>>' where ''' + where

    adoCommand = win32com.client.Dispatch('ADODB.Command')
    adoCommand.ActiveConnection = adoConnection
    adoCommand.Properties('Page Size').Value = 1000
    adoCommand.CommandText = qry
    
    adoRecordset = win32com.client.Dispatch('ADODB.Recordset')
    adoRecordset.Open(adoCommand)
    lst = []
    while not adoRecordset.EOF:
        d = dict()
        for c in cols:
            d[c] = str(adoRecordset(c))
        lst.append(d)
        adoRecordset.MoveNext()
    adoRecordset.Close()
    adoConnection.Close()

    return pd.DataFrame(lst).clean_names()     

def ldap_groups(user)->pd.DataFrame:
    ''' lookup a user's ad groups '''

    #first get distinguished name
    dn = ldap_query(cols=['distinguishedName'],where=''' sAMAccountName='{}' '''.format(user)).distinguishedname.iloc[0]
    
    df = ldap_query(cols=['cn','sAMAccountName','managedBy','groupType'],where=''' member='{}' '''.format(dn))
    df['group_mapping'] = ['distribution' if i.grouptype in(['2','4','8']) else 'security' for i in df.itertuples()]
    return df.sort_values('group_mapping').sort_values('samaccountname').reset_index(drop=True)

def ldap_members(grp_samaccountname, recurse=True)->pd.DataFrame:
    """lookup members of an ad group::    
    Args:
        grp_samaccountname (str): group name
        recurse (bool, optional): explode groups into users. Defaults to True.
    """     
    #first get distinguished name
    dn = ldap_query(cols=['distinguishedName'],where=''' sAMAccountName='{}' '''.format(grp_samaccountname)).distinguishedname.iloc[0]
    
    # dict of columns to select and rename
    cols = {
        'givenname':'first_name',
        'sn':'last_name',
        'samaccountname':'ntname',
        'mail':'mail',
        'physicaldeliveryofficename':'office',
        'company':'company',
        'title':'title',
        'whencreated':'created',
        'objectclass':'objectclass',
        'useraccountcontrol':'uac',
    }
       
    where = ''' memberof='{}' '''.format(dn)
    
    df = ldap_query(cols=cols.keys(),where=where).rename(columns=cols)

    if recurse:
        members = (
            df
            # .filter_string('objectclass','user')
            .query("uac in (['512','66048'])") #active person (512) service account (66048)
        )
        for i in (
            df
            # .filter_string('objectclass','group')
            .itertuples()
        ):
            members = members.append(ldap_members(i.ntname,recurse=True))
    else:
        members = df
               
    members['created'] = pd.to_datetime(members.created).dt.date  
    members['ntname'] = members.ntname.str.lower()
    
    return members.sort_values('last_name').sort_values('first_name').reset_index(drop=True)
	
def ldap_find_common(users # ['u1','u2']
                     ,group_mapping='security' #distribution
                    ):
    groups = [ldap_groups(u).query(f''' group_mapping == '{group_mapping}' ''') for u in users]
    common_groups = groups[0]
    for g in groups:
        common_groups = common_groups.merge(g[['cn']],on=['cn'],how='inner')
    return common_groups
#endregion ldap

# region azure ad 
from functools import lru_cache

@lru_cache
def get_headers():
    from azure.identity import InteractiveBrowserCredential
    scope = 'https://graph.microsoft.com/.default'
    token_string = InteractiveBrowserCredential().get_token(scope).token
    headers={"Authorization": f'Bearer {token_string}'}
    return(headers)

def request_groups(user,groups,url):
    import requests
    json = requests.get(url,headers=get_headers()).json()

    #append the parsed group information
    groups += list(map(lambda i: i['displayName'], json['value']))

    #paginate to next set of records
    if '@odata.nextLink' in json:  
        user,groups,url=request_groups(user,groups,json['@odata.nextLink'])

    return user,groups,url

def get_azure_users_groups(user):
    url = f'https://graph.microsoft.com/v1.0/users/{user}@<<domain>>/memberOf'
    _,groups,_=request_groups(user,[],url)
    return pd.DataFrame({'azure_grp':groups}).assign(user=user)

# endregion azure

if __name__ == '__main__':
    pass

