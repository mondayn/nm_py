from cryptography.fernet import Fernet
import os

def get_cipher(): 
    return Fernet(os.environ['some_var'])

def encrypt(s):
    ''' returns binary'''
    return get_cipher().encrypt(s.encode())

def decrypt(b): return get_cipher().decrypt(b).decode()
