# %%
import pandas as pd
import geopandas as gp
# %%
import pprint
from bs4 import BeautifulSoup
import requests
import time
import random
# %%
from ftplib import FTP
import sys

 # %%
def getFile(ftp, filename):
    try:
        ftp.retrbinary("RETR " + filename ,open(filename, 'wb').write)
    except:
      print('error')
# %%
ftp_site = 'ftpe.rrc.texas.gov'
dir = '/shpwell/Wells'
shp_files = 'Wells'
local_dr = '/shape_files/'

# %%
ftp = FTP(ftp_site)
ftp.login()
ftp.cwd(dir)
filenames = ftp.nlst()
# %%
for f in filenames:
  getFile(ftp, f)

  # %%
import fiona
# %%
