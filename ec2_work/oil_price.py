# %%
import pandas as pd
import numpy as np
from datetime import date
from datetime import datetime
from selenium import webdriver
from bs4 import BeautifulSoup
import urllib3
from requests import get
import time
import re
import os
import math

pd.options.display.max_rows = 8
# %%
#write url based on current time
url = "https://www.cmegroup.com/CmeWS/mvc/Quotes/Future/425/G?pageSize=50&_="
urlext = np.round(time.time(), decimals=0)
urlext = str(urlext)
url = url + urlext

# create a new  session
# System.setProperty("webdriver.chrome.driver","/usr/bin/chromedriver");
# WebDriver driver = new ChromeDriver();
driver = webdriver.Chrome(executable_path='C:/usr/bin/chromedriver')
driver.get(url)
soup = BeautifulSoup(url)

#scrape site source page
urltext = soup.findAll(text=True)
urltext = driver.current_url
httptext = urllib3.PoolManager()
responsetext = httptext.request('GET', urltext)

#close browser
driver.close()

#convert BeautifulSoup object to text
souptext = BeautifulSoup(responsetext.data)
souptext2 = str(souptext)
# %%
