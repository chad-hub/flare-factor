# Flare-Factor

Deploy Machne Learning technologies, based on publically avaiable data from the Texas Railroad Commission, to predict flaring down to the individual lease level in Texas

## Why Flaring
Flaring, the burning of natural gas that is more expensive to get to market than to burn, has been a decades-long concern for environmental groups, and a frustrating problem for those who understand the energy industry. In my many years working in the oil and gas industry, I always understood why flaring happened and the cost/benefit of doing so - but it always bothered me. In my previous project I sought to quantify flaring energy wasted at the state level, and put it into perspective. Knowing the opportunity cost of the energy wasted inspired me to keep digging.  Many enterprising groups are developing innovative solutions for using the energy wasted to flaring. Identifying, locating and quantifying current flaring levels and locations is a cumbersome challenge with hundreds of thousands of wells reporting production and flaring monthly. This project will lay the ground work for building a 'marketplace' for current flaring energy in Texas by investigating the feasibility of predicting flaring levels down to the lease level. 

## The Tech

![Tech Stack](ec2_work/plots/ff_tech.png)

## The Data
As previously mentioned, the Texas Railraod Commission (TRRC) dumps all the data they collect monthly. This amounts to about 135,000 reports monthy in 2019/2020 from all active leases in Texas. The unzipped dataset exceeds 25 GB, and contains upwards of 65,000,000 entries for production and ~35,000,000 entries that contain information on flaring, dating back to 1993. After initally reading in the full dataset, I determine that there was a good reason to ignore pre-2000 data, as there is a clear departure from what was the normal in 2010, made evident in the plot below. 

![Flaring_district_full](ec2_work/plots/flare_by_district.png)
