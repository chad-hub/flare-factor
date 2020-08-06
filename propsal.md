# flare-factor

## Descrption:
This project will seek to predict natural gas flare in Texas via training a model to all available data.  


## Approach
Supervised learning via random forest / gradient boosted regression. This will have various levels of complexity:

### Level 1
Utilize previously explored data avilable on Texas Railroad Commission Site. Merge with other data sets including price of Natural Gas to train 

### Level 2
Combine level 1 with deeper levels of information including drilling rig count, well permiting rates etc. This will train in longer-leading indicators into the model. 

### Level 3
Level 2 plus proximty paramaterization. The location of flare sites is public information. While I consider it the most difficult data I will explore, I feel strongly that it is an important feature in the quantity of gas flared. 

## Data Sources
I've narrowed the scope of the study to Texas, so all public data should be scrapable through the Texas Railroad Commision site. I will also be bringing in information from other sources, such as price of natural gas. 

