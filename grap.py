import re
import os

f = open( "learingrate_02.out.262859" , "r")

train = re.search(\[train:(\d+)\] cross entropy loss: (\d+\.\d+) - center loss: (\d+\.\d+) - total weighted loss: (\d+\.\d+) , f )

train_end = re.search( \[train:(\d+)\] finished. cross entropy loss: (\d+\.\d+) - center loss: (\d+\.\d+) - together: (\d+\.\d+) - top1 acc: (\d+\.\d+) % - top3 acc: (\d+\.\d+), f )

validate =  re.search( \[validate:(\d+)\] cross entropy loss: (\d+\.\d+) - center loss: (\d+\.\d+) - total weighted loss: (\d+\.\d+), f )

print( train.group() + train.group(1) + train.group(2) + train.group(3) )
