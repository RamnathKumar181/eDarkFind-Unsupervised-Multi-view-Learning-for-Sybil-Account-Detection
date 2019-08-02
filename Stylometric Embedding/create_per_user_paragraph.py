import pandas as pd, numpy as np, csv
import math, json, sys

# Filepaths
DATASET = "./All_sites_stylometric.csv"

# Output Files
USER_COMMENTS = "./train_stylometric_user.csv"


# Load the dataset
sys.stdout.write("Loading dataset ..."+'\r')
sys.stdout.flush()
csv_data = pd.read_csv(DATASET)
X = csv_data.values
# users = set(X[1])
# Set of unique users
sys.stdout.write("Calculating set of users..."+'\r')
sys.stdout.flush()
users = [row[1] for row in X]
users = set(users)
print(users)

output_file = open(USER_COMMENTS,'w')
wr = csv.writer(output_file, quoting=csv.QUOTE_ALL)


# Accumulate comments of each user into paragraphs
sys.stdout.write("Accumulating user comments..."+'\r')
sys.stdout.flush()
for ind, user in enumerate(users):
    if ind%100 ==0:
	     sys.stdout.write(str(ind+1) +"/"+ str(len(users))+" users done..."+'\r' )
	     sys.stdout.flush()
    comments = [row[0] for row in X if row[1]==user]
    comments = [x for x in comments if str(x) != 'nan']
    comment = " <END> ".join(comments)
    ls=[]
    ls.append(user)
    ls.append(comment)
    wr.writerow(ls)
