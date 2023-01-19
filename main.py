# Topic: Data science using Python
# Analyze the following dataset
# Fetch and display various parameters of the "titanic" dataset
# Filter out any anomalies (if any) from the dataset
# Plot histograms for important parameters

!pip install tensorflow-gpu
!pip install tensorflow-datasets
!pip install tfds-nightly

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import csv

from google.colab import drive
drive.mount('/drive')

# >> loading in data
ds = tfds.load('titanic', split='train', shuffle_files=True)
assert isinstance(ds, tf.data.Dataset)


# >>> Filtering out any anomalies from the dataset
ds = ds.filter(lambda data: data['sex'] == 0 or data['sex'] == 1)
ds = ds.filter(lambda data: data['age'] >= 0)
ds = ds.filter(lambda data: data['fare'] >= 0)
# ds = ds.filter(lambda data: data['cabin'] != b'Unknown')


# >>> Age parameter
ages = [example['age'] for example in ds]

plt.hist(ages, bins=80)
plt.title('Age distribution')
plt.xlabel('Age')
plt.ylabel('# of people')
plt.show()


# >>> Fetch and draw histogram for the sex parameter
sex_list = []
for example in ds:
    sex_list.append(example['sex'])
genders = ["Men", "Women"]

unique, counts = np.unique(sex_list, return_counts=True)
gender_amount = [counts[0], counts[1]]

# raw data
print(f"Total # of people: {counts[0] + counts[1]}\n")
print(f"Number of men: {counts[0]}")
print(f"Number of women: {counts[1]}\n")

# graph
index_sex = np.arange(len(genders))
plt.bar(index_sex, gender_amount, facecolor='blue', alpha=0.5)
plt.xlabel('sex')
plt.ylabel('# of people')
plt.xticks(index_sex, genders)
plt.title('Sex Distribution - 1')
plt.show()
print()

# Pie Chart
genders = ["Men", "Women"]
plt.pie(counts, explode = (0, 0.1), labels=genders, autopct='%1.1f%%',
        shadow=True, startangle=90)
plt.title('Sex Distribution - 2')
plt.show()


# >>> Fetch and plot graph for the "survived" parameter
survived = []
for example in ds:
    survived.append(example["survived"])
    
# Pie chart
labels = 'Survived', 'Not Survived'
explode = (0, 0.1)
sizes = [sum(survived), len(survived) - sum(survived)]

fig1, ax1 = plt.subplots()
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
plt.title('Survival Rate')
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle

plt.show()


# >> Cabin
ds = ds.filter(lambda data: data['cabin'] != b'Unknown')
cabins = np.array([])

for example in ds:
    cabins = np.append(cabins, example['cabin'].numpy())

n, bins, patches = plt.hist(cabins, bins=50)

plt.xlabel('Cabin Number')
plt.ylabel('Number of Passengers')
plt.title('Number of Passengers in Each Cabin')

plt.show()


# >>> the number of men that survived and the number of women that survived

# Get all the data
data = list(ds.as_numpy_iterator())

# Create two lists to store the survived men and women
survived_men = []
survived_women = []

# Iterate through the data, and check if the person survived and if they are male or female
for example in data:
    if example['survived'] == 1:
        if example['sex'] == 0:
            survived_men.append(example)
        else:
            survived_women.append(example)

# Plot the result
plt.bar(['Men', 'Women'], [len(survived_men), len(survived_women)])
plt.title("Survived Men and Women")
plt.xlabel("Gender")
plt.ylabel("Number of Survivors")
plt.show()

# Plot the result (pie chart)
plt.pie(
    [len(survived_men), len(survived_women)],
    labels=['Men', 'Women'],
    shadow=True,
    startangle=90,
    autopct='%1.1f%%'
)
plt.title("Distribution of Men and Women that survived")
plt.show()


# >>> price
prices = []
for example in ds:
    price = example['fare'].numpy()
    prices.append(round(price,2))

plt.hist(prices, bins=100)
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.title('Histogram of Ticket Prices on Titanic')
plt.show()


# >>> exporting to .csv
temp_list = list(ds)

ds_numpy = tfds.as_numpy(ds)
my_list = list()

filter_list = list()

filter_list.append(["sex", "age", "fare", "cabin", "survived"])

for i in ds_numpy:
    my_list.append(i)

    # grab your parameters!!!!
    sex = i["sex"]
    if (sex == 0):
        sex = "Man"
    else:
        sex = "Woman"
    age = i["age"]
    age = int(age)
    fare = i["fare"]
    fare = round(fare, 2)
    cabin = i["cabin"]
    cabin = str(cabin)
    cabin = cabin[2:-1]
    survived = i["survived"] # 1 == survived
    if (survived == 1):
        survived = "Yes"
    else:
        survived = "No"

    nested_list = [sex, age, fare, cabin, survived]
    filter_list.append(nested_list)

# exporting to .csv file
with open('export_ece326_file.csv', 'w') as csvfile:
    file_write = csv.writer(csvfile)

    for example in filter_list:
        file_write.writerow(example)

csvfile.close()
