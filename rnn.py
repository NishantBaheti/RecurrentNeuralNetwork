# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## LSTM
# 
# - Solving `vanishing gradient` problem 
# 

# %%
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
plt.style.use("dark_background")

plt.rcParams.update({
    "axes.grid" : True
})


# %%
train_df = pd.read_csv("./Google_Stock_Price_Train.csv")
test_df = pd.read_csv("./Google_Stock_Price_Test.csv")


# %%
train_df["Date"] = train_df["Date"].apply(pd.to_datetime)
test_df["Date"] = test_df["Date"].apply(pd.to_datetime)


# %%
train_df.head()


# %%
test_df.head()


# %%
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
ax.set(
    title = "Values to Train",
    xlabel = "Date",
    ylabel = "Open"
)
ax.plot(train_df["Date"],train_df["Open"],label ="'Open' values over Date")
ax.legend(loc="best")
fig.show()


# %%
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
ax.set(
    title = "Values to test",
    xlabel = "Date",
    ylabel = "Open",
)
ax.plot(test_df["Date"],test_df["Open"],label ="'Open' values over Date")
ax.legend(loc="best")
fig.show()


# %%
training_set = train_df["Open"].values.reshape(-1,1)


# %%
training_set

# %% [markdown]
# - whenver there is RNN , it is recommended to apply normalization to the data

# %%
from sklearn.preprocessing import MinMaxScaler


# %%
normalizer = MinMaxScaler(feature_range=(0,1))
training_set_scaled = normalizer.fit_transform(training_set)
training_set_scaled

# %% [markdown]
# - creating a training seq of window 60(60 days) and for forcast training it will take next day's value. Lets see how it goes

# %%
x_train = []
y_train = []


# %%
window = 60
total_length = len(training_set_scaled)
for i in range(window,total_length):
    x_train.append(training_set_scaled[i-window:i,0])
    y_train.append(training_set_scaled[i,0])


# %%
x_train, y_train = np.array(x_train),np.array(y_train)


# %%
x_train.shape

# %% [markdown]
# - Here it is a 2D matrix making sense of only one set of features
# - Our goal is to convert it into a 3D matrix where if we have other features as well ,can be added.

# %%
x_train = np.reshape(x_train,(x_train.shape[0],x_train.shape[1],1))


# %%
x_train.shape


# %%
y_train.shape

# %% [markdown]
# ## building model

# %%
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# %% [markdown]
# - initialize model

# %%
regressor = Sequential()

# %% [markdown]
# - adding first LSTM layer and Dropout Regularization

# %%
regressor.add(
    LSTM(
        units = 50,
        return_sequences = True,
        input_shape = (x_train.shape[1],1)
    )
)


# %%
regressor.add(
    Dropout(
        rate = 0.2
    )
)

# %% [markdown]
# - adding second LSTM layer and Dropout Regularization

# %%
regressor.add(
    LSTM(
        units = 50,
        return_sequences = True
    )
)
regressor.add(
    Dropout(
        rate = 0.2
    )
)

# %% [markdown]
# - adding third LSTM layer and Dropout Regularization

# %%
regressor.add(
    LSTM(
        units = 50,
        return_sequences = True
    )
)
regressor.add(
    Dropout(
        rate = 0.2
    )
)

# %% [markdown]
# - adding fourth LSTM layer and Dropout Regularization (dont return Sequence)

# %%
regressor.add(
    LSTM(
        units = 50,
        return_sequences = False
    )
)
regressor.add(
    Dropout(
        rate = 0.2
    )
)

# %% [markdown]
# - add output layer

# %%
regressor.add(
    Dense(
        units = 1
    )
)

# %% [markdown]
# - Compile model

# %%
regressor.compile(
    optimizer = 'adam',
    loss = 'mean_squared_error',
)


# %%
regressor.fit(x_train, y_train, epochs = 200, batch_size = 32)

# %% [markdown]
# - prepare testing set

# %%
testing_set = test_df["Open"].values.reshape(-1,1)
testing_set

# %% [markdown]
# - Now to get predictions for a month (20 working days) similar size to the testing set , we need previous 60 days values as input features to get 61th prediction(60+1).
# - so joining both train df and test df to create a full dataset

# %%
total_dataset = pd.concat([train_df["Open"],test_df["Open"]],axis=0)


# %%

inputs_to_model = total_dataset[len(training_set) - len(testing_set) - window:].values
inputs_to_model = inputs_to_model.reshape(-1,1)
inputs_to_model = normalizer.transform(inputs_to_model)


x_test = []
upper_limit = window + len(testing_set)
for i in range(window,upper_limit):
    x_test.append(inputs_to_model[i-60:i,0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))



# %%

predicted_values = regressor.predict(x_test)
predicted_values = normalizer.inverse_transform(predicted_values)


# %%
fig = plt.figure(figsize=(12,8))
plt.plot(test_df["Date"],testing_set,label="real stock price")
plt.plot(test_df["Date"],predicted_values,label="predicted stock price")
plt.legend(loc="best")
plt.title("Comparison in prices")
plt.show()


# %%



