import xarray as xr
import geopandas as gpd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from shapely.geometry import Point
import pandas as pd
import numpy as np
#!pip install cupy
#import matplotlib.pyplot   as plt
#from cuml.neighbors import KNeighborsRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.impute import SimpleImputer
import streamlit as st
from sklearn.linear_model import LinearRegression
#import pygwalker as pyg
import wget as wget

st.title("Geo Spatial Interpolation")

st.markdown("---")



st.set_option('deprecation.showPyplotGlobalUse', False)

# Download dataset file from GitHub
dataset_url = "https://github.com/Tanvi-Jain01/Delhi-AirQuality-/blob/main/Streamlit/daily_data.nc"
#dataset_file = wget.download(dataset_url)

# Read the NetCDF file
ds = xr.open_dataset(dataset_url)
#ds = xr.open_dataset(r'C:\Users\Harshit Jain\Desktop\delhiaq\daily_data.nc')
#ds = xr.open_dataset(r'daily_data.nc')
df = ds.to_dataframe().reset_index()
#----------------------------------------------------------------------------------

unique=df[['station','latitude','longitude']].drop_duplicates()
#print(unique)
#print(len(unique))
#type(unique)

lat = unique['latitude']
lon = unique['longitude']

geometry = [Point(x, y) for x, y in zip(lon, lat)]
stationgeo=gpd.GeoDataFrame(unique,geometry=geometry)
#print(stationgeo)
#type(stationgeo)


#-------------------------------------------------------------------------------------------------------

gdf_shape = (r'C:\Users\Harshit Jain\Desktop\delhiaq\Delhi\Districts.shp')
gdf_shape = gpd.read_file(gdf_shape)


#--------------------------------------------------------------------------------------------------------------

gdf_data = gpd.GeoDataFrame(unique, geometry=geometry)

# Set the CRS of the GeoDataFrame to match the shapefile
gdf_data.crs = gdf_shape.crs   #try directly stationgeo here


#-------------------------------------------------------------------------------------------
df = df.fillna(method='ffill')
df = df.fillna(method='bfill')


#-------------------------------------------------------------------------------------------

all_stations = df.station.unique()
train_station, test_station = train_test_split(all_stations, test_size=0.2, random_state=42)
X_train = df[df.station.isin(train_station)]
X_test = df[df.station.isin(test_station)]

#X_train.station.unique().shape, X_test.shape, X_test.columns    


#------------------------------------------------------------------------------
#df.set_index("time_")

X_train = X_train[['Date','latitude','longitude','PM2.5']]
X_test = X_test[['Date','latitude','longitude','PM2.5']]

#---------------------------------------------------------------------------
st.sidebar.title("Train Test")

selected_date = st.sidebar.date_input('Select Date', value=pd.to_datetime('2022-08-23'))

#st.write(selected_date)

lr_selected = st.sidebar.checkbox('Linear Regression')
rf_selected = st.sidebar.checkbox('Random Forest')
dt_selected = st.sidebar.checkbox('Decision Tree')
knn_selected = st.sidebar.checkbox('K-Nearest Neighbor',value=True)
mean_selected = st.sidebar.checkbox('Mean Prediction')

selected_date = pd.to_datetime(selected_date)
#-------------------------------------------------------------------------------------------


# Extract the features and target variable for the selected date from the training data

X_train_date = X_train[X_train['Date'] == selected_date]
X_train_date = X_train_date.drop('PM2.5', axis=1)
X_train_date['Date'] = X_train_date['Date'].astype(np.int64)

y_train_date = X_train[X_train['Date'] == selected_date]['PM2.5'].values
    

mean_fork = len(X_train_date['latitude'])
st.write(mean_fork)
# Define the parameter grid for grid search
param_grid = {'n_neighbors': [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]}  # Specify the range of K values to try

knn = KNeighborsRegressor()

grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_date, y_train_date)


best_k = grid_search.best_params_['n_neighbors']
best_mse = -grid_search.best_score_







# Random Forest
param_grid = {
    'n_estimators': [100, 150,200,250,300],  # Number of trees in the forest
    'max_depth': [None,1,2,4,5,7,8,10],  # Maximum depth of the trees
    
   
}

# Create the Random Forest regressor
rf = RandomForestRegressor()

# Perform grid search to find the best hyperparameters
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train_date, y_train_date)

# Retrieve the best hyperparameters and the corresponding mean squared error
best_params=grid_search.best_params_
n_estimator = grid_search.best_params_['n_estimators']  
max_depth = grid_search.best_params_['max_depth']

best_mse = -grid_search.best_score_

# Print the best hyperparameters and the corresponding mean squared error
st.write('Best Hyperparameters:', best_params)
st.write('Best Mean Squared Error:', best_mse)



#-------------------------------------------------------------------------------------------

def fit_model(x,model_name):
    X, y = x.iloc[:, 1:-1], x.iloc[:, -1]
    models = {
        'lr': LinearRegression().fit(X, y),
        'rf': RandomForestRegressor(n_estimators=n_estimator, max_depth=max_depth, n_jobs=-1, random_state=42).fit(X, y),
        'dt': DecisionTreeRegressor().fit(X, y),
        'knn': KNeighborsRegressor(n_neighbors=best_k, metric='euclidean', n_jobs=-1).fit(X, y),
        'mean':KNeighborsRegressor(n_neighbors=mean_fork, metric='euclidean', n_jobs=-1).fit(X, y)
    }
    return models

model_listlr = []
model_listrf = []
model_listdt = []
model_listknn = []
model_listmean=[]
train_time_df = X_train[X_train['Date'] == selected_date].groupby('Date')

for model_name in ['lr', 'rf', 'dt', 'knn','mean']:
    models = train_time_df.apply(fit_model, model_name=model_name)
    if model_name == 'lr':
        model_listlr.extend(models)
    elif model_name == 'rf':
        model_listrf.extend(models)
    elif model_name == 'dt':
        model_listdt.extend(models)
    elif model_name == 'knn':
        model_listknn.extend(models)
    elif model_name == 'mean':
        model_listmean.extend(models)

# -------------------------------------------------------------------------------------------

### TRAINING RMSE

st.subheader("Training")
st.markdown("---")

rmse_values = []

predn_listlr = []
predn_listrf = []
predn_listdt = []
predn_listknn = []
predn_listmean = []

fig, ax = plt.subplots(figsize=(30, 15))
ax.set_facecolor('black')
train_time_df_i=pd.DataFrame()

#bar_width = 0.4
index_offset = 0.4

for i, j in enumerate(train_time_df.groups.keys()):
    X_train_i = train_time_df.get_group(j)
    y_train_i = X_train_i.iloc[:, -1]
    train_time_df_i['true_y']=y_train_i
    train_time_df_i.reset_index(drop=True, inplace=True)
     
    #ax.bar(train_time_df_i.index, train_time_df_i['true_y'], label='True Y',width=bar_width)
    #index_offset += bar_width
    if lr_selected:
        y_train_pred_lr = model_listlr[i]['lr'].predict(X_train_i.iloc[:, 1:-1])
        predn_listlr.append(y_train_pred_lr)

        
        train_time_df_i['pred_lr']=np.concatenate(predn_listlr)

        #ax.bar(train_time_df_i.index+index_offset, train_time_df_i['pred_lr'], label='Linear Regression',width=bar_width)

        #train_time_df_i.plot.bar(ax=ax)
        
        rmse_lr = mean_squared_error(y_train_i, y_train_pred_lr, squared=False)
        st.write('Training RMSE Linear Regression:', rmse_lr)
       # index_offset += bar_width

    if rf_selected:

        y_train_pred_rf = model_listrf[i]['rf'].predict(X_train_i.iloc[:, 1:-1])
        predn_listrf.append(y_train_pred_rf)

        train_time_df_i['pred_rf']=np.concatenate(predn_listrf)
    
        rmse_rf = mean_squared_error(y_train_i, y_train_pred_rf, squared=False)

       # train_time_df_i.plot.bar(ax=ax)
        #ax.bar(train_time_df_i.index+index_offset, train_time_df_i['pred_rf'], label='Random Forest',width=bar_width)
        st.write('Training RMSE Random Forest:', rmse_rf)
        #index_offset += bar_width

    
    if dt_selected:

        y_train_pred_dt = model_listdt[i]['dt'].predict(X_train_i.iloc[:, 1:-1])
        predn_listdt.append(y_train_pred_dt)


        train_time_df_i['pred_dt']=np.concatenate(predn_listdt)
        rmse_dt = mean_squared_error(y_train_i, y_train_pred_dt, squared=False)

       # train_time_df_i.plot.bar(ax=ax)
        #ax.bar(train_time_df_i.index+index_offset, train_time_df_i['pred_dt'], label='Decision Tree',width=bar_width)
        st.write('Training RMSE Decision Tree:', rmse_dt)
        #index_offset += bar_width

    
    if knn_selected:

        y_train_pred_knn = model_listknn[i]['knn'].predict(X_train_i.iloc[:, 1:-1])
        predn_listknn.append(y_train_pred_knn)

        train_time_df_i['pred_knn']=np.concatenate(predn_listknn)

        rmse_knn = mean_squared_error(y_train_i, y_train_pred_knn, squared=False)

       # train_time_df_i.plot.bar(ax=ax)
        #ax.bar(train_time_df_i.index+index_offset, train_time_df_i['pred_knn'], label='K-Nearest Neighbor',width=bar_width)
        st.write('Training RMSE K-Nearest Neighbor:', rmse_knn,'Best K:', best_k)
        #index_offset += bar_width
    

    if mean_selected:
        y_train_pred_lr = model_listlr[i]['mean'].predict(X_train_i.iloc[:, 1:-1])
        predn_listmean.append(y_train_pred_lr)

        
        train_time_df_i['pred_mean']=np.concatenate(predn_listmean)

        #ax.bar(train_time_df_i.index+index_offset, train_time_df_i['pred_mean'], label='Mean Prediction',width=bar_width)
        #train_time_df_i.plot.bar(ax=ax)
        rmse_lr = mean_squared_error(y_train_i, y_train_pred_lr, squared=False)
        st.write('Training RMSE Mean Prediction:', rmse_lr)
        #index_offset += bar_width




st.write(train_time_df_i.T)



# -------------------------------------------------------------------------------------------
train_time_df_i.plot.bar(ax=ax)
plt.xlabel('Stations')
plt.ylabel('PM2.5 Values')
plt.title('Predictions')
plt.xticks(rotation=90)
plt.xticks(fontsize=8)
plt.tight_layout()
#plt.show()
st.pyplot()

# Set the axis labels and title
# ax.set_xlabel('Station', color='Black')
# ax.set_ylabel('Value', color='Black')
# ax.set_title('True Y vs Pred Y', color='Black')
# ax.legend()
# st.pyplot(fig)


# -------------------------------------------------------------------------------------------



st.subheader("Testing")
st.markdown("---")


predn_list_lr = []
predn_list_knn = []
predn_list_dt = []
predn_list_rf = []
predn_list_mean = []


test_time_=pd.DataFrame()
test_time_ = X_test[X_test['Date'] == selected_date].reset_index()
     
#st.write(test_time_['PM2.5'])  

test_time_df = X_test[X_test['Date'] == selected_date].groupby('Date')
#st.write(test_time_df)




fig, ax = plt.subplots(figsize=(30, 15))
ax.set_facecolor('black')

#ax.scatter(test_time_.index, test_time_['PM2.5'], label='True Y')

# Predict using the trained model for each time step
for i, j in enumerate(test_time_df.groups.keys()):
    group_a = test_time_df.get_group(j)


    if lr_selected:
        predns = model_listlr[i]['lr'].predict(group_a.iloc[:, 1:-1])
        predn_list_lr.append(predns)

        test_time_['pred_lr'] =  np.concatenate(predn_list_lr)
        
        rmse = mean_squared_error(test_time_['PM2.5'], np.concatenate(predn_list_lr)) ** 0.5
        st.write('Testing RMSE Linear Regression', rmse)

        # ax.bar(test_time_.index, test_time_['pred_lr'], label='Linear Regression',sorted=True)



    if rf_selected:
        predns = model_listrf[i]['rf'].predict(group_a.iloc[:, 1:-1])
        predn_list_rf.append(predns)

        test_time_['pred_rf'] =  np.concatenate(predn_list_rf)

        rmse = mean_squared_error(test_time_['PM2.5'], np.concatenate(predn_list_rf)) ** 0.5
        st.write('Testing RMSE Random Forest', rmse)

        # ax.bar(test_time_.index, test_time_['pred_rf'], label='Random Forest',sorted=True)


    if dt_selected:
        predns = model_listdt[i]['dt'].predict(group_a.iloc[:, 1:-1])
        predn_list_dt.append(predns)

        test_time_['pred_dt'] =  np.concatenate(predn_list_dt)

        rmse = mean_squared_error(test_time_['PM2.5'], np.concatenate(predn_list_dt)) ** 0.5
        st.write('Testing RMSE Decision Tree', rmse)

        # ax.bar(test_time_.index, test_time_['pred_dt'], label='Decision Tree',sorted=True)


    if knn_selected:
        predns = model_listknn[i]['knn'].predict(group_a.iloc[:, 1:-1])
        predn_list_knn.append(predns)

        test_time_['pred_knn'] =  np.concatenate(predn_list_knn)
        
        rmse = mean_squared_error(test_time_['PM2.5'], np.concatenate(predn_list_knn)) ** 0.5
        st.write('Testing RMSE KNN', rmse)

        # ax.bar(test_time_.index, test_time_['pred_knn'], label='K Nearest Neighbor',sorted=True)



    if mean_selected:
        predns = model_listknn[i]['mean'].predict(group_a.iloc[:, 1:-1])
        predn_list_mean.append(predns)

        test_time_['pred_mean'] =  np.concatenate(predn_list_mean)
        
        rmse = mean_squared_error(test_time_['PM2.5'], np.concatenate(predn_list_mean)) ** 0.5
        st.write('Testing RMSE Mean', rmse)

        # ax.bar(test_time_.index, test_time_['pred_mean'], label='Mean Prediction',sorted=True)

#test_time_['pred_lr']=predn_list_lr

# -------------------------------------------------------------------------------------------
test_time_ = test_time_.drop(['Date', 'latitude', 'longitude'], axis=1)
st.write(test_time_.T)


train_time_df_i.plot.bar(ax=ax)
plt.xlabel('Stations')
plt.ylabel('PM2.5 Values')
plt.title('Predictions')
plt.xticks(rotation=90)
plt.xticks(fontsize=8)
plt.tight_layout()
#plt.show()
st.pyplot()

# Set the axis labels and title
# ax.set_xlabel('Station', color='Black')
# ax.set_ylabel('Value', color='Black')
# ax.set_title('True Y vs Pred Y', color='Black')
# ax.legend()

# Display the plot
# st.pyplot(fig)









# -------------------------------------------------------------------------------------------


