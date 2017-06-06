import time
import pandas as pd
import numpy as np
import math
from sklearn.metrics import log_loss
from scipy import sparse
import xgboost as xgb
import matplotlib.pyplot as plt
#import seaborn
from sklearn import model_selection, preprocessing, ensemble
from sklearn.metrics import log_loss
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cross_validation import StratifiedKFold
from sklearn.cluster import Birch
from sklearn.cluster import KMeans

from ast import literal_eval

import model_rfr
import model_xgb_uniclass
import model_xgb_multiclass
import model_lgbm_multiclass

OUTPUT_DIR = './layer1_04a/'

class DataFeatures():

    def __init__(self, verbose=True):
        self.featuresSet = {}
        train_df = pd.read_json('./input/train.json')
        test_df = pd.read_json('./input/test.json')
        train_df["Source"] = 'train'
        test_df["Source"] = 'test'
        self.data = pd.concat([train_df, test_df])

        exif_df = self.GetExifData()
        self.data = pd.merge(self.data, exif_df, on='listing_id', copy=True, how='left')

        magic_feat = pd.read_csv('./input/listing_image_time.csv')
        magic_feat = magic_feat.rename(columns={'Listing_Id':'listing_id', 'time_stamp':'magic_timestamp'})
        self.data = pd.merge(self.data, magic_feat, on='listing_id', how='left')

        if (verbose):
            print("Train shape: {}".format(train_df.shape))
            print("Test shape: {}".format(test_df.shape))
            #print(train_df.head())
            #print(test_df.head())
    
        if (verbose):
            print("Feature engineering")

        self.featuresSet['Basic features'] = ["bathrooms", "bedrooms", "latitude", "longitude", "price", "listing_id"]

        #####
        # Listing_id features
        #####
        minListingId = train_df['listing_id'].min()
        self.data['elapsed_time'] = self.data['listing_id'] - minListingId
        self.featuresSet['Basic - Elapsed_time'] = ["bathrooms", "bedrooms", "latitude", "longitude", "price", "elapsed_time"]

        #####
        # Latitude/Longitude features
        #####
        LOCATIONS = { #'Roscoe' : [-74.913498, 41.933144],
            'StatenIsland' : [-74.151535, 40.579021],
            'NoMad' : [-73.988976, 40.742836],
            'Midtown' : [-73.984016, 40.754932],
            'KipsBay' : [-73.980064, 40.742329],
            'AlphabetCity' : [-73.97905, 40.724545],
            'Brooklyn' : [-73.949997, 40.650002],
            'BreezyPointQueens' : [-73.926315, 40.554504],
            'CanarsieBrooklyn' : [-73.906059, 40.640232],
            'RidgewoodQueens' : [-73.897766, 40.71085],
            'JacksonHeightsQueens' : [-73.883072, 40.755684],
            'FlushingQueens' : [-73.832764, 40.768452],
            'FreshMeadowsQueens' : [-73.784866, 40.732689],
            'CentralPark' : [-73.968285, 40.785091],
            'TimesSquare' : [-73.98513, 40.758896],
        }

        features_to_use  = []
        for k, v in LOCATIONS.items():
            self.data["Distance_"+k] = (self.data['longitude']-v[0])**2 + (self.data['latitude']-v[1])**2
            self.data["Distance_"+k] = self.data["Distance_"+k].apply(math.sqrt)
            features_to_use.append("Distance_"+k)
        self.featuresSet['LatLon_my_locations'] = features_to_use[:]

        #####
        # KMeans clustering
        #####
        def kmeans_cluster(n_clusters, data):
            #split the data between "around NYC" and "other locations" basically our first two clusters
            data_c=data[(data.longitude>-74.05)&(data.longitude<-73.75)&(data.latitude>40.4)&(data.latitude<40.9)]
            data_e=data[(data.longitude<-74.05) | (data.longitude > -73.75) | (data.latitude < 40.4) | (data.latitude > 40.9)]
            r = KMeans(n_clusters, random_state=1)
            # Normalize (longitude, latitude) before K-means
            temp = data_c[['longitude', 'latitude']].copy()
            temp['longitude'] = (temp['longitude']-temp['longitude'].mean())/temp['longitude'].std()
            temp['latitude'] = (temp['latitude']-temp['latitude'].mean())/temp['latitude'].std()
            # Fit k-means and get labels
            r.fit(temp[['longitude', 'latitude']])
            data_c['kmeans_'+str(n_clusters)] = r.labels_
            data_e['kmeans_'+str(n_clusters)] = -1
            data=pd.concat([data_c,data_e])
            return data

        self.data = kmeans_cluster(20, self.data)
        temp = self.data.groupby('kmeans_20')['longitude', 'latitude'].mean()
        neig_kmeans = dict(zip(temp.index, temp.values))

        features_to_use  = []
        for k, v in neig_kmeans.items():
            neig_name = 'kmeans20_'+str(k)
            self.data["distance_"+neig_name] = (self.data['longitude']-v[0])**2 + (self.data['latitude']-v[1])**2
            self.data["distance_"+neig_name] = self.data["distance_"+neig_name].apply(math.sqrt)
            features_to_use.append("distance_"+neig_name)
        self.featuresSet['LatLon_kmeans_locations'] = features_to_use[:]
        
        #####
        # Features engineering
        #####
        self.data["price_t"] = self.data["price"]/self.data["bedrooms"]
        self.data["room_dif"] = self.data["bedrooms"]-self.data["bathrooms"]
        self.data["room_sum"] = self.data["bedrooms"]+self.data["bathrooms"]
        self.data["price_per_room"] = self.data["price"]/self.data["room_sum"]
        self.data["price_per_bedroom"] = self.data["price"]/self.data["bedrooms"]
        self.data["fold_t1"] = self.data["bedrooms"]/self.data["room_sum"]
        self.featuresSet['Features 1'] = ["price_t", "room_dif", "room_sum", "price_per_room", "price_per_bedroom", "fold_t1"]

        #####
        # count of photos
        #####
        self.data["num_photos"] = self.data["photos"].apply(len)
        self.featuresSet['Features photo'] = ['num_photos']

        #####
        # 'features' and 'description'
        #####
        self.data["num_features"] = self.data["features"].apply(len)
        self.data['num_features_stars'] = self.data['features'].apply(lambda x: ' '.join(x).count('*'))
        self.data["num_description_words"] = self.data["description"].apply(lambda x: len(x.split(" ")))
        self.data["num_description_letters"] = self.data["description"].apply(lambda x: len(x))
        self.featuresSet['Features text'] = ['num_features', 'num_features_stars', 'num_description_words', 'num_description_letters']

        #####
        # Created time features
        #####
        # convert the created column to datetime object so as to extract more features 
        self.data["created"] = pd.to_datetime(self.data["created"])
        self.data["passed"] = self.data["created"].max() - self.data["created"]
        self.data["created_year"] = self.data["created"].dt.year
        self.data["created_month"] = self.data["created"].dt.month
        self.data["created_day"] = self.data["created"].dt.day
        self.data["created_hour"] = self.data["created"].dt.hour
        self.data['created_weekday'] = self.data["created"].dt.weekday
        self.featuresSet['Time features'] = ["created_year", "created_month", "created_day", "created_hour", "created_weekday"]

        #####
        # Address features
        #####
        for f in ["display_address", "street_address"]:
            self.data[f] = self.data[f].apply(str)
            self.data[f] = self.data[f].map(lambda x: x.replace('W ', 'West'))
            self.data[f] = self.data[f].map(lambda x: x.replace('E ', 'East'))
            self.data[f] = self.data[f].map(lambda x: x.replace('N ', 'North'))
            self.data[f] = self.data[f].map(lambda x: x.replace('S ', 'South'))
            self.data[f] = self.data[f].map(lambda x: x.replace('.', ' '))
            self.data[f] = self.data[f].map(lambda x: x.replace(',', ' '))
            self.data[f] = self.data[f].map(lambda x: x.replace('(', ' '))
            self.data[f] = self.data[f].map(lambda x: x.replace(')', ' '))
            self.data[f] = self.data[f].map(lambda x: x.replace('-', ' '))
            self.data[f] = self.data[f].map(lambda x: x.replace('\'', ' '))
            self.data[f] = self.data[f].map(lambda x: x.replace('\\', ' '))
            self.data[f] = self.data[f].map(lambda x: x.replace('/', ' '))
            self.data[f] = self.data[f].map(lambda x: x.replace('"', ' '))
            self.data[f] = self.data[f].map(lambda x: x.lower())

        def extract_road_type(x):
            data = x.split()
            for d in data:
                if (d in ['blvd', 'boulevard']):
                    return 1
                elif (d in ['av', 'ave', 'aven', 'avenu', 'avenue', 'aveune']):
                    return 2
                elif (d in ['park', 'parkside', 'parkway']):
                    return 3
                elif (d in ['sreet', 'srteet', 'st', 'steet', 'stree', 'streeet', 'street', 'streets', 'stret', 'str']):
                    return 4
                elif (d in ['rd', 'road']):
                    return 5
            return 0
        self.data['street_type'] = self.data[f].map(lambda x: int(extract_road_type(x)))
        self.featuresSet['Address features'] = ["street_type"]

        #####
        # Categorical
        #####
        categorical = ["display_address", "manager_id", "building_id", "street_address"]
        features_to_use = []
        for f in categorical:
            if self.data[f].dtype=='object':
                lbl = preprocessing.LabelEncoder()
                lbl.fit(list(self.data[f].values))
                self.data[f] = lbl.transform(list(self.data[f].values))
                features_to_use.append(f)
        self.featuresSet['Categorical features'] = features_to_use[:]

        #####
        # Text features - Manual selection
        #####
        # Add the number of exclamation signs:
        self.data['num_exclamations'] = self.data['description'].apply(lambda x: len(x.split('!')))
        self.featuresSet['Features text exclamations'] = ['num_exclamations']

        strNum = {'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9}
        NumStr = {0:'zero', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five', 6:'six', 7:'seven', 8:'eight', 9:'nine'}
        def CleanText(s):
            if (isinstance(s, str)):
                s = s.lower()
                s = s.replace("  "," ")
                s = s.replace("twenty four hour", "24 hour")
                s = s.replace('24/7', '24 hour')
                s = s.replace('24-hour', '24 hour')
                s = s.replace('24hour', '24 hour')
                s = s.replace('24hr', '24 hour')
                s = s.replace('hr', 'hour')
                s = s.replace('concierge', 'doorman')
                s = s.replace('ac', 'air conditioning')
                s = s.replace('a/c', 'air conditioning')
                s = s.replace('air condition', 'air conditioning')
                s = s.replace('air conditioned', 'air conditioning')
                s = s.replace('air conditioningcepts credit cards (fee applies)', 'air conditioning')
                s = s.replace('dish washer', 'dishwasher')
                s = s.replace('dishwaser', 'dishwasher')
                s = s.replace('dishwasher/microwave', 'dishwasher microwave')
                s = s.replace('diswasher', 'dishwasher')
                s = s.replace("bicycle", "bike")
                s = s.replace(' st ', ' street ')
                s = s.replace(' ave ', ' avenue ')
                s = s.replace(' blk ', ' block ')
                s = s.replace(' blks ', ' blocks ')
                s = s.replace('pre-war', 'prewar')
                s = s.replace('publicoutdoor', 'public outdoor')
                s = s.replace('roof-deck', 'roof deck')
                s = s.replace("ss appliances", "stainless appliances")
                s = s.replace("'s", '')
                s = s.replace('.', '')
                s = s.strip()
            return s

        def FeaturesPre(x):
            newx = []
            for i in x:
                i = CleanText(i)
                data = i.split('*')
                for d in data:
                    newx.append(d.strip())
            return newx

        self.data['features'] = self.data["features"].apply(lambda x: FeaturesPre(x))

        FEATS = ['elevator', 'hardwood', 'cats', 'dogs', 'doorman', 'dishwasher', 'laundry', 'no fee', 'fitness', 'prewar', 'roof deck', 'air condition', 'dining room', 'internet', 'balcony', 'swimming', 'pool', 'new construction', 'exclusive', 'loft', 'garden', 'wheelchair', 'firepla', 'garage', 'furnished', 'multi-level', 'high ceil', 'private outdoor', 'public outdoor', 'parking', 'renovated', 'renovated', 'balcony']

        def ContainsFeat(x, f):
            for i in x:
                if (f in i):
                    return 1
            return 0

        features_to_use = []
        for f in FEATS:
            self.data['feat_'+f] = self.data['features'].apply(lambda x: ContainsFeat(x, f))
            features_to_use.append('feat_'+f)
        self.featuresSet['Features text manual'] = features_to_use[:]

        self.data['description'] = self.data["description"].apply(lambda x: CleanText(x))
        features_to_use = []
        for f in FEATS:
            self.data['desc_'+f] = self.data['description'].apply(lambda x: 1 if f in x else 0)
            features_to_use.append('desc_'+f)
        self.featuresSet['Description text manual'] = features_to_use[:]
        
        #####
        # Group by manager_id
        #####
        NEW_F = ['bathrooms', 'bedrooms', 'fold_t1', 'num_features', 'room_dif', 'num_description_letters', 'price_t', 'num_description_words', 'price', 'price_per_bedroom', 'price_per_room', 'num_photos', 'room_sum']
        features_to_use = []
        features_to_use_red = []
        train = self.data[self.data['Source']=='train']
        for f in NEW_F:
            temp = train.groupby('manager_id', as_index=False)[f].agg({f+'_per_manager_max':'max', f+'_per_manager_min':'min', f+'_per_manager_mean':'mean', f+'_per_manager_median':'median'})
            self.data = pd.merge(self.data, temp, on='manager_id', how='left')
            features_to_use.extend([f+'_per_manager_max', f+'_per_manager_min', f+'_per_manager_mean', f+'_per_manager_median'])
            features_to_use_red.extend([f+'_per_manager_mean', f+'_per_manager_median'])
        self.featuresSet['Manager group-full'] = features_to_use[:]
        self.featuresSet['Manager group-red'] = features_to_use_red[:]
        
        #####
        # Group by neighborhood
        #####
        NEW_F = ['bathrooms', 'bedrooms', 'fold_t1', 'num_features', 'room_dif', 'num_description_letters', 'price_t', 'num_description_words', 'price', 'price_per_bedroom', 'price_per_room', 'num_photos', 'room_sum']
        features_to_use = []
        features_to_use_red = []
        for f in NEW_F:
            temp = train.groupby('kmeans_20', as_index=False)[f].agg({f+'_per_kmeans_max':'max', f+'_per_kmeans_min':'min', f+'_per_kmeans_mean':'mean', f+'_per_kmeans_median':'median'})
            self.data = pd.merge(self.data, temp, on='kmeans_20', how='left')
            for c in [f+'_per_kmeans_max', f+'_per_kmeans_min', f+'_per_kmeans_mean', f+'_per_kmeans_median']:
                self.data[c+'_ratio'] = (self.data[f] - self.data[c])/(self.data[c]+1e-15)
            features_to_use.extend([f+'_per_kmeans_max', f+'_per_kmeans_min', f+'_per_kmeans_mean', f+'_per_kmeans_median', f+'_per_kmeans_max_ratio', f+'_per_kmeans_min_ratio', f+'_per_kmeans_mean_ratio', f+'_per_kmeans_median_ratio'])
            features_to_use_red.extend([f+'_per_kmeans_mean', f+'_per_kmeans_median', f+'_per_kmeans_mean_ratio', f+'_per_kmeans_median_ratio'])
        self.featuresSet['kmeans group-full'] = features_to_use[:]
        self.featuresSet['kmeans group-red'] = features_to_use_red[:]

        #####
        # Configure all of the features
        #####
        self.featuresSet['Full'] = list(self.data.columns)
        for c in self.data.columns:
            if (self.data[c].dtype == np.object):
                self.featuresSet['Full'].remove(c)
        self.featuresSet['Full'].remove('created')
        self.featuresSet['Full'].remove('passed')

        train_df['features'] = train_df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
        tfidf = CountVectorizer(stop_words='english', max_features=100)
        tr_sparse = tfidf.fit_transform(train_df["features"])

        self.data = self.data.replace([np.inf, -np.inf], np.nan)
        self.data = self.data.fillna(-9999)


    def GetExifData(self):
        exif = pd.read_csv('./input/exif_unique_gt_8k.csv', low_memory=False)
        exif['listing_id'] = exif['my_filename'].apply(lambda x: int(x.split('/')[0]))
        exif['jpeg_name'] = exif['my_filename'].apply(lambda x: x.split('/')[1])

        temp = exif.groupby('listing_id', as_index=False)['jpeg_name'].count().rename(columns={'jpeg_name': 'jpeg_count'})
        temp = temp.fillna(0)
        exif = pd.merge(exif, temp, on='listing_id', copy=True)


        def CustomFloat(x):
            try:
                return float(x)
            except:
                return -5000

        FLOAT_COL = ['Flash', 'ISOSpeedRatings']
        for c in FLOAT_COL:
            exif[c] = exif[c].fillna(-9999)
            exif[c] = exif[c].apply(lambda x: CustomFloat(x))

        def CustomComplex(x):
            if (x == -9999):
                return x
            a = literal_eval(x)
            if (float(a[1]) == 0.0):
                return -5000
            else:
                return float(a[0]) / float(a[1])

        COMPLEX_COL = ['ApertureValue', 'DigitalZoomRatio', 'ExposureBiasValue', 'ExposureTime', 'FNumber', 'FocalLength', 'FocalPlaneXResolution', 'FocalPlaneYResolution', 'MaxApertureValue', 'ShutterSpeedValue', 'XResolution', 'YResolution']
        for c in COMPLEX_COL:
            exif[c] = exif[c].fillna(-9999)
            exif[c] = exif[c].apply(lambda x: CustomComplex(x))


        CATEGORICAL = ['my_md5sum', 'ComponentsConfiguration', 'ExifVersion', 'FileSource', 'FlashPixVersion']
        for c in CATEGORICAL:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(list(exif[c].values))
            exif[c] = lbl.transform(list(exif[c].values))

        exif['DateTime'] = pd.to_datetime(exif['DateTime'], format="%Y:%m:%d %H:%M:%S", errors='coerce')
        exif['Date_year'] = exif['DateTime'].dt.year
        exif['Date_month'] = exif['DateTime'].dt.month
        exif['Date_day'] = exif['DateTime'].dt.day
        exif['Date_hour'] = exif['DateTime'].dt.hour
        exif['Date_weekday'] = exif['DateTime'].dt.weekday

        cols = []
        for c in exif.columns:
            if (exif[c].dtype == np.int) or (exif[c].dtype == np.float):
                cols.append(c)
        cols.remove('listing_id')

        exif_grouped = pd.DataFrame(exif['listing_id'].unique(), columns=['listing_id'])
        exif = exif.fillna(-9999)
        for f in cols:
            temp = exif.groupby('listing_id', as_index=False)[f].agg({f+'_max':'max', f+'_min':'min', f+'_mean':'mean', f+'_median':'median', f+'_first':'first'})

            already_in = set()
            tempcols = []
            corrtemp = temp.corr(method='pearson')
            for col in corrtemp:
                if (col not in already_in):
                    tempcols.append(col)
                    perfect_corr = corrtemp[col][corrtemp[col] > 0.9999].index.tolist()
                    perfect_corr.remove(col)
#                    print('{} - {}'.format(col, perfect_corr))
                    already_in.update(perfect_corr)
            
            exif_grouped = pd.merge(exif_grouped, temp[tempcols], on='listing_id', copy=True, how='left')

        return exif_grouped


    def GetData(self, features):        
        train = self.data[self.data['Source']=='train']
        test = self.data[self.data['Source']=='test']
        target_num_map={"high":0, "medium":1, "low":2}
        train_y = np.array(train["interest_level"].apply(lambda x: target_num_map[x]).copy())
        return sparse.csr_matrix(train[features].values), train_y, train['listing_id'].values, sparse.csr_matrix(test[features].values), test['listing_id'].values


    def GetDataWithText(self, features):
        train_df = self.data[self.data['Source']=='train']
        test_df = self.data[self.data['Source']=='test']
        target_num_map={"high":0, "medium":1, "low":2}
        train_y = np.array(train_df["interest_level"].apply(lambda x: target_num_map[x]).copy())

        train_df['features2'] = train_df['features'].apply(lambda x: " ".join([j for i in x for j in i.split()]))
        test_df['features2'] = test_df['features'].apply(lambda x: " ".join([j for i in x for j in i.split()]))

        train_df['features'] = train_df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))
        test_df['features'] = test_df["features"].apply(lambda x: " ".join(["_".join(i.split(" ")) for i in x]))

        #print(train_df["features"].head())
        tfidf = CountVectorizer(stop_words='english', max_features=500)
        tr_sparse = tfidf.fit_transform(train_df["features"])
        te_sparse = tfidf.transform(test_df["features"])

        tfidf2 = CountVectorizer(stop_words='english', max_features=500)
        tr_sparse2 = tfidf2.fit_transform(train_df["features2"])
        te_sparse2 = tfidf2.transform(test_df["features2"])

        train_X = sparse.hstack([train_df[features], tr_sparse, tr_sparse2]).tocsr()
        test_X = sparse.hstack([test_df[features], te_sparse, te_sparse2]).tocsr()

        return train_X, train_y, train_df['listing_id'].values, test_X, test_df['listing_id'].values


def CreateCorrelationMap(df, filename='corr_features.png'):
    pear_df = df.corr(method='pearson')
    vmin = pear_df.min().min()
    vmax = pear_df.max().max()
#    seaborn.heatmap(pear_df, cmap='RdYlGn_r', vmax=1.0, vmin=-1.0 , linewidths=0.5)
    seaborn.heatmap(pear_df, cmap='RdYlGn_r', vmax=vmax, vmin=vmin, linewidths=0.5, annot=True)
    plt.yticks(rotation=0)
    plt.xticks(rotation=90)
    plt.savefig(filename)


t1 = time.time()
target_txt_map={0:"high", 1:"medium", 2:"low"}

data = DataFeatures()
featGroups = {}

featGroups['full_04'] = data.featuresSet['Full']

featGroups['g1'] = data.featuresSet['Basic features'] + data.featuresSet['LatLon_my_locations'] + data.featuresSet['Features 1'] + data.featuresSet['Features photo'] + data.featuresSet['Features text'] + data.featuresSet['Time features'] + data.featuresSet['Address features'] + data.featuresSet['Categorical features'] + data.featuresSet['Features text exclamations'] + data.featuresSet['Features text manual']

for featsName, featsList in featGroups.items():
    ti1 = time.time()
    corr_matrix = pd.DataFrame()
    print("\n * {} - {}".format(featsName, featsList))

    train_x_org, train_y_org, train_id, test, test_id = data.GetData(featsList)

    n_folds = 5
    skf = list(StratifiedKFold(train_y_org, n_folds, shuffle=True, random_state=1234))

    xgb_params_reg_log = {
        'objective' : 'reg:logistic', #'binary:logistic', # 'reg:linear', 'reg:logistic'
        'silent' : 1,
        'eval_metric' : "logloss",
        'eta' : 0.01,
        'max_depth' : 5,
        'min_child_weight' : 2,
        'subsample' : 0.8,
        'colsample_bytree' : 0.5,
        'seed' : 2017,
    }
    clfs = {
        'XGB-uni-reg_log' : model_xgb_uniclass.Model,
        'RFR_150' : model_rfr.Model,
    }
    clfparams = {
        'XGB-uni-reg_log' : xgb_params_reg_log,
        'RFR_150' : {'n_estimators':150},
    }
    for modelName, clfFunc in clfs.items():
        t21 = time.time()
        out_pred = pd.DataFrame()
        out_blend_test = pd.DataFrame()
        out_blend_train = pd.DataFrame()
        for i in range(3):
            train_y = (train_y_org == i).astype(np.int8)
            clf = clfFunc(train_x_org, train_y, test, skf, log_loss, clfparams[modelName])
            pred_y, blend_train, blend_test, score = clf.Predicting()
            out_pred.insert(out_pred.shape[1], target_txt_map[i], pd.Series(pred_y))
            out_blend_test.insert(out_blend_test.shape[1], target_txt_map[i], pd.Series(blend_test))
            out_blend_train.insert(out_blend_train.shape[1], target_txt_map[i], pd.Series(blend_train))
        score = log_loss(train_y_org, out_blend_train[['high', 'medium', 'low']].values)
        out_pred['listing_id'] = test_id
        out_blend_test['listing_id'] = test_id
        out_blend_train['listing_id'] = train_id
        out_pred.to_csv(OUTPUT_DIR+"{}.{}-cv_{}-{}.csv".format(featsName, modelName, round(score, 6), "test"), index=False)
        out_blend_test.to_csv(OUTPUT_DIR+"{}.{}-cv_{}-{}.csv".format(featsName, modelName, round(score, 6), "blend_test"), index=False)
        out_blend_train.to_csv(OUTPUT_DIR+"{}.{}-cv_{}-{}.csv".format(featsName, modelName, round(score, 6), "blend_train"), index=False)
        corr_matrix.insert(corr_matrix.shape[1], modelName, out_pred['high'])
        t22 = time.time()
        print("{} - score: {}\t{} min".format(modelName, score, round((t22-t21)/60.0, 2)))

    xgb_params = {
        'objective' : 'multi:softprob',
        'silent' : 1,
        'num_class' : 3,
        'eval_metric' : "mlogloss",
        'eta' : 0.01,
        'max_depth' : 5,
        'min_child_weight' : 2,
        'subsample' : 0.8,
        'colsample_bytree' : 0.5,
        'seed' : 2017,
    }
    lgbm_p1 = {
        'boosting_type': 'gbdt',
        'objective': 'multiclass',
        'num_class':3,
        'metric':'multi_logloss',
        'nthread': -1,
        'silent': True,
        'num_leaves': 2**5,
        'learning_rate': 0.005,
        'max_depth': -1,
        'max_bin': 255,
        'subsample_for_bin': 50000,
        'subsample': 0.8,
        'subsample_freq': 1,
        'colsample_bytree': 0.5,
        'reg_alpha': 1,
        'reg_lambda': 0,
        'min_split_gain': 0.5,
        'min_child_weight': 1,
        'min_child_samples': 10,
        'scale_pos_weight': 1
        }
    clfs = {
        'XGB-p1':model_xgb_multiclass.Model(train_x_org, train_y_org, test, skf, log_loss, xgb_params),
        'lgbm_p1':model_lgbm_multiclass.Model(train_x_org, train_y_org, test, skf, log_loss, lgbm_p1),
    }

    for modelName, clf in clfs.items():
        t21 = time.time()
        pred_y, blend_train, blend_test, score = clf.Predicting()
        out_df = pd.DataFrame(pred_y)
        out_df.columns = ["high", "medium", "low"]
        out_df["listing_id"] = test_id
        out_df.to_csv(OUTPUT_DIR+"{}.{}-cv_{}-{}.csv".format(featsName, modelName, round(score, 6), "test"), index=False)
        out_df = pd.DataFrame(blend_test)
        out_df.columns = ["high", "medium", "low"]
        out_df["listing_id"] = test_id
        out_df.to_csv(OUTPUT_DIR+"{}.{}-cv_{}-{}.csv".format(featsName, modelName, round(score, 6), "blend_test"), index=False)
        out_df = pd.DataFrame(blend_train)
        out_df.columns = ["high", "medium", "low"]
        out_df["listing_id"] = train_id
        out_df.to_csv(OUTPUT_DIR+"{}.{}-cv_{}-{}.csv".format(featsName, modelName, round(score, 6), "blend_train"), index=False)
        corr_matrix.insert(corr_matrix.shape[1], modelName, out_df['high'])
        t22 = time.time()
        print("{} - score: {}\t{} min".format(modelName, score, round((t22-t21)/60.0, 2)))

    ti2 = time.time()
#    CreateCorrelationMap(corr_matrix, featsName + '.png')

#CreateCorrelationMap(solutions_h, 'corr_high.png')
#CreateCorrelationMap(solutions_m, 'corr_medium.png')
#CreateCorrelationMap(solutions_l, 'corr_low.png')

t2 = time.time()
print("Time {} min".format(round((t2-t1)/60.0, 2)))
