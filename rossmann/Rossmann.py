import math
import pickle
import datetime
import inflection
import numpy as np
import pandas as pd

class Rossmann(object):
    def __init__(self):
        self.home_path = ''
        self.rescaling_competition_distance   = pickle.load(open(self.home_path + "parameter/rescaling_competition_distance.pkl","rb"))
        self.rescaling_competition_time_month = pickle.load(open(self.home_path + "parameter/rescaling_competition_time_month.pkl","rb"))
        self.rescaling_promo_time_week        = pickle.load(open(self.home_path + "parameter/rescaling_promo_time_week.pkl","rb"))

        self.rescaling_year                   = pickle.load(open(self.home_path + "parameter/rescaling_year.pkl","rb"))

        self.label_encoder_store_type         = pickle.load(open(self.home_path + "parameter/label_encoder_store_type.pkl","rb"))
                   
    def data_cleaning(self, df1):
        # Excluir colunas Sales e Customers
        df1.drop(columns=[col for col in ['Sales', 'Customers'] if col in df1.columns], inplace=True)

        # Renomeando colunas
        df1.columns = [inflection.underscore(col) for col in df1.columns]

        # Convertendo a coluna date para datetime
        df1['date'] = pd.to_datetime(df1['date'])

        # Tratando coluna competition_distance 
        df1['competition_distance'] = df1['competition_distance'].apply(lambda x: 200000.0 if math.isnan(x) else x)

        # Tratando coluna competition_open_since_month
        df1['competition_open_since_month'] = df1.apply(lambda x: x["date"].month if math.isnan(x["competition_open_since_month"]) else x["competition_open_since_month"], axis = 1)

        # Tratando coluna competition_open_since_year
        df1['competition_open_since_year'] = df1.apply(lambda x: x["date"].year if math.isnan(x["competition_open_since_year"]) else x["competition_open_since_year"], axis = 1)

        # Tratando coluna promo2_since_week
        df1['promo2_since_week'] = df1.apply(lambda x: x["date"].week if math.isnan(x["promo2_since_week"]) else x["promo2_since_week"], axis = 1)

        # Tratando coluna promo2_since_year 
        df1['promo2_since_year'] = df1.apply(lambda x: x["date"].year if math.isnan(x["promo2_since_year"]) else x["promo2_since_year"], axis = 1)

        # Tratando coluna promo_interval
        month_map = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
        df1.fillna({'promo_interval': 0}, inplace=True)
        df1['month_map'] = df1['date'].dt.month.map(month_map)
        df1['is_promo'] = df1[['promo_interval', 'month_map']].apply(lambda x: 0 if x['promo_interval'] == 0 else 1 if x['month_map'] in x['promo_interval'].split(',') else 0, axis=1)

        df1['competition_open_since_month'] = df1['competition_open_since_month'].astype(int)
        df1['competition_open_since_year'] = df1['competition_open_since_year'].astype(int)
        df1['promo2_since_week'] = df1['promo2_since_week'].astype(int)
        df1['promo2_since_year'] = df1['promo2_since_year'].astype(int)

        return df1   
    
    def feature_engineering(self, df2):
        # Extraindo o ano da data
        df2['year'] = df2['date'].dt.year
        # Extraindo o mês da data
        df2['month'] = df2['date'].dt.month
        # Extraindo o dia da data
        df2['day'] = df2['date'].dt.day
        # Extraindo s semana do ano da data
        df2['week_of_year'] = df2['date'].dt.isocalendar().week
        # Year week
        df2['year_week'] = df2['date'].dt.strftime('%Y-%W')

        # Tempo em meses em que há lojas concorrentes 
        df2['competition_since'] = df2.apply(lambda x: datetime.datetime(year=x['competition_open_since_year'], month=x['competition_open_since_month'], day=1), axis=1)
        df2['competition_time_month'] = ((df2['date'] - df2['competition_since'])/30).apply(lambda x: x.days).astype(int)

        # Obter o tempo de promoção
        df2['promo_since'] = pd.to_datetime(df2['promo2_since_year'].astype(str) + df2['promo2_since_week'].astype(str) + '-1', format='%Y%W-%w') - pd.Timedelta(days=7)
        df2['promo_time_week'] = ((df2['date'] - df2['promo_since']).dt.days // 7).astype(int)

        # Renomeando as categorias das lojas a partir da variedade de produtos
        df2['assortment'] = df2['assortment'].map({'a': 'basic', 'b': 'extra', 'c': 'extended'})

        # Renomeando os feriados
        mapping = {'a': 'public_holiday', 'b': 'easter_holiday', 'c': 'christmas'}
        df2['state_holiday'] = df2['state_holiday'].map(mapping).fillna('regular_day')

        df2 = df2[df2['open']!=0]

        cols_drop = ['open', 'promo_interval', 'month_map']
        df2 = df2.drop(cols_drop, axis = 1)

        return df2
    
    def data_preparation(self, df3):
        # Rescaling
        # competition distance
        df3['competition_distance'] = self.rescaling_competition_distance.transform(df3[['competition_distance']].values)
        # competition time month   
        df3['competition_time_month'] = self.rescaling_competition_time_month.transform(df3[['competition_time_month']].values)
        # promo time week
        df3['promo_time_week'] = self.rescaling_promo_time_week.transform(df3[['promo_time_week']].values)

        # year
        df3['year'] = self.rescaling_year.transform(df3[['year']].values)

        # Store holiday - One Hot Encoding
        df3 = pd.get_dummies(df3, prefix=['state_holiday'], columns=['state_holiday'])

        # Store type - Label Encoding
        df3['store_type'] = self.label_encoder_store_type.transform(df3['store_type'])

        # Assortment - Ordinal Encoding
        assortment_dic = {'basic': 1, 'extra': 2, 'extended': 3}
        df3['assortment'] = df3['assortment'].map(assortment_dic)

        # Transformacao de natureza
        # Day of week
        df3['day_of_week_sin'] = df3['day_of_week'].apply(lambda x: np.sin(x * (2. * np.pi / 7)))
        df3['day_of_week_cos'] = df3['day_of_week'].apply(lambda x: np.cos(x * (2. * np.pi / 7)))

        # month
        df3['month_sin'] = df3['month'].apply(lambda x: np.sin(x * (2. * np.pi / 12)))
        df3['month_cos'] = df3['month'].apply(lambda x: np.cos(x * (2. * np.pi / 12)))

        # day
        df3['day_sin'] = df3['day'].apply(lambda x: np.sin(x * (2. * np.pi / 30)))
        df3['day_cos'] = df3['day'].apply(lambda x: np.cos(x * (2. * np.pi / 30)))

        # week of year
        df3['week_of_year_sin'] = df3['week_of_year'].apply(lambda x: np.sin(x * (2. * np.pi / 52)))
        df3['week_of_year_cos'] = df3['week_of_year'].apply(lambda x: np.cos(x * (2. * np.pi / 52)))

        features_selected = ['store', 'promo', 'store_type', 'assortment', 'competition_distance', 'competition_open_since_month',
                             'competition_open_since_year', 'promo2', 'promo2_since_week', 'promo2_since_year', 'competition_time_month',
                             'promo_time_week', 'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos', 'day_sin', 'day_cos',
                             'week_of_year_sin', 'week_of_year_cos'
                             ]

        return df3[features_selected]
    
    def get_prediction(self, model, original_data, test_data):
        # Predict
        pred = model.predict(test_data)

        # Join pred + original data
        original_data['prediction'] = np.expm1(pred)

        return original_data.to_json(orient='records', date_format='iso')