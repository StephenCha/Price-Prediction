import pandas as pd
import numpy as np
from tqdm import tqdm
import xml.etree.ElementTree as ET
import json
import requests
import urllib
import urllib.request
import datetime
import folium

### 주산지 선정 및 관측지점 mapping
### 2019년 1년치 데이터를 기준으로 주산지 선정

filename_list = !ls ../data/public_data/train_AT_TSALET # GitHub에는 없음. 직접 만들어야 함.
filename_list_2019 = []    
for filename in filename_list :
    if '2019' in filename :
        filename_list_2019.append(filename)
        
path = '../data/public_data/train_AT_TSALET/' # GitHub에는 없음. 직접 만들고 넣어야함.
df_list = []
for filename in tqdm(filename_list_2019) :
    df = pd.read_csv(path + filename)
    df_list.append(df)
data = pd.concat(df_list, sort=False).reset_index(drop=True)

### 품목 품종별 주산지 mapping

unique_pum = [
    '배추', '무', '양파', '건고추','마늘',
    '대파', '얼갈이배추', '양배추', '깻잎',
    '시금치', '미나리', '당근',
    '파프리카', '새송이', '팽이버섯', '토마토',
]

unique_kind = [
    '청상추', '백다다기', '애호박', '캠벨얼리', '샤인마스캇'
]

joosan_dict = dict()

# 품종별 주산지 mapping
for pum in tqdm(unique_pum) :
    pum_df = data[data['PUM_NM']==pum]
    joosan_list = pum_df.groupby(['SAN_NM'])['TOT_QTY'].sum().sort_values(ascending=False).index
    for i in range(10) :
        if joosan_list[i][-1] in ['군','구','도','시'] : # 국산만 골라내기
            joosan_dict[pum] = joosan_list[i]
            break

# 품종별 주산지 mapping
for pum in tqdm(unique_kind) :
    pum_df = data[data['KIND_NM']==pum]
    joosan_list = pum_df.groupby(['SAN_NM'])['TOT_QTY'].sum().sort_values(ascending=False).index
    for i in range(10) :
        if joosan_list[i][-1] in ['군','구','도','시'] : # 국산만 골라내기
            joosan_dict[pum] = joosan_list[i]
            break
            
# print(joosan_dict)

### 주산지 - 관측지점 mapping
### 주산지의 위도 경도 구하기

y = [] #위도
x = [] #경도

Kakao_ServiceKey = open('../ServiceKey/Kakao_ServiceKey.txt', 'r').read()
headers = {"Authorization": Kakao_ServiceKey}
for san in tqdm(joosan_dict.values()) :
    url = 'https://dapi.kakao.com/v2/local/search/address.json?query=' + san
    result = json.loads(str(requests.get(url, headers=headers).text))
    match_first = result['documents'][0]['address']
    y.append(float(match_first['y'])) #위도
    x.append(float(match_first['x'])) #경도
joosan_xy = pd.DataFrame({'SAN_NM' : joosan_dict.values(), 'y' : y, 'x' : x}).reset_index()

#print(joosan_xy)

### 농업기상관측지점정보

stn_info = pd.read_csv('RDA_SPOT_INFO.csv')
stn_info['관측시작일'] = pd.to_datetime(stn_info['관측시작일'])
stn_info = stn_info[stn_info['관측시작일'].dt.year<=2015]

# print(stn_info)

### 주산지별로 농업기상관측지점과 거리 기준으로 mapping

y_san = joosan_xy['y']
x_san = joosan_xy['x']
y_stn = stn_info['위도']
x_stn = stn_info['경도']
code_dict = dict()

for y_san, x_san, san_name in zip(joosan_xy['y'], joosan_xy['x'], joosan_xy['SAN_NM']) :
    min_distance = 1000 #임의로 초기값 설정
    for y_stn, x_stn, stn_code in zip(stn_info['위도'], stn_info['경도'], stn_info['지점코드']) :
        distance = ((y_san-y_stn)**2 + (x_san-x_stn)**2)**0.5 #거리
        if distance < min_distance :
            min_distance = distance 
            stn_nearby = str(stn_code)
    code_dict[san_name] = stn_nearby
    
# print(code_dict)

### 농업기상관측지점 및 산지 분포 시각화

stn_location = stn_info[['지점명','위도', '경도']]
san_location = joosan_xy

map = folium.Map(location = [36, 128], zoom_start =7)

# 농업기상관측지점 분포 (파랑)
for index in stn_location.index:
    stn_latitude = stn_location.loc[index,"위도"]
    stn_longtitude = stn_location.loc[index,"경도"]
    stn_tooltip = stn_location.loc[index,'지점명']
    folium.Marker([stn_latitude, stn_longtitude], popup = '('+str(stn_latitude)+', '+str(stn_longtitude)+')', tooltip = stn_tooltip).add_to(map) 

# 산지 분포 (빨강)    
for index in san_location.index:    
    san_latitude = san_location.loc[index,"y"]
    san_longtitude = san_location.loc[index,"x"]
    san_tooltip = san_location.loc[index,'SAN_NM']    
    folium.Marker([san_latitude, san_longtitude], popup = '('+str(san_latitude)+', '+str(san_longtitude)+')', tooltip = san_tooltip, icon = folium.Icon(color = 'red')).add_to(map)
    
# print(map)

### 농업기상데이터 API - 월별 일 기본 관측데이터 조회
### 관측년도, 관측월, 관측지점명, 관측지점코드 값으로 조회
### 일일 트래픽 10000

CropWeather_ServiceKey = open('../ServiceKey/CropWeather_ServiceKey.txt', 'r').read()
year_list = ['2015','2016', '2017', '2018', '2019', '2020']
month_list = ['01','02','03','04','05','06','07','08','09','10','11','12']
weather = pd.DataFrame()
first_run = 0
code_list = code_dict.values() # 주산지에 mapping 된 지점에 대해서만 조회
year_error, month_error, stn_code_error, url_error, f_obs_date_error = [],[],[],[],[]
colname_dict = dict()

for stn_code in tqdm(code_list) :
    for year in year_list :
        for month in month_list :
            url = 'http://apis.data.go.kr/1390802/AgriWeather/WeatherObsrInfo/GnrlWeather/getWeatherMonDayList?'
            params = {
                'serviceKey' : CropWeather_ServiceKey, #인증키
                'Page_No' : '1', # 페이지 번호
                'Page_Size' : '31', # 한 페이지 결과 수(1~100) (31일 이내 전체 표기)
                'search_Year' : year, # 관측년도
                'search_Month' : month, #관측월
                'obsr_Spot_Code' : stn_code # 관측지점코드
            }

            # url에 params 적용하기(붙이기)
            for key, value in zip(params.keys(), params.values()):
                if key == 'serviceKey' :
                    url = url + key +'=' + value
                else :
                    url = url + '&' + key + '=' + value
            
            try :
                response = urllib.request.urlopen(url).read()
                response_string = ET.fromstring(response)

                # response - header(0) / body(1) - ...items(3) - item(0) 
                items = response_string[1][3]

                if first_run == 0 :
                    for i in items[0] :
                        colname_dict[i.tag] = [] # {'no' : [], 'stn_Code' : [], ...} 
                        first_run += 1

                # 일자별로 반복 실행
                num_days = len(items) #28 or 30 or 31
                for index in range(num_days) : 
                    # 해당하는 리스트에 원소 넣기
                    for i in items[index] :
                        colname_dict[i.tag].append(i.text)
            except :
                year_error.append(year)
                month_error.append(month)
                stn_code_error.append(stn_code)
                url_error.append(url)
                f_obs_date_error.append(stn_info[stn_info['지점코드']==stn_code].reset_index()['관측시작일'][0])
               
            
# DataFrame에 값 채워넣기             
for col in colname_dict.keys() :
    weather[col] = colname_dict[col]

# 에러 발생한 요청 모음    
error = pd.DataFrame({'year': year_error, 
                      'month': month_error,
                      'stn_code': stn_code_error,
                      'url': url_error,  
                      '관측시작일': f_obs_date_error})

# print(weather)
# print(error)

# 중복 제거
weather = weather.drop_duplicates().reset_index(drop=True)
# print(weather)

# dtype변환(object --> float)
for col in weather.columns[4:] :
    weather[col] = weather[col].astype(float)

### 전처리 (기상변수로 추가)
### 30일씩 12쿼터로 나누어서 평균 기온, 평균 습도, 누적 강수량, 이상 기후 누적 일수 등 추가

first_date = datetime.datetime.strptime('2020-09-28', '%Y-%m-%d') - datetime.timedelta(360)
date_list = [] 
for delta in range(360) :
    date = first_date + datetime.timedelta(days = delta)
    date = datetime.datetime.strftime(date, '%Y-%m-%d')
    date_list.append(date)
date_df = pd.DataFrame({'date' : date_list})
train = pd.read_csv('train.csv')
train2 = pd.concat([date_df, train], sort = False).reset_index(drop=True) #2015~2020-09-28

def weather_feature(temp_df, train2, date_df, allweather, pum, joosan_dict, code_dict, quater_days=30, num_quaters=12) :
    # 2015년도 일자 추가
    temp_df = train2[['date',f'{pum}_거래량(kg)', f'{pum}_가격(원/kg)']]
    
    # 품종과 주산지 날씨 mapping
    joosanji = joosan_dict[pum]
    joosan_code = code_dict[joosanji]
    joosan_weather = allweather[allweather['stn_Code']==joosan_code].reset_index(drop=True)
    end_index = np.where(joosan_weather['date']=='2020-09-28')[0][0]
    joosan_weather = joosan_weather.iloc[:end_index+1] #2020-09-28 까지만 자르기
    temp_df = temp_df.merge(joosan_weather, on='date', how='left')

    # weather feature 추가
    col_list = temp_df.columns[6:]
    for num in range(1,num_quaters+1) :
        for index in range(360, len(temp_df)) :
            temp_quater_df = temp_df.iloc[index-quater_days*num : index-quater_days*(num-1)] #
            quater_temp = temp_quater_df['temp']
            temp_df.loc[index, f'rain_sum_{num}q'] = temp_quater_df['rain'].sum() # 누적 강수량
            temp_df.loc[index, f'heavy_rain_count_{num}q'] = np.where(temp_quater_df['rain']>90, 1, 0).sum() # 평균 강수량 90mm 이상 누적 일수
            temp_df.loc[index, f'low_temp_count_{num}q'] = np.where(quater_temp<5, 1, 0).sum() # 일평균 기온 5도 이하 누적 일수
            temp_df.loc[index, f'middle_temp_count_{num}q'] = np.where(((quater_temp>15)&(quater_temp<22)), 1, 0).sum() # 일평균 기온 15~22도 누적 일수
            temp_df.loc[index, f'high_temp_count_{num}q'] = np.where(quater_temp>32, 1, 0).sum() # 일평균 기온 32도 이상 누적 일수
            for col in col_list :
                temp_df.loc[index, f'avg_{col}_{num}q'] = temp_quater_df[col].mean() # 각 기상 요소의 평균값
    
    drop_col_list = temp_df.columns[3:19]
    temp_df = temp_df.drop(drop_col_list, 1).reset_index(drop=True)
    temp_df = temp_df.iloc[360:].reset_index(drop=True)
    
    return temp_df

# weather_feature 함수 예시
pum = '배추'
temp_df = train[['date',f'{pum}_거래량(kg)', f'{pum}_가격(원/kg)']]
weather_feature(temp_df, train2, date_df, weather, pum, joosan_dict, code_dict, quater_days=30, num_quaters=12)