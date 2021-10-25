import csv
import os
import numpy as np
import pandas as pd
import datetime
from natsort import natsorted

def filename_load(dirname): # 해당 디렉토리 내의 파일명들 모두를 natural sorted 된 list로 반환해 주는 함수.
    filename_array = []
    filenames = os.listdir(dirname)
    for filename in filenames:
        full_filename = os.path.join(dirname, filename)
        filename_array.append(full_filename)

    filename_array = natsorted(filename_array) # natural sorting

    return filename_array


class AreaReader(): # AT_TSALET_ALL이 모여있는 디렉토리를 받아 주산지를 반환하는 코드
    def __init__(self, directory):
        self.directory = directory
        self.DATA = {} # 날짜에 따른 전체 데이터 딕셔너리
        self.DATA_best = {} # self.DATA에서 주산지만 모아놓은 리스트
        self.pum_list = ['배추', '무', '양파', '건고추', '마늘', '대파', 
        '얼갈이배추', '양배추', '깻잎', '시금치', '미나리', '당근', 
        '파프리카', '새송이', '팽이버섯', '토마토', '청상추', 
        '백다다기', '애호박', '캠벨얼리', '샤인마스캇'] # 과일 품목 리스트. 주산지 계산에 쓰임.
        file_list = filename_load(self.directory)

        for file_name in file_list: # file_list 안에 있는 파일 하나를 읽는다.
            print(file_name)
            file = open(file_name, 'r', encoding='utf-8')
            rdr = csv.reader(file)
            
            itr = -1
            for line in rdr: # 파일 라인 따라 읽는다.
                if itr == -1:
                    name_list = line
                
                if itr >= 0:
                    sale_date = line[0] # 경락 일자
                    pum_nm = line[3] # 품목
                    san_nm = line[9] # 산지
                    tot_qty = line[13] # 총 물량

                    if self.DATA.get(sale_date) == None: # 날짜가 없는 경우
                        self.DATA[sale_date] = {}

                        if self.DATA[sale_date].get(pum_nm) == None: # 품목이 없는 경우
                            self.DATA[sale_date][pum_nm] = {}
                            #if pum_nm not in self.pum_list: # pum list에도 품목이 없는 경우(다른 파일(다른 월)로 넘어가면서 새로운 작물이 등장할 수도 있으니까.)
                            #    self.pum_list.append(pum_nm) # 품목 리스트에 품목 추가

                            if self.DATA[sale_date][pum_nm].get(san_nm) == None: # 산지가 없는 경우
                                self.DATA[sale_date][pum_nm][san_nm] = float(tot_qty)
                        
                    else: # 날짜가 있는 경우
                        if self.DATA[sale_date].get(pum_nm) == None: # 품목이 없는 경우
                            self.DATA[sale_date][pum_nm] = {}
                            #if pum_nm not in self.pum_list: # pum list에도 품목이 없는 경우(다른 파일(다른 월)로 넘어가면서 새로운 작물이 등장할 수도 있으니까.)
                            #    self.pum_list.append(pum_nm) # 품목 리스트에 품목 추가

                            if self.DATA[sale_date][pum_nm].get(san_nm) == None: # 산지가 없는 경우
                                self.DATA[sale_date][pum_nm][san_nm] = float(tot_qty)
                        
                        else: # 품목이 있는 경우
                            if self.DATA[sale_date][pum_nm].get(san_nm) == None: # 산지가 없는 경우
                                self.DATA[sale_date][pum_nm][san_nm] = float(tot_qty)
                            
                            else: # 산지가 있는 경우 -> 해당 산지에 기존 값은 무조건 존재
                                self.DATA[sale_date][pum_nm][san_nm] += float(tot_qty) # 기존 qty에 qty를 더해준다.
                itr += 1
                
        # print(self.DATA['2020-11-06'])
        
        ############ dictionary 에서 값 추출을 위한 date module ###########
        start_date = datetime.date(2020, 11, 6) # 시작일 -> 최종 파일 입력할 때 바꿔줘야 할 파라미터
        end_date = datetime.date(2021, 8, 21) # 마지막일 -> 최종 파일 입력할 때 바꿔줘야 할 파라미터
        day_count = (end_date - start_date).days + 1
        date_list = []
        for single_date in (start_date + datetime.timedelta(n) for n in range(day_count)):
            date_list.append(single_date.strftime("%Y-%m-%d")) # date list에 값을 추가한다
        ###################################################################

        # date에 따라서 dictionary를 탐색하며 최고 값만 따로 self.DATA_best에 저장한다.
        for date in date_list: # date list로 dictionary 탐색 시작
            self.DATA_best[date] = {} # date에 해당하는 빈 dictionary를 먼저 만든다.
            date_ = date.replace('-', '') # 파일이 바뀌면서 날짜를 0000-00-00 에서 00000000 형태로 바꿔 표기하는 문제를 해결해야한다.

            for pum in self.pum_list: # pum list로 우선 dictionary에 빈 data 넣어준다.
                self.DATA_best[date][pum] = None

            for pum in self.pum_list: # pum list로 dictionary 탐색 시작
                if self.DATA.get(date) != None: # 해당 날짜에 값이 존재하면(0000-00-00 형태로)
                    if self.DATA[date].get(pum) != None: # 해당 date에 해당 품종이 존재할 때,
                        area = max(self.DATA[date][pum], key=self.DATA[date][pum].get) #가장 높은 value를 기준으로 주산지(area)를 산출하고
                        self.DATA_best[date][pum] = area # 주산지를 date dictionary의 품종 값으로 넣는다.
            
                if self.DATA.get(date_) != None: # 해당 날짜에 값이 존재하면(00000000 형태로)
                    if self.DATA[date_].get(pum) != None: # 해당 date에 해당 품종이 존재할 때,
                        area = max(self.DATA[date_][pum], key=self.DATA[date_][pum].get) #가장 높은 value를 기준으로 주산지(area)를 산출하고
                        self.DATA_best[date][pum] = area # 주산지를 date dictionary의 품종 값으로 넣는다.(쓸 때는 0000-00-00 형태로 일관되게 써준다)
        print(self.DATA_best)

    def getArea(self, date, breed): # 날짜는 0000-00-00 형태, breed는 품종
        return self.DATA_best[date][breed]



# 아래와 같은 방식으로 getArea 함수를 이용해 사용하면 됨

dir = './data/AT_TSALET_ALL_files/AT_TSALET_ALL'
areareader = AreaReader(dir)
print(areareader.getArea('2020-11-06', '배추'))
print(areareader.getArea('2020-11-07', '무'))
print(areareader.getArea('2020-11-08', '양파'))
print(areareader.getArea('2020-11-09', '건고추'))
print(areareader.getArea('2020-11-10', '마늘'))
print(areareader.getArea('2020-11-11', '대파'))
print(areareader.getArea('2020-11-12', '얼갈이배추'))
print(areareader.getArea('2020-11-13', '양배추'))
print(areareader.getArea('2020-11-14', '깻잎'))
print(areareader.getArea('2020-11-15', '시금치'))
print(areareader.getArea('2020-11-16', '미나리'))
print(areareader.getArea('2020-11-17', '당근'))
print(areareader.getArea('2020-11-18', '파프리카'))
print(areareader.getArea('2020-11-19', '새송이'))
print(areareader.getArea('2020-11-20', '팽이버섯'))
print(areareader.getArea('2020-11-21', '토마토'))
print(areareader.getArea('2020-11-22', '상추'))
print(areareader.getArea('2020-11-23', '오이'))
print(areareader.getArea('2020-11-24', '호박'))
print(areareader.getArea('2020-11-25', '포도'))
print(areareader.getArea('2020-11-26', '포도'))
print(areareader.getArea('2020-11-27', '배추'))
print(areareader.getArea('2020-11-28', '배추'))
print(areareader.getArea('2020-11-29', '배추'))
print(areareader.getArea('2020-11-30', '배추'))
print(areareader.getArea('2020-12-01', '배추'))

# 매우 중요한 사실: '포도 = 샤인마스캇, 포도 = 캠벨얼리, 호박 = 애호박, 오이 = 백다다기, 청상추 = 상추' 이다.