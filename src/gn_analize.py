import re
import os
import glob
import pandas as pd
import numpy as np
import PIL
import math
import datetime as dt
from scipy.stats import wilcoxon,shapiro,ttest_rel
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import random

'''
utility function
'''

def is_target_data(i,file):
    ret_res = False
    pat = re.compile('.*?/\d{6}/2021'+str(i))
    res = pat.match(file)
    
    # print(res)
    if res:
        ret_res=True
        
    # print(ret_res)
    return ret_res

def is_users_file(i,file,m):
    ret_res = False
    pat = re.compile('^.*?/'+m+"/"+str(i))
    res = pat.match(file)
    
    if res:
        ret_res=True
    
    # print(ret_res)
    return ret_res



def extract_location_data(log_file_path):
    # print("[デバッグ] ディレクトリ : ",log_file_path)
    loc_pat = re.compile('(.*?),lat:(.*),lon:(.*),ax:(.*?)')
    loc_data = []
    with open(log_file_path, 'r') as f:
        buffer = f.readlines()

    for line in buffer:
        match = loc_pat.match(line)
        lat = match.group(2)
        lon = match.group(3)
        loc_data.append(lat)
        loc_data.append(lon)
    
    return loc_data

def calc_walking_distance(lat1,lon1,lat2,lon2,ellipsoid=None):
    '''
    Vincenty法(逆解法)
    2地点の座標(緯度経度)から、距離を計算する
    :param lat1: 始点の緯度
    :param lon1: 始点の経度
    :param lat2: 終点の緯度
    :param lon2: 終点の経度
    :param ellipsoid: 楕円体
    :return: 距離
    '''
    # 楕円体
    ELLIPSOID_GRS80 = 1 # GRS80
    ELLIPSOID_WGS84 = 2 # WGS84
    
    # 楕円体ごとの長軸半径と扁平率
    GEODETIC_DATUM = {
        ELLIPSOID_GRS80: [
            6378137.0,         # [GRS80]長軸半径
            1 / 298.257222101, # [GRS80]扁平率
        ],
        ELLIPSOID_WGS84: [
            6378137.0,         # [WGS84]長軸半径
            1 / 298.257223563, # [WGS84]扁平率
        ],
    }
    
    # 反復計算の上限回数
    ITERATION_LIMIT = 1000
    
    # 差異が無ければ0.0を返す
    if math.isclose(lat1, lat2) and math.isclose(lon1, lon2):
        return 0.0

    # 計算時に必要な長軸半径(a)と扁平率(ƒ)を定数から取得し、短軸半径(b)を算出する
    # 楕円体が未指定の場合はGRS80の値を用いる
    a, ƒ = GEODETIC_DATUM.get(ellipsoid, GEODETIC_DATUM.get(ELLIPSOID_GRS80))
    b = (1 - ƒ) * a
    
    φ1 = math.radians(lat1)
    φ2 = math.radians(lat2)
    λ1 = math.radians(lon1)
    λ2 = math.radians(lon2)

    # 更成緯度(補助球上の緯度)
    U1 = math.atan((1 - ƒ) * math.tan(φ1))
    U2 = math.atan((1 - ƒ) * math.tan(φ2))
    
    sinU1 = math.sin(U1)
    sinU2 = math.sin(U2)
    cosU1 = math.cos(U1)
    cosU2 = math.cos(U2)

    # 2点間の経度差
    L = λ2 - λ1
    
    # λをLで初期化
    λ = L
    
    # 以下の計算をλが収束するまで反復する
    # 地点によっては収束しないことがあり得るため、反復回数に上限を設ける
    for i in range(ITERATION_LIMIT):
        sinλ = math.sin(λ)
        cosλ = math.cos(λ)
        sinσ = math.sqrt((cosU2 * sinλ) ** 2 + (cosU1 * sinU2 - sinU1 * cosU2 * cosλ) ** 2)
        cosσ = sinU1 * sinU2 + cosU1 * cosU2 * cosλ
        σ = math.atan2(sinσ, cosσ)
        sinα = cosU1 * cosU2 * sinλ / sinσ
        cos2α = 1 - sinα ** 2
        cos2σm = cosσ - 2 * sinU1 * sinU2 / cos2α
        C = ƒ / 16 * cos2α * (4 + ƒ * (4 - 3 * cos2α))
        λʹ = λ
        λ = L + (1 - C) * ƒ * sinα * (σ + C * sinσ * (cos2σm + C * cosσ * (-1 + 2 * cos2σm ** 2)))

        # 偏差が.000000000001以下ならbreak
        if abs(λ - λʹ) <= 1e-12:
            break
    else:
        # 計算が収束しなかった場合はNoneを返す
        return None
    
    # λが所望の精度まで収束したら以下の計算を行う
    u2 = cos2α * (a ** 2 - b ** 2) / (b ** 2)
    A = 1 + u2 / 16384 * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
    B = u2 / 1024 * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))
    Δσ = B * sinσ * (cos2σm + B / 4 * (cosσ * (-1 + 2 * cos2σm ** 2) - B / 6 * cos2σm * (-3 + 4 * sinσ ** 2) * (-3 + 4 * cos2σm ** 2)))
    
    # 2点間の楕円体上の距離
    s = b * A * (σ - Δσ)
    

    return s

def calc_distance(loc_data):
    i = 0
    dist = 0

    length = int(len(loc_data)/2)-1
    while i <= length:
        if length == 0:
            break

        lat1 = float(loc_data[i])
        lon1 = float(loc_data[(i+1)])
        lat2 = float(loc_data[(i+2)])
        lon2 = float(loc_data[(i+3)])

        dist += calc_walking_distance(lat1,lon1,lat2,lon2)
        # print(dist)
        i += 2
        
    return dist

def truncate_float_number(val):
    interval = 0.0002
    digit = 5
    win_val = math.floor(float(val)/interval)
    return round(interval * win_val*10**digit)/(10**digit)

def calc_distance_with_bool(data):
    
    i = 0
    dist = 0
    bot_dist = 0
    user_dist = 0
    length = int(len(data)/2)-1
    bot_get_erea = 0
    user_get_erea = 0
    while i <= length:
        if length == 0:
            break

        lat1 = float(data[i])
        lon1 = float(data[(i+1)])
        lat2 = float(data[(i+2)])
        lon2 = float(data[(i+3)])

        dist += calc_walking_distance(lat1,lon1,lat2,lon2)
        
        truc_lat1 = truncate_float_number(lat1)
        truc_lon1 = truncate_float_number(lon1)
        truc_lat2 = truncate_float_number(lat2)
        truc_lon2 = truncate_float_number(lon2)
        
        bot_dist += dist
        user_dist += dist
            
        if lat1 == lat2 and lon1 == lon2 and bot_dist >= 10:
            bot_get_erea += 1
            bot_dist = 0
            

        if lat1 == lat2 and lon1 == lon2 and user_dist >= 5:
            user_get_erea += 1
        
        i += 2
        
    return dist,user_get_erea,bot_get_erea

def extract_time_data(filename):
    # print(filename)
    time_pat = re.compile('(\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}.\d{3}),(\d{3})')
    time_data = []
    with open(filename,'r') as f:
        buffer = f.readlines()
    length = len(buffer)
    match_init = time_pat.match(buffer[0])
    match_end = time_pat.match(buffer[length-1])
    start_time = match_init.group(1)
    end_time = match_end.group(1)
    # print("開始時刻 : "+start_time)
    # print("終了時刻 : "+end_time)
    start_time = start_time.replace("_"," ")
    end_time = end_time.replace("_"," ")
    # print("開始時刻 : "+match_init.group(0))
    # print("終了時刻 : "+match_end.group(0))
    # dt1 = dt.datetime.strptime(start_time,'%Y-%m-%d %H:%M:%S.%f')
    # dt2 = dt.datetime.strptime(end_time,'%Y-%m-%d %H:%M:%S.%f')
    return start_time,end_time

def calc_time_diff(start_time,end_time):
    dt1 = dt.datetime.strptime(start_time,'%Y-%m-%d %H:%M:%S.%f')
    dt2 = dt.datetime.strptime(end_time,'%Y-%m-%d %H:%M:%S.%f')
    return dt2-dt1

'''
データ分析処理 歩行データ投稿傾向
'''

def posted_data_per_day(data,daya,dayb):
    result = []
    # print(len(daya))
    for i in range(len(daya)):
        post_count = 0
        for j in range(len(data)):
            if is_target_data(daya[i],data[j]):
                post_count += 1
            
            if is_target_data(dayb[i],data[j]):
                post_count += 1
                
        result.append(post_count)
        
    return result


def posted_data_per_day_and_user(data,daya,dayb,id,mode):
    res = []

    for i in range(len(id)):
        col = []
        for j in range(len(daya)):
            post_count = 0
            for k in range(len(data)):
                if is_users_file(id[i],data[k],mode) and is_target_data(daya[j],data[k]):
                    # print(id[i],daya[j],data[k])
                    post_count += 1
                if is_users_file(id[i],data[k],mode) and is_target_data(dayb[j],data[k]):
                    # print(id[i],daya[j],data[k])
                    post_count += 1
            
            col.append(post_count)
        res.append(col)            
    return res
                    
                
    
        
'''
データ分析処理 歩行距離　
'''
def calc_dist_walking_data(i,file):
    loc_data = extract_location_data(file)
    dist = calc_distance(loc_data)
    
    return dist

def calc_dist_walking_data_per_files(file):
    loc_data = extract_location_data(file)
    dist,user,bot = calc_distance_with_bool(loc_data)
        
    
    return dist,user,bot
    
def walking_dis_per_day(data,daya,dayb):
    result = []
    for i in range(len(daya)):
        dist = 0
        for j in range(len(data)):
            if is_target_data(daya[i],data[j]):
                dist += calc_dist_walking_data(daya[i],data[j])
            
            if is_target_data(dayb[i],data[j]):
                dist += calc_dist_walking_data(dayb[i],data[j])
                
        result.append(dist)
        
    return result

def walking_dis_per_day_and_user(data,daya,dayb,id,mode):
    res = []

    for i in range(len(id)):
        col = []
        for j in range(len(daya)):
            dist = 0
            for k in range(len(data)):
                if is_users_file(id[i],data[k],mode) and is_target_data(daya[j],data[k]):
                    # print(id[i],daya[j],data[k])
                    dist += calc_dist_walking_data(daya[j],data[k]) 
                if is_users_file(id[i],data[k],mode) and is_target_data(dayb[j],data[k]):
                    # print(id[i],daya[j],data[k])
                    dist += calc_dist_walking_data(dayb[j],data[k])
            
            col.append(dist)
        res.append(col)            
    return res

def calc_walking_distance_per_data(data,id,mode):
    res = []
    
    # print(len(data))
    for i in id:
        for j in data:
            col = []
            if is_users_file(i,j,mode):
                col.append(i)
                col.append(j)
                dist,user,bot = calc_dist_walking_data_per_files(j)
                col.append(dist)
                col.append(user)
                col.append(bot)
                res.append(col)

    df = pd.DataFrame(res,columns=['user_id','file','distance','user_erea','bot_erea'])
    df.to_excel('歩行距離_ファイル毎.xlsx')
    df.to_csv('歩行距離_ファイル毎.csv')
    
    return res

'''
データ分析処理 歩行時間
'''
def calc_time_walking_data(i,file):
    time_data1,time_data2 = extract_time_data(file)
    time_diff = calc_time_diff(time_data1,time_data2)
    
    return time_diff.total_seconds()

def walking_time_per_day(data,daya,dayb):
    result = []
    for i in range(len(daya)):
        sum_time = dt.timedelta()
        for j in range(len(data)):
            if is_target_data(daya[i],data[j]):
                sum_time += calc_time_walking_data(daya[i],data[j])
            
            if is_target_data(dayb[i],data[j]):
                sum_time += calc_time_walking_data(dayb[i],data[j])
                
        result.append(sum_time.total_seconds())
        
    return result

def walking_time_per_day_and_user(data,daya,dayb,id,mode):
    res = []

    for i in range(len(id)):
        col = []
        for j in range(len(daya)):
            dist = 0
            for k in range(len(data)):
                if is_users_file(id[i],data[k],mode) and is_target_data(daya[j],data[k]):
                    # print(id[i],daya[j],data[k])
                    dist += calc_time_walking_data(daya[j],data[k]) 
                if is_users_file(id[i],data[k],mode) and is_target_data(dayb[j],data[k]):
                    # print(id[i],daya[j],data[k])
                    dist += calc_time_walking_data(dayb[j],data[k])
            
            col.append(dist)
        res.append(col)            
    return res



'''
メインルーチン
'''
def post_sum(data):
    sum = 0
    for i in data:
        sum += i
        
    return sum

def print_all(data):
    for i in data:
        print(i)
        
def print_array(data,id):
    u = 0
    for i in data:
        d = 1
        for j in i:
            print("user:{} days:{} $ {}".format(id[u],d,j))
            d += 1
        u += 1
            

def times_sum(data):
    sum = dt.timedelta()
    for i in data:
        sum += i
        
    return sum.total_seconds()

def array_sum(data):
    all_total = 0
    for i in data:
        total = 0
        for j in i:
            total += j
        all_total += total
        
    print(all_total)

def array_sum_time(data):
    all_total = dt.timedelta()
    for i in data:
        total = dt.timedelta()
        for j in i:
            total += j
        all_total += total
        
    print(all_total.total_seconds())     

def shape_data(id,mode):
    res = []
    count = 0 
    for i in id:
        data = glob.glob("./data20220416/{}/{}/*/*.log".format(mode,i),recursive=True)
        # print(len(data))
        # print(data)
        count += len(data)
        for j in data:
            # print(j)
            res.append(j)
        # res.append(data)
    
    # print(count)
    # print(res)
    return res

def print_virtul(data,id):
    
    files = 0
    bot = 0
    user = 0
    res = []
    for i in id:
        col = []
        files = 0
        bot = 0
        user = 0
        for j in data:
            if i == j[0]:
                files += 1
                if j[2] >= 5:
                    user += 1
                if j[2] >= 10:
                    bot += 1
        
        col.append(files)
        col.append(user)
        col.append(bot)
        res.append(col)
    
    files = 0
    bot_cont = 0 
    for i in res:
        files += i[0]
        bot_cont += i[2]
        print("files: {} bot get areas: {} user get areas: {}".format(i[0],i[2],i[1]))
    print("files: {}".format(files))
    print("bot: {}".format(bot_cont))
    print("--------------------------------------")
    # print("files: {} bot get areas: {} user get areas: {}".format(files,bot,user))

def measure_data_per_days():
    id = [196,201,198,202,200,199,194,203]
    daya = [1020,1021,1022,1023,1024,1025,1026]
    dayb = [1027,1028,1029,1030,1031,1101,1102]
    gwalker_walking_data = glob.glob("./data20220416/g_walker/*/*/*.log",recursive=True)
    walker_walking_data = glob.glob("./data20220416/walker/*/*/*.log",recursive=True)

    walker_data = shape_data(id,"walker")
    gwalker_data = shape_data(id,"g_walker")
    # print(walker_data)
    # print(gwalker_data)
    # print(len(walker_data))
    # print(len(gwalker_data))
    # pdpdw =  posted_data_per_day(walker_data,daya,dayb)
    # pdpdgw =  posted_data_per_day(gwalker_data,daya,dayb)
    
    
    # wdpdw =  walking_dis_per_day(walker_data,daya,dayb)
    # wdpdgw =  walking_dis_per_day(gwalker_data,daya,dayb)
    
    # wtpdw =  walking_time_per_day(walker_data,daya,dayb)
    # wtpdgw =  walking_time_per_day(gwalker_data,daya,dayb)
    # print(gwalker_data)
    # print(walker_data)
    
    print('------- 歩行データ投稿数　時系列 -----------')
    # print(pdpdw)
    # print(len(pdpdw))
    # print(post_sum(pdpdw))
    
    # print(pdpdgw)
    # print(len(pdpdgw))
    # print(post_sum(pdpdgw))
    
    # print_all(pdpdw)
    # print_all(pdpdgw)
    
    print('------- 歩行距離　時系列 -----------')
    
    # print(wdpdw)
    # print(len(wdpdw))
    # print(post_sum(wdpdw))
    
    # print(wdpdgw)
    # print(len(wdpdgw))
    # print(post_sum(wdpdgw))
    
    # print_all(wdpdw)
    # print_all(wdpdgw)
    
    print('------- 歩行時間　時系列 -----------')

    # print(wtpdw)
    # print(len(wtpdw))
    # print(post_sum(wtpdw))
    
    # print(wtpdgw)
    # print(len(wtpdgw))
    # print(post_sum(wtpdgw))

    # print_all(wtpdw)
    # print_all(wtpdgw)
    

def measure_data_per_days_and_user(flag,n):
    id = [196,201,198,202,200,199,194,203]
    daya = [1020,1021,1022,1023,1024,1025,1026]
    dayb = [1027,1028,1029,1030,1031,1101,1102]
    gwalker_walking_data = glob.glob("./data20220416/g_walker/*/*/*.log",recursive=True)
    walker_walking_data = glob.glob("./data20220416/walker/*/*/*.log",recursive=True)
    mode_w = "walker"
    mode_gw = "g_walker"

    walker_data = shape_data(id,"walker")
    gwalker_data = shape_data(id,"g_walker")
    # print(walker_data)
    # print(len(walker_data))
    # print(len(gwalker_data))
    pdpdw =  posted_data_per_day_and_user(walker_data,daya,dayb,id,mode_w)
    pdpdgw =  posted_data_per_day_and_user(gwalker_data,daya,dayb,id,mode_gw)
    
    
    
    wdpdw =  walking_dis_per_day_and_user(walker_data,daya,dayb,id,mode_w)
    wdpdgw =  walking_dis_per_day_and_user(gwalker_data,daya,dayb,id,mode_gw)
    
    wtpdw =  walking_time_per_day_and_user(walker_data,daya,dayb,id,mode_w)
    wtpdgw =  walking_time_per_day_and_user(gwalker_data,daya,dayb,id,mode_gw)
    
    # wdpfile_w = calc_walking_distance_per_data(walker_data,id)
    wdpfile_gw = calc_walking_distance_per_data(gwalker_data,id,mode_gw)
    # print(gwalker_data)
    # print(walker_data)
    
    print('------- 歩行データ投稿数　時系列 -----------')
    # print(pdpdw)
    # print(len(pdpdw))
    # print(post_sum(pdpdw))
    # pdpdw.to_csv('歩行データ投稿数_walker.csv')
    # pdpdgw.to_csv('歩行データ投稿数_gwalker.csv')
    # print(pdpdgw)
    # print(len(pdpdgw))
    # print(post_sum(pdpdgw))
    
    # print_all(pdpdw)
    # print_all(pdpdgw)
    # print(pdpdgw[1])
    # print_array(pdpdw,id)
    # print_array(pdpdgw,id)
    # array_sum(pdpdw)
    # array_sum(pdpdgw)
    
    print('------- 歩行距離　時系列 -----------')
    
    # print(wdpdw)
    # print(len(wdpdw))
    # print(post_sum(wdpdw))
    # wdpdw.to_csv('歩行距離_walker.csv')
    # wdpdgw.to_csv('歩行距離_gwalker.csv')
    # print(wdpdgw)
    # print(len(wdpdgw))
    # print(post_sum(wdpdgw))
    
    # print_all(wdpdw)
    # print_all(wdpdgw)
    # print_array(wdpdw,id)
    # print_array(wdpdgw,id)
    # array_sum(wdpdw)
    # array_sum(wdpdgw)
    
    # print_virtul(wdpfile_w)
    print_virtul(wdpfile_gw,id)
    
    print('------- 歩行時間　時系列 -----------')

    # print(wtpdw)
    # print(len(wtpdw))
    # print(post_sum(wtpdw))
    
    # print(wtpdgw)
    # print(len(wtpdgw))
    # print(post_sum(wtpdgw))
    # wtpdw.to_csv('歩行時間_walker.csv')
    # wtpdgw.to_csv('歩行時間_gwalker.csv')
    # print_all(wtpdw)
    # print_all(wtpdgw)
    # print_array(wtpdw,id)
    # print_array(wtpdgw,id)
    # array_sum(wtpdw)
    # array_sum(wtpdgw)
    
    if flag == 1 and n == 1:
        res1 = pdpdw
        res2 = pdpdgw
        res3 = id
        
    elif flag == 1 and n == 2:
        res1 = wdpdw
        res2=wdpdgw
        res3 = id
        
    elif flag == 1 and n == 3:
        res1 = wtpdw
        res2 = wtpdgw
        res3 = id
        
    else:
        res1 = None
        res2 = None
        res3 = None
        
    return res1,res2,res3

def create_dataframe(data1,data2,user):
    mode = "Mode"
    df1 = pd.DataFrame(data=data1,columns=['1 day','2 day','3 day','4 day','5 day','6 day','7 day'])
    # df = pd.DataFrame(data=data1)
    df1 = df1.assign(mode="Walker")
    
    # print(len(df))
    df2= pd.DataFrame(data2,columns=['1 day','2 day','3 day','4 day','5 day','6 day','7 day'])
    # print(df)
    df2 = df2.assign(mode="Gaming walker")
    
    df = df1.append(df2,ignore_index=True)
    return df
    
def plot_data(data1,data2,user,fn):
    
    df = create_dataframe(data1,data2,user)
    # print(df)
    # print(df.columns.values)
    cols=df.columns.values
    data = []
    col = []
    for j in range(len(df.columns)-1):
        for i in range(len(df.index)):
            col = []
            if i <= 7:
                col.append("Walker")
            else:
                col.append("Gaming walker")
                
            col.append(cols[j])
            col.append(df.iloc[i,j])
            data.append(col)
            # print(df.iloc[i,j])
    
    plot_df = pd.DataFrame(data,columns=['mode','date','value'])

    print(plot_df.head())
    # for i in range(len(df)):
    #     for j in range(len(df.iloc[i])):
    #         print(df.iloc[i,j])
    
    fig = plt.figure(figsize=(12,8))
    sns.boxplot(x="date",y="value",data=plot_df,hue="mode")
    plt.title(fn)
    if os.path.exists('./img') is not True:
        os.mkdir('./img')
    plt.savefig("./img/{}.png".format(fn))
    plt.show()
    
def test_data(d1,d2):

    
    row_len = len(d1)
    col_len = len(d1[0])
    print(row_len)
    print(col_len)
    for j in range(col_len):
        testA = []
        testB = []
        for i in range(row_len):
            testA.append(d1[i][j])
            testB.append(d2[i][j])
        
        A = np.array(testA)
        B = np.array(testB)
        # print(A)
        # print(B)
        is_standard_distribution_A =shapiro(A)
        is_standard_distribution_B = shapiro(B)
        
        # print("shapiro A : {}".format(is_standard_distribution_A))
        # print("shapiro B : {}".format(is_standard_distribution_B))
        
        # print("{} days : {}")
        
        p = wilcoxon(A,B,correction=True)
        t = ttest_rel(A,B)
        print("{} days : {}".format(j+1,p))
        # print("{} days : {}".format(j+1,t))

    
def main():
    # measure_data_per_days()
    is_matplotlib=0
    is_test=0
    data1,data2,user = measure_data_per_days_and_user(0,3)
    if (data1 is not None or data2 is not None) and is_matplotlib == 1:
        plot_data(data1,data2,user,"walking_time")
    if (data1 is not None or data2 is not None) and is_test == 1:
        test_data(data1,data2)
    
if __name__ == "__main__":
    main()