import streamlit as st
import pandas as pd
import numpy as np
from surprise import SVD, Dataset, Reader
from PIL import Image
import sys
import random
import pickle
import base64
import os
from pathlib import Path
import pathlib
from ImageUnderstanding import getAnswer, setImage
import websocket
from fastai.vision.all import load_learner
#import cloudinary
#import cloudinary.uploader

pathlib.PosixPath = pathlib.WindowsPath

#cloudinary.config( cloud_name = "dvjx06ia", api_key = "24594734477959", api_secret = "<your_api_secret>"  # 直接复制上面)

#def upload_image(image_file):
    #if image_file is not None:
        #upload_result = cloudinary.uploader.upload(image_file)
        #return upload_result["secure_url"]
    #return None

                        
# 定义数据导入函数
# @st.cache_data
def load_data():
    # 导入模型
    data_df = pd.read_excel('data1.xlsx', header=None)
    # 删除 columns[0]
    data_df = data_df.drop(data_df.columns[0], axis=1)
    # 重命名列名
    data_df.columns = range(data_df.shape[1])

    # 将数据框从宽格式转换为长格式
    data_df = data_df.stack().reset_index()
    # 对每列更名
    data_df.columns = ['user_id', 'sport_id', 'rating']
    # 过滤掉缺失评分
    data_df = data_df[data_df['rating'] != 0]

    # 导入运动数据，并将其存储于sport_df
    sport_df = pd.read_excel('sport1.xlsx', header=None)
    # 对该数据集中正文列增加列名“sport”
    sport_df.columns = ['sport']
    # 将索引定义为"sport_id"
    sport_df.index.name = 'sport_id'

    return data_df, sport_df

#定义训练模型函数
def train_model(data_df):
    # 建立Reader对象
    reader = Reader(rating_scale=(0, 5))

    # 将数据加载到一个Dataset对象中
    data = Dataset.load_from_df(data_df[['user_id', 'sport_id', 'rating']], reader)

    # 使用build_full_trainset()方法建立trainset对象,算法作用于整个数据集
    trainset = data.build_full_trainset()

    #在训练集上训练SVD模型
    algo = SVD()
    algo.fit(trainset)

    return algo

#定义推荐运动函数
def recommend_sports(algo, data_df, sport_df, new_user_id, new_ratings):
    new_ratings = {sport_id: info['rating'] for sport_id, info in new_ratings.items()}

    # 把新用户评分加到data里
    new_ratings_df = pd.DataFrame({
    'user_id': [new_user_id]*len(new_ratings),
    'sport_id': list(new_ratings.keys()),
    'rating': list(new_ratings.values())
    })

    #将data_df, new_ratings_df拼接起来
    data_df = pd.concat([data_df, new_ratings_df])

    #为用户生成推荐
    iids = data_df['sport_id'].unique() # 获取所有运动id的列表
    iids_new_user = data_df.loc[data_df['sport_id'] == new_user_id, 'sport_id'] # 获取被新用户评分运动id的列表
    iids_to_pred = np.setdiff1d(iids, iids_new_user) # 获取未被新用户评分运动id的列表

    # 预测所有未被评分的运动的评分
    testset_new_user = [[new_user_id, iid, 0.] for iid in iids_to_pred]
    predictions = algo.test(testset_new_user)

    # 获取预测分数最高的五种运动
    top_5_iids = [pred.iid for pred in sorted(predictions, key=lambda x: x.est, reverse=True)[:5]]
    top_5_sports = sport_df.loc[sport_df.index.isin(top_5_iids), 'sport']

    return top_5_sports

# 定义主函数
def main():
    # 导入数据
    data_df, sport_df = load_data()

    # 为新用户设置一个未被使用的user_id
    new_user_id = data_df['user_id'].max() + 1

    # 进入页面能够随机显示3种运动
    if 'initial_ratings' not in st.session_state:
        st.session_state.initial_ratings = {}
        random_sports = sport_df.sample(n=3)       #随机选取3条
        for sport_id, sport in zip(random_sports.index, random_sports['sport']):
            st.session_state.initial_ratings[sport_id] = {'sport': sport, 'rating': 3}

    # 用户评分
    for sport_id, info in st.session_state.initial_ratings.items():
        st.write(info['sport'])
        info['rating'] = st.slider('Rate this sport', 0, 5, step=1, value=info['rating'], key=f'init_{sport_id}')    # 设置一个滑动条，用户能够拖动滑动条对这3条笑话进行评分

    # 设置一个按钮“Submit Ratings”，用户在点击按钮后，能够生成对该用户推荐的5种运动
    if st.button('Submit Ratings'):
        # 把新用户评分加到data里
        new_ratings_df = pd.DataFrame({
            'user_id': [new_user_id] * len(st.session_state.initial_ratings),
            'sport_id': list(st.session_state.initial_ratings.keys()),
            'rating': [info['rating'] for info in st.session_state.initial_ratings.values()]  # Convert scale from 0-5 to -10-10
        })
        data_df = pd.concat([data_df, new_ratings_df])
        # 训练模型
        algo = train_model(data_df)

        # 基于用户对随机生成运动的评分对用户进行运动推荐
        recommended_sports = recommend_sports(algo, data_df, sport_df, new_user_id, st.session_state.initial_ratings)

        # 把推荐的运动存入session state
        st.session_state.recommended_sports = {}
        for sport_id, sport in zip(recommended_sports.index, recommended_sports):
            st.session_state.recommended_sports[sport_id] = {'sport': sport, 'rating': 3}

    # 显示推荐的运动并且评分
    if 'recommended_sports' in st.session_state:
        st.write('We recommend the following jokes based on your ratings:')
        # 显示基于用户评分所推荐的运动
        for sport_id, info in st.session_state.recommended_sports.items():
            st.write(info['sport'])
            info['rating'] = st.slider('Rate this sport', 0, 5, step=1, value=info['rating'], key=f'rec_{sport_id}')

        #设置按钮“Submit Recommended Ratings”，点击按钮生成本次推荐的分数percentage_of_total，
        #计算公式为：percentage_of_total = (total_score / 25) * 100。。
        if st.button('Submit Recommended Ratings'):
            # 综合计算本次推荐的用户满意度并且显示
            total_score = sum([info['rating'] for info in st.session_state.recommended_sports.values()])
            percentage_of_total = (total_score / 25) * 100
            st.write(f'Your percentage of total possible score: {percentage_of_total}%')

model_path =r"tupianshibie.pkl"

# 加载模型
learn_inf = load_learner(model_path)

st.title("运动识别与推荐系统")

# 上传图片
st.write("上传一张图片，应用将预测对应的标签。")

# 允许用户上传图片
uploaded_file = st.file_uploader("上传一张运动图片", type=["jpg", "jpeg", "png"])
# 获取上传图片的 URL
# image_url = upload_image(uploaded_file)

if uploaded_file is not None:
    encode = str(base64.b64encode(uploaded_file.read()), 'utf-8')
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='上传的图片', use_column_width=True)
    #st.image(image_url, width=200)
    # 图像识别
    pred, pred_idx, probs = learn_inf.predict(image)
    st.write(f"预测结果: {pred}; 概率: {probs[pred_idx]:.04f}")

    # 使用滑块获取用户评分，范围从0到5
    user_feedback_score = st.slider("请对识别结果的准确性打分（0表示非常不准，5表示非常准确）",0, 5)

    if user_feedback_score > 0:
        st.write(f"感谢您的反馈！您对识别结果的评分为: {user_feedback_score}")

        # 仅当用户进行了评分，才初始化聊天界面
        # 初始化对话历史
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # 显示聊天界面
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
            #with st.container():
                st.markdown(message["content"])

        # 用户提问
        if prompt := st.chat_input("请针对上传的图片输入您的问题"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.rerun()  # 重新运行页面以显示用户消息
            #st.experimental_rerun()

        # 在适当的位置处理用户提问并生成回答
        if st.session_state.messages and st.session_state.messages[-1]["role"] == "user":
            user_question = st.session_state.messages[-1]["content"]
            setImage(encode)
            answer = getAnswer(user_question)  # 假定getAnswer函数根据问题生成回答
            st.session_state.messages.append({"role": "assistant", "content": answer})
            st.rerun()
            #st.experimental_rerun()

        if __name__ == '__main__':
            main()
