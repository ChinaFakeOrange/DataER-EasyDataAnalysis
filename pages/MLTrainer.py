# MIT License

# Copyright (c) 2024 ChinaFakeOrange

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import pandas as pd
import streamlit as st
from pages.utils.ml_util import *
import sys
from streamlit_lottie import st_lottie

def ML_trainer():
    st.set_page_config(page_title='数据探索分析', page_icon=None, layout="wide")
    file_create('ml_saved\\model')
    file_create("ml_saved\\res")
    file_create("ml_saved\\plot")
    ml_list = os.listdir('ml_saved\\model')
    full_ml_path = [os.path.join('ml_saved\\model',i) for i in ml_list]
    load_model_path =dict(zip(ml_list,full_ml_path))
    ml_list_up = ml_list
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    fig1,fig2,pred_res = None,None,None
    def load_data(file_path, ftt, sh, h):
        if ftt == 'Excel':
            try:
                data = pd.read_excel(file_path, header=h, sheet_name=sh, engine='openpyxl')
            except:
                st.info("这不是Excel文件")
                sys.exit()
        elif ftt == 'csv':
            try:
                data = pd.read_csv(file_path)
            except:
                st.info("这不是CSV文件")
                sys.exit()
        return data

    st.header('机器学习训练器')
    st.divider()
    ftt = st.sidebar.selectbox("请选择文件类型", ["Excel", "csv"],index=0)
    train_up = st.sidebar.file_uploader('上传训练数据')
    test_up = st.sidebar.file_uploader('上传预测数据')

    train = pd.DataFrame()
    if train_up is not None:
        file_path = train_up

        if ftt == 'Excel':
            try:
                # User prompt to select sheet name in uploaded Excel
                sh = st.sidebar.selectbox("*请选择需要阅读哪一个Sheet*",
                                          pd.ExcelFile(file_path).sheet_names)
                # User prompt to define row with column names if they aren't in the header row in the uploaded Excel
                h = st.sidebar.number_input("*请选择包含列名称的那一行*", 0, 100)
            except:
                st.info("这不是Excel文件")
                sys.exit()

        elif ftt == 'csv':
            try:
                # No need for sh and h for csv, set them to None
                sh = None
                h = None
            except:
                st.info("这不是CSV文件")
                sys.exit()

        train = load_data(file_path, ftt, sh, h)
    if test_up is not None:
        file_path = test_up

        if ftt == 'Excel':
            try:
                # User prompt to select sheet name in uploaded Excel
                sh = st.sidebar.selectbox("*请选择需要阅读哪一个Sheet*",
                                          pd.ExcelFile(file_path).sheet_names)
                # User prompt to define row with column names if they aren't in the header row in the uploaded Excel
                h = st.sidebar.number_input("*请选择包含列名称的那一行*", 0, 100)
            except:
                st.info("这不是Excel文件")
                sys.exit()

        elif ftt == 'csv':
            try:
                # No need for sh and h for csv, set them to None
                sh = None
                h = None
            except:
                st.info("这不是CSV文件")
                sys.exit()

        test = load_data(file_path, ftt, sh, h)


    #任务参数——————————————————
    with st.container():
        with st.container():
            mcol1, mcol2,mcol3,mcol4 = st.columns(4)
            with mcol1:
                mission = st.selectbox(label='任务类型',options=['回归任务','分类任务'])
            with mcol2:
                if mission == "回归任务":
                    task_tp = st.selectbox(label='损失标准',options=['RMSE','MAE','MSE','R2'],index=0)
                else:
                    task_tp=st.selectbox(label='损失标准', options=['F1','AUC','ACC'],index=0)
            with mcol3:
                times_hp = st.slider(min_value=1,max_value=25,label='循环调试次数',value=5)
            with mcol4:
                meta_w = st.slider(min_value=0.0,max_value=1.0,label='模型混合权重',value=0.5,step=0.01)
        #优化参数——————————————————
        with st.container():
            ocol1, ocol2,ocol3,ocol4 = st.columns(4)
            with ocol1:
                # a,b,c,d,e
                pre_set = st.multiselect(label='数据优化',options=["载入模型", '数据平衡',"共线性处理",
                                      "自选变量",'变量扩展'])
            with ocol2:
                dp_var = st.multiselect(label='移除变量', options=train.columns.tolist())
            with ocol3:
                target = st.selectbox(label='目标变量', options=train.columns.tolist(),index=len(train.columns.tolist())-1)
            with ocol4:
                test_ratio = st.slider(label='测试数据比例',min_value=0.01,max_value=0.99,value=0.25)

        with st.container():
            col1, col2,col3,col4 = st.columns([1,1,1,1])
            with col1:
                train_bt = st.button('训练',use_container_width=True)

            with col2:
                pred_bt = st.button('预测',use_container_width=True)
            with col3:
                if ml_list_up == ml_list:
                    load_model = st.selectbox('选择载入的模型',ml_list)
                else:
                    load_model = st.selectbox('选择载入的模型', ml_list_up)
            with col4:
                de_bt = st.button('删除当前模型')

            if de_bt:
                os.remove(load_model_path[load_model])

            if train_bt:
                try:
                    (res_txt, fig1, fig2,
                     load_model_path,ml_list_up) = ml_pip(train,dp_var,target,pre_set,
                                                       mission,task_tp,times_hp,meta_w,
                                                       test_ratio,load_model,load_model_path)
                except:
                    st.info("参数错误请检查")
                fcol1, fcol2,= st.columns(2)
                if fig1 is not None:
                    with fcol1:
                        html1 = st.plotly_chart(fig1)
                        st.text(res_txt)
                    with fcol2:
                        html2 = st.plotly_chart(fig2)
            if pred_bt:
                try:
                    pred_res =  ml_pred(load_model,load_model_path, meta_w, test)
                except:
                    st.info("参数错误请检查")
                try:
                    st.dataframe(pred_res,height=int(35.2*12))
                except:
                    st.dataframe(pd.DataFrame())
if __name__ == "__main__":
    ML_trainer()
