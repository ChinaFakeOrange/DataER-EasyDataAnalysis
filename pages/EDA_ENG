
# -*- ecoding: utf-8 -*-
# @Author: NUO


import plotly.express as px
import streamlit as st
import pandas as pd
import sys
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
#Setting up web app page

def EDA():
    st.set_page_config(page_title='DataER', page_icon=None, layout="wide")
    st.header('DataER')
    st.write("**Click Browse Files for reading your file！**")
    st.divider()
    # Creating section in sidebar
    st.sidebar.write("Upload file")
    # User prompt to select file type
    ft = st.sidebar.selectbox("Select File type", ["Excel", "csv"])

    # Creating dynamic file upload option in sidebar
    uploaded_file = st.sidebar.file_uploader("Here for file upload")

    if uploaded_file is not None:
        file_path = uploaded_file

        if ft == 'Excel':
            try:
                # User prompt to select sheet name in uploaded Excel
                sh = st.sidebar.selectbox("*Select the sheet you want to import*",
                                          pd.ExcelFile(file_path).sheet_names)
                # User prompt to define row with column names if they aren't in the header row in the uploaded Excel
                h = st.sidebar.number_input("*Which row is the column head?*", 0, 100)
            except:
                st.info("It is not a Excel File")
                sys.exit()

        elif ft == 'csv':
            try:
                # No need for sh and h for csv, set them to None
                sh = None
                h = None
            except:
                st.info("It is not a CSV File")
                sys.exit()


        # Caching function to load data
        @st.cache_data(experimental_allow_widgets=True)
        def load_data(file_path, ft, sh, h):
            if ft == 'Excel':
                try:
                    data = pd.read_excel(file_path, header=h, sheet_name=sh, engine='openpyxl')
                except:
                    st.info("It is not a Excel File")
                    sys.exit()
            elif ft == 'csv':
                try:
                    data = pd.read_csv(file_path)
                except:
                    st.info("It is not a CSV File")
                    sys.exit()
            return data


        data = load_data(file_path, ft, sh, h)

        # =====================================================================================================
        ## 1. Overview of the data
        st.write('### Data Preview')

        try:
            # View the dataframe in streamlit
            st.data_editor(data, use_container_width=True)
            st.write('###### Data Dimension (row,col) :', data.shape)

        except:
            st.info("File Reading Error")
            sys.exit()
        ov_select = st.sidebar.checkbox("**Data Analysis Tools**",value=True)

        if ov_select:
        ## 2. Understanding the data
            # st.write( '### 数据信息综合 ')

            #Creating radio button and sidebar simulataneously
            selected1 = st.sidebar.checkbox( "Data Type")
            selected2 = st.sidebar.checkbox("Statistical Data")
            selected3 = st.sidebar.checkbox("Categorical Distribution")

            if selected1 or selected2 or selected3:
                st.write('### Data Overview ')

            #Showing field types
            if selected1:# == '数据类型':
                fd = data.dtypes.reset_index().rename(columns={'index':'Field Name',0:'Field Type'}).sort_values(by='Field Type',ascending=False).reset_index(drop=True)
                st.dataframe(fd, use_container_width=True)

            #Showing summary statistics
            if selected2:# == '统计学信息':
                # idx = ['数量', '唯一数', "高频", "高频次", "均值", "方差", "最小值", "最大值"]
                # oidx = ['count','unique','top','freq','mean','std','min','max']
                ss = pd.DataFrame(data.describe(include='all').round(2).fillna(''))
                # ss.rename(index=dict(zip(oidx,idx)),inplace=True)
                st.dataframe(ss, use_container_width=True,height=int(35.2*(12)))

            #Showing value counts of object fields
            if selected3:# == '数值统计分布':
                sub_selected = st.sidebar.selectbox(
                    "Select the variable you want to explore",data.select_dtypes('object').columns,  # 也可以用元组
                    index=1)

                # sub_selected = st.sidebar.radio( "你想要了解哪一个变量？",data.select_dtypes('object').columns)
                vc = data[sub_selected].value_counts().reset_index().rename(columns={'count':'count'}).reset_index(drop=True)
                st.dataframe(vc, use_container_width=True)
        st.divider()

    ## 3. Visualisation
        vis_select = st.sidebar.checkbox("**Data Visualization**")
        fig = None
        if vis_select:
            st.write( '### Data Visualization')
            # plot_selected =st.sidebar.selectbox('选择图像种类',['散点图','棒状图','散点矩阵',"数据缺失图",'变量关系','趋势图'],index=0)
            plot_selected =st.sidebar.selectbox('Select Plot Type',['Scatter','Bar','ScatterMatrix',"MissingData",'Correlation','HistogramDistribution'],index=0)
            if plot_selected == '散点图' or plot_selected == 'Scatter':
                x_selected = st.sidebar.selectbox(
                    "X Axis", data.select_dtypes(['int','float']).columns,  # 也可以用元组
                    index=0, )
                y_selected = st.sidebar.selectbox(
                    "Y Axis", data.select_dtypes(['int','float']).columns,  # 也可以用元组
                    index=0)
                z_selected = st.sidebar.selectbox(
                    "Z Axis", data.select_dtypes(['object']).columns,  # 也可以用元组
                    index=0)
                fig = px.scatter(data, x=x_selected, y=y_selected, color=z_selected)

            elif plot_selected == '棒状图' or plot_selected == 'Bar':
                x_selected = st.sidebar.selectbox(
                    "X Axis", data.select_dtypes('object').columns)
                barp=data[x_selected].value_counts().reset_index().rename(columns={'count':'count'}).reset_index(drop=True)
                fig=px.bar(barp, x=x_selected,y='count',)

            elif plot_selected == '散点矩阵' or plot_selected == 'ScatterMatrix':
                data_drop = data.dropna(axis=1)
                xs = data_drop.select_dtypes(['int', 'float']).columns
                x_selected = st.sidebar.multiselect(
                    "X Axis", xs,default =[xs[0],xs[1]])
                z_selected = st.sidebar.selectbox(
                    "Z Axis", data_drop.select_dtypes('object').columns,)
                fig = px.scatter_matrix(data_drop, dimensions=x_selected,color =z_selected)
                fig.update_traces(diagonal_visible=False,showupperhalf=False)

            elif plot_selected == '变量关系' or plot_selected == 'Correlation':
                data_cor = data[[col for col in data.columns if data[col].dtype != 'object']]
                r = data_cor.corr('pearson')
                mask = np.triu(np.ones_like(r, dtype=bool))
                rLT = r.mask(mask)
                fig = go.Figure(go.Heatmap(z=rLT,x=rLT.columns.values,y=rLT.columns.values,
                    zmin=- 0.25,zmax=1,xgap=1, ygap=1,colorscale='Agsunset',
                ))

                fig.update_layout(
                        title='Correlation',title_x=0.5, xaxis_showgrid=False,
                        yaxis_showgrid=False,yaxis_autorange='reversed',
                        # paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',hovermode='closest'
                )

            elif plot_selected == '数据缺失图' or plot_selected == 'MissingData':
                percent_missing = (data.isna().sum() / data.shape[0]) * 100
                num_values = len(data) - data.isna().sum()
                missing_df = pd.DataFrame({'Number of Values': num_values,
                                           'Missing Percentage (%)': percent_missing}).reset_index().sort_values(
                    by='Missing Percentage (%)')
                missing_df = missing_df.loc[(missing_df != 0).all(axis=1)]
                fig = go.Figure(go.Bar(
                    y=missing_df['index'],
                    x=missing_df['Missing Percentage (%)'],
                    textposition='auto',
                    text=[f"{value:.2f}%" for value in missing_df['Missing Percentage (%)']],
                    orientation='h'))

                # Customize plot layout
                fig.update_layout(
                    title="Missing Data",
                    xaxis_title='Missing (%)',
                    yaxis_title='Variables',
                )
            elif plot_selected == '趋势图' or plot_selected =='HistogramDistribution' :
                xs = data.select_dtypes(['int', 'float']).columns
                x_selected = st.sidebar.multiselect(
                    "X Axis", xs, default=[xs[0]])
                bin_sd = st.slider(label="**Interval**",min_value=0,max_value=100,value=10)
                dis_data = [data[i].values.tolist() for i in x_selected]
                if x_selected:
                    fig = ff.create_distplot(dis_data,x_selected, bin_size=bin_sd)
                    fig.update_layout(yaxis_title='Percentage %')

            if fig:
                fig.update_layout(height=600,width=1000)
                st.plotly_chart(fig)
if __name__ == "__main__":
    EDA()
